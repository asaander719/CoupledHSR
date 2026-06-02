# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import numpy as np
import random


class CoupledHamiltonianBlock(nn.Module):
    """Multi-behavior extension of HSR's HamiltonianBlock.

    HSR uses a scalar per-channel impedance D(w)=k-m w^2+i c w (independent
    oscillators). We make the impedance a B x B per-frequency COUPLED system
    across behavior types, with directional (causal) cross-behavior coupling:
    force in a lower funnel stage (view) transfers to higher stages (cart, buy).

    coupling_mode: 'none' -> identical to HSR (the C=0 ablation)
                   'symmetric' -> undirected coupling
                   'causal' -> lower-triangular (view->fav->cart->buy)   [ours]
    """
    def __init__(self, d_model, max_len, kernel_size, num_behaviors,
                 coupling_mode, dropout):
        super().__init__()
        self.d_model = d_model
        self.B = num_behaviors
        self.coupling_mode = coupling_mode
        self.max_len = max_len
        N_f = max_len // 2 + 1

        self.norm1 = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 3 * d_model)

        # Per-(behavior, channel) physical params  (was per-channel in HSR)
        self.m_raw = nn.Parameter(torch.zeros(self.B, d_model))
        self.c_raw = nn.Parameter(torch.zeros(self.B, d_model))
        self.k_raw = nn.Parameter(torch.zeros(self.B, d_model))

        self.psi_re = nn.Parameter(torch.ones(self.B, d_model, N_f))
        self.psi_im = nn.Parameter(torch.zeros(self.B, d_model, N_f))

        # Cross-behavior coupling (our contribution). Zero-init => starts at HSR.
        if coupling_mode != 'none':
            self.coupling = nn.Parameter(torch.zeros(self.B, self.B, d_model))
            if coupling_mode == 'causal':
                cmask = torch.zeros(self.B, self.B)
                # funnel_order: behavior IDs from EARLIEST to LATEST funnel stage.
                # Set per dataset in config. IDs not listed (e.g. mask/pad) get no coupling.
                ## retail
                # funnel = [2, 1, 3] 
                ## Tmall
                # funnel_order= [0, 3, 1, 2] 

                funnel = [2, 1, 3]
                for k in range(len(funnel)):
                    for j in range(k):
                        tgt, src = funnel[k], funnel[j]   # src is earlier, drives later tgt
                        if tgt < self.B and src < self.B:
                            cmask[tgt, src] = 1.0
            elif coupling_mode == 'symmetric':
                cmask = torch.ones(self.B, self.B) - torch.eye(self.B)
                cmask[0, :] = 0.0; cmask[:, 0] = 0.0
            self.register_buffer('cmask', cmask.unsqueeze(-1))  # (B,B,1)

        omega = 2 * math.pi * torch.arange(N_f).float() / max_len
        self.register_buffer('omega', omega)

        self.kernel_size = kernel_size
        self.impulse_conv = nn.Conv1d(d_model, d_model, kernel_size,
                                      groups=d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.Dropout(dropout))

    def _coupled_propagate(self, F_br, behavior_ids):
        """F_br:(N,T,d) force; behavior_ids:(N,T) in [0,B-1]."""
        N, T, d = F_br.shape
        Bn = self.B
                # Normalize behavior ids to [0,B-1] if they are token ids starting from 1.
        behavior_ids = behavior_ids.long()
        if behavior_ids.max() >= Bn and behavior_ids.min() >= 0:
            behavior_ids = (behavior_ids - 1).clamp(min=0)


        # print(f"behavior_ids range: min={behavior_ids.min().item()} max={behavior_ids.max().item()} shape={behavior_ids.shape}")

        # Route each step's force to its behavior channel -> (N,B,T,d)
        onehot = F.one_hot(behavior_ids.long(), num_classes=Bn).to(F_br.dtype)
        force_b = torch.einsum('ntb,ntd->nbtd', onehot, F_br).contiguous()

        # FFT along time
        try:
            F_hat = torch.fft.rfft(force_b, n=self.max_len, dim=2)   # (N,B,Nf,d)
        except RuntimeError as e:
            if 'CUFFT_INTERNAL_ERROR' in str(e):
                force_b = force_b.cpu().contiguous()
                F_hat = torch.fft.rfft(force_b, n=self.max_len, dim=2).to(F_br.device)
            else:
                raise
        Nf = F_hat.shape[-2]

        w = self.omega[:Nf].view(1, Nf, 1)                       # (1,Nf,1)
        m = F.softplus(self.m_raw).unsqueeze(1) + 1e-1           # (B,1,d)
        c = F.softplus(self.c_raw).unsqueeze(1) + 1e-2
        k = F.softplus(self.k_raw).unsqueeze(1) + 1e-2

        D_diag = torch.complex(k - m * w.pow(2), c * w)          # (B,Nf,d)
        psi = torch.complex(self.psi_re[:, :, :Nf], self.psi_im[:, :, :Nf])  # (B,d,Nf)
        psi = psi.permute(0, 2, 1)                               # (B,Nf,d)
        rhs = psi.unsqueeze(0) * F_hat                           # (N,B,Nf,d)

        if self.coupling_mode == 'none':
            Q_hat = rhs / D_diag.unsqueeze(0)                    # HSR path
        else:
            # Build (N,Nf,d,B,B) impedance with the batch dim from the start.
            Dm = torch.zeros(N, Nf, d, Bn, Bn, dtype=torch.cfloat, device=F_br.device)
            di = torch.arange(Bn)
            # diagonal: D_diag is (B,Nf,d) -> (Nf,d,B) -> broadcast over N
            Dm[:, :, :, di, di] = D_diag.permute(1, 2, 0).unsqueeze(0).expand(N, -1, -1, -1)
            # off-diagonal coupling: (B,B,d) masked -> (d,B,B) -> (1,1,d,B,B)
            coup = (self.coupling * self.cmask).to(torch.cfloat).permute(2, 0, 1)  # (d,B,B)
            Dm = Dm + coup.view(1, 1, d, Bn, Bn)
            # numerical regularizer on the diagonal
            Dm = Dm + 1e-4 * torch.eye(Bn, dtype=torch.cfloat, device=F_br.device)
            # RHS -> (N,Nf,d,B,1)
            b = rhs.permute(0, 2, 3, 1).unsqueeze(-1)            # (N,Nf,d,B,1)
            Q = torch.linalg.solve(Dm, b)                        # (N,Nf,d,B,1)
            Q_hat = Q.squeeze(-1).permute(0, 3, 1, 2)            # (N,B,Nf,d)

        Q = torch.fft.irfft(Q_hat, n=self.max_len, dim=2)[:, :, :T, :]  # (N,B,T,d)
        Q_seq = torch.einsum('ntb,nbtd->ntd', onehot, Q)
        return Q_seq

    def forward(self, x, mask, behavior_ids):
        h = self.norm1(x)
        F_br, U_br, A_br = self.in_proj(h).chunk(3, dim=-1)
        F_br, U_br = F.gelu(F_br), F.gelu(U_br)
        g = torch.sigmoid(A_br)

        Q_tilde = self._coupled_propagate(F_br, behavior_ids)

        U_pad = F.pad(U_br.transpose(1, 2), (self.kernel_size - 1, 0))
        H_loc = self.impulse_conv(U_pad).transpose(1, 2)

        Z = (Q_tilde + H_loc) * g
        x = x + self.dropout1(self.out_proj(Z))
        x = x * mask
        x = x + self.ffn(x)
        x = x * mask
        return x


class HSR(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.buy_type = dataset.field2token_id["item_type_list"]['0']

        # load dataset info
        self.mask_token = self.n_items
        self.mask_ratio = config['mask_ratio']
        self.hidden_size = config['hidden_size']
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.type_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        self.initializer_range = config['initializer_range']

        self.n_layers = config['num_layers']
        self.max_len = config['MAX_ITEM_LIST_LENGTH']
        self.dropout_prob = config['dropout_prob']
        self.kernel_size = config['kernel_size']
        # self.num_behaviors = config['num_behaviors']
        self.coupling_mode = config['coupling_mode']
        # Behavior-type sequence field. RecBole names it <base_field>_list.

        self.BEHAVIOR_SEQ = config['BEHAVIOR_FIELD'] if config['BEHAVIOR_FIELD'] is not None else 'item_type_list'
        behavior_tokens = dataset.field2token_id[self.BEHAVIOR_SEQ]
        pad_count = 1 if '[PAD]' in behavior_tokens else 0
        behavior_classes = len(behavior_tokens) - pad_count
        self.num_behaviors = behavior_classes
        if 'num_behaviors' in config and config['num_behaviors'] != self.num_behaviors:
            print(
                f"Warning: config num_behaviors={config['num_behaviors']} does not match dataset behavior classes={self.num_behaviors}. "
                "Using dataset-derived value."
            )


        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len + 1, self.hidden_size)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.emb_norm = nn.LayerNorm(self.hidden_size)

        self.blocks = nn.ModuleList([
            CoupledHamiltonianBlock(self.hidden_size, self.max_len, self.kernel_size,
                                    self.num_behaviors, self.coupling_mode, self.dropout_prob)
            for _ in range(self.n_layers)])
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.inv_mass_dt = nn.Parameter(torch.zeros(self.hidden_size))
        self.momentum_dropout = nn.Dropout(self.dropout_prob)
        omega = 2 * math.pi * torch.arange(self.max_len // 2 + 1).float() / self.max_len
        self.register_buffer('omega_final', omega)

        self.loss_type = config['loss_type']
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' is one of ['BPR', 'CE'] for HSR.")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    # def _init_weights(self, module):
    #     """ Initialize the weights """
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.initializer_range)
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def forward(self, item_seq, item_seq_len, behavior_seq, return_sequence=False):
        B, T = item_seq.shape
        item_emb = self.item_embedding(item_seq)
        positions = torch.arange(T, device=item_seq.device).unsqueeze(0).expand(B, T)
        mask = (item_seq != 0).float().unsqueeze(-1)
        x = self.emb_norm(self.emb_dropout(item_emb + self.position_embedding(positions)))
        x = x * mask

        for block in self.blocks:
            x = block(x, mask, behavior_seq)

        x = self.final_norm(x)
        if return_sequence:
            return x

        last_idx = (item_seq != 0).cumsum(dim=1).argmax(dim=1)
        idx = torch.arange(B, device=x.device)
        q_L = x[idx, last_idx]

        # FIX (Bug 1): velocity from the paper's Eq.18  P̂ = m·(iω)·Q̂,
        # WITHOUT the undocumented exp(-ω) decay. Damping already lives in the
        # propagator; re-applying it here double-counts and is not in the paper.
        X_hat = torch.fft.rfft(x.transpose(1, 2), dim=-1)
        Nf = X_hat.shape[-1]
        omega_exp = self.omega_final[:Nf].view(1, 1, Nf)
        V_hat = (1j * omega_exp) * X_hat                      # removed freq_decay
        v_seq = torch.fft.irfft(V_hat, n=T, dim=-1).transpose(1, 2)
        v_L = v_seq[idx, last_idx]

        v_L_dropped = self.momentum_dropout(v_L)
        q_hat = q_L + v_L_dropped * self.inv_mass_dt
        return q_hat

    def _score(self, q_hat, item_emb):
        if item_emb.dim() == 2 and item_emb.shape[0] != q_hat.shape[0]:
            return torch.matmul(q_hat, item_emb.t())
        return (q_hat * item_emb).sum(dim=-1)

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]                 [[ 0,  0,  0,  ..., 44, 47, 49],...[ 0,  0,  0,  ...,  0,  0,  1]]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1) #torch.Size([2560]) 
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot #torch.Size([2560, 200])
    
    def reconstruct_train_data(self, item_seq, type_seq, last_buy):
        """
        Mask item sequence for training.
        """
        last_buy = last_buy.tolist()
        device = item_seq.device
        batch_size = item_seq.size(0)

        zero_padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)
        item_seq = torch.cat((item_seq, zero_padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        type_seq = torch.cat((type_seq, zero_padding.unsqueeze(-1)), dim=-1)
        n_objs = (torch.count_nonzero(item_seq, dim=1)+1).tolist()
        for batch_id in range(batch_size):
            n_obj = n_objs[batch_id]
            item_seq[batch_id][n_obj-1] = last_buy[batch_id]
            type_seq[batch_id][n_obj-1] = self.buy_type

        sequence_instances = item_seq.cpu().numpy().tolist()
        type_instances = type_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        masked_index = []

        for instance_idx, instance in enumerate(sequence_instances):
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if index_id == n_objs[instance_idx]-1:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)
            #The list of masked items (pos_items) and their indices (masked_index) are padded to a fixed length (self.mask_item_length).
            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        #pos_items.size() = torch.Size([64, 40])
        # [B mask_len] #[[ 0,     0,     0,  ...,     0, 16308,  1998],[    0,     0,     0,  ...,  7592,  3486,  8531]]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        type_instances = torch.tensor(type_instances, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence, pos_items, masked_index, type_instances

    def reconstruct_test_data(self, item_seq, item_seq_len, item_type):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        item_type = torch.cat((item_type, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        return item_seq, item_type

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        behavior_seq = interaction[self.BEHAVIOR_SEQ]
        q_hat = self.forward(item_seq, item_seq_len, behavior_seq)
        pos = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            try:
                neg = interaction[self.NEG_ITEM_ID]
            except KeyError:
                neg = None
            if neg is None:
                neg = torch.randint(1, self.n_items, (q_hat.shape[0],), device=q_hat.device)
            return self.loss_fct(self._score(q_hat, self.item_embedding(pos)),
                                 self._score(q_hat, self.item_embedding(neg)))

        elif self.loss_type == "CE":
            item_type = interaction["item_type_list"]
            last_buy = interaction["item_id"]
            masked_item_seq, pos_items, masked_index, item_type_seq = self.reconstruct_train_data(item_seq, item_type, last_buy)

            seq_output = self.forward(masked_item_seq, torch.count_nonzero(masked_item_seq, dim=1), item_type_seq, return_sequence=True)
            pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
            pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]
            seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

            test_item_emb = self.item_embedding.weight[:self.n_items]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num]
            targets = (masked_index > 0).view(-1).long()
            loss = torch.sum(self.loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets.float()) / torch.sum(targets.float())
            return loss



    # def full_sort_predict(self, interaction):
    #     q_hat = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN],
    #                          interaction[self.BEHAVIOR_SEQ])
    #     return self._score(q_hat, self.item_embedding.weight)

    # def predict(self, interaction):
    #     q_hat = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN],
    #                          interaction[self.BEHAVIOR_SEQ])
    #     return self._score(q_hat, self.item_embedding(interaction[self.ITEM_ID]))

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        behavior_seq = interaction[self.BEHAVIOR_SEQ]
        q_hat = self.forward(item_seq, item_seq_len, behavior_seq)
        return self._score(q_hat, self.item_embedding(interaction[self.ITEM_ID]))

    def full_sort_predict(self, interaction):
        item_seq = interaction['item_id_list']
        type_seq = interaction['item_type_list']
        item_seq_len = torch.count_nonzero(item_seq, 1)
        item_seq, type_seq = self.reconstruct_test_data(item_seq, item_seq_len, type_seq)
        seq_output = self.forward(item_seq, item_seq_len, type_seq, return_sequence=True)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores

    def customized_sort_predict(self, interaction):
        item_seq = interaction['item_id_list']
        type_seq = interaction['item_type_list']
        truth = interaction['item_id']
        if self.dataset == "ijcai_beh":
            raw_candidates = [73, 3050, 22557, 5950, 4391, 6845, 1800, 2261, 13801, 2953, 4164, 32090, 3333, 44733, 7380, 790, 1845, 2886, 2366, 21161, 6512, 1689, 337, 3963, 3108, 715, 169, 2558, 6623, 888, 6708, 3585, 501, 308, 9884, 1405, 5494, 6609, 7433, 25101, 3580, 145, 3462, 5340, 1131, 6681, 7776, 8678, 52852, 19229, 4160, 33753, 4356, 920, 15312, 43106, 16669, 1850, 2855, 43807, 15, 8719, 89, 3220, 36, 2442, 9299, 8189, 701, 300, 526, 4564, 516, 1184, 178, 2834, 16455, 9392, 22037, 344, 15879, 3374, 2984, 3581, 11479, 6927, 779, 5298, 10195, 39739, 663, 9137, 24722, 7004, 7412, 89534, 2670, 100, 6112, 1355]
        elif self.dataset == "retail_beh":
            raw_candidates = [101, 11, 14, 493, 163, 593, 1464, 12, 297, 123, 754, 790, 243, 250, 508, 673, 1161, 523, 41, 561, 2126, 196, 1499, 1093, 1138, 1197, 745, 1431, 682, 1567, 440, 1604, 145, 1109, 2146, 209, 2360, 426, 1756, 46, 1906, 520, 3956, 447, 1593, 1119, 894, 2561, 381, 939, 213, 1343, 733, 554, 2389, 1191, 1330, 1264, 2466, 2072, 1024, 2015, 739, 144, 1004, 314, 1868, 3276, 1184, 866, 1020, 2940, 5966, 3805, 221, 11333, 5081, 685, 87, 2458, 415, 669, 1336, 3419, 2758, 2300, 1681, 2876, 2612, 2405, 585, 702, 3876, 1416, 466, 7628, 572, 3385, 220, 772]
        elif self.dataset == "tmall_beh":
            raw_candidates = [2544, 7010, 4193, 32270, 22086, 7768, 647, 7968, 26512, 4575, 63971, 2121, 7857, 5134, 416, 1858, 34198, 2146, 778, 12583, 13899, 7652, 4552, 14410, 1272, 21417, 2985, 5358, 36621, 10337, 13065, 1235, 3410, 14180, 5083, 5089, 4240, 10863, 3397, 4818, 58422, 8353, 14315, 14465, 30129, 4752, 5853, 1312, 3890, 6409, 7664, 1025, 16740, 14185, 4535, 670, 17071, 12579, 1469, 853, 775, 12039, 3853, 4307, 5729, 271, 13319, 1548, 449, 2771, 4727, 903, 594, 28184, 126, 27306, 20603, 40630, 907, 5118, 3472, 7012, 10055, 1363, 9086, 5806, 8204, 41711, 10174, 12900, 4435, 35877, 8679, 10369, 2865, 14830, 175, 4434, 11444, 701]
        customized_candidates = list()
        for batch_idx in range(item_seq.shape[0]):
            seen = item_seq[batch_idx].cpu().tolist()
            cands = raw_candidates.copy()
            for i in range(len(cands)):
                if cands[i] in seen:
                    new_cand = random.randint(1, self.n_items)
                    while new_cand in seen:
                        new_cand = random.randint(1, self.n_items)
                    cands[i] = new_cand
            cands.insert(0, truth[batch_idx].item()) 
            customized_candidates.append(cands)
        candidates = torch.LongTensor(customized_candidates).to(item_seq.device)
        item_seq_len = torch.count_nonzero(item_seq, 1)
        item_seq, type_seq = self.reconstruct_test_data(item_seq, item_seq_len, type_seq)
        seq_output = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN],
                             interaction[self.BEHAVIOR_SEQ])
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_items_emb = self.item_embedding(candidates)  # delete masked token
        scores = torch.bmm(test_items_emb, seq_output.unsqueeze(-1)).squeeze()  # [B, item_num]
        return scores
