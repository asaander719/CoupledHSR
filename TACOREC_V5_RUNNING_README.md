# TacoRec / CoupledHSR-v5 AMPL-style running guide

## 1. Install files

```bash
cp run_tacorec_v5.py /home/asaliao/RecBole/run_tacorec_v5.py
cp TacoRec_v5_GRID_SMALL.yaml /home/asaliao/RecBole/configs/model/TacoRec_v5_GRID_SMALL.yaml
cp TacoRec_v5_SAFE_FINAL.yaml /home/asaliao/RecBole/configs/model/TacoRec_v5_SAFE_FINAL.yaml
```

Make sure the v5 model is installed as:

```bash
cp CoupledHSR_v5.py /home/asaliao/RecBole/recbole/model/sequential_recommender/coupledhsr.py
rm -rf /home/asaliao/RecBole/recbole/model/sequential_recommender/__pycache__
```

## 2. Run AMPL-style final result

JD only:

```bash
python run_tacorec_v5.py \
  --mode final \
  --model CoupledHSR \
  --dataset_prefix ampl_jd \
  --behaviors auto \
  --config_files configs/model/TacoRec_v5_SAFE_FINAL.yaml \
  -g 0
```

UB only:

```bash
python run_tacorec_v5.py \
  --mode final \
  --model CoupledHSR \
  --dataset_prefix ampl_ub \
  --behaviors auto \
  --config_files configs/model/TacoRec_v5_SAFE_FINAL.yaml \
  -g 0
```

JD + UB together:

```bash
python run_tacorec_v5.py \
  --mode final \
  --model CoupledHSR \
  --dataset_prefix ampl_jd,ampl_ub \
  --behaviors auto \
  --config_files configs/model/TacoRec_v5_SAFE_FINAL.yaml \
  -g 0
```

The output directory contains:

```text
summary.csv
best_by_behavior.csv
ours_table_fillin.tex
```

Use `ours_table_fillin.tex` to fill the blank "ours" columns in Table II.

## 3. Tune purchase first

JD purchase:

```bash
python run_tacorec_v5.py \
  --mode tune \
  --model CoupledHSR \
  --dataset_prefix ampl_jd \
  --behaviors purchase \
  --config_files configs/model/TacoRec_v5_GRID_SMALL.yaml \
  -g 0
```

UB purchase:

```bash
python run_tacorec_v5.py \
  --mode tune \
  --model CoupledHSR \
  --dataset_prefix ampl_ub \
  --behaviors purchase \
  --config_files configs/model/TacoRec_v5_GRID_SMALL.yaml \
  -g 0
```

After tuning, open:

```text
saved/tacorec_v5/tune/<time>/best_by_behavior.csv
```

Pick the row with the best `valid_recall10`. Its scalar YAML config is saved in `config_file`.

Then use that scalar config to run final JD/UB behavior tasks.

## 4. Run v5 ablation

JD purchase ablation:

```bash
python run_tacorec_v5.py \
  --mode ablation \
  --model CoupledHSR \
  --dataset_prefix ampl_jd \
  --behaviors purchase \
  --config_files configs/model/TacoRec_v5_SAFE_FINAL.yaml \
  -g 0
```

UB purchase ablation:

```bash
python run_tacorec_v5.py \
  --mode ablation \
  --model CoupledHSR \
  --dataset_prefix ampl_ub \
  --behaviors purchase \
  --config_files configs/model/TacoRec_v5_SAFE_FINAL.yaml \
  -g 0
```

All behavior ablation after purchase is stable:

```bash
python run_tacorec_v5.py \
  --mode ablation \
  --model CoupledHSR \
  --dataset_prefix ampl_jd,ampl_ub \
  --behaviors auto \
  --config_files configs/model/TacoRec_v5_SAFE_FINAL.yaml \
  -g 0
```

Ablation variants:

```text
A0_anchor_only
A1_anchor_memory
A2_anchor_hsr
A3_no_coupling
A4_symmetric_coupling
A5_static_causal
A6_full_v5
```

Interpretation:

```text
A1 > A0: target-conditioned behavior memory helps.
A2 > A0: spectral Hamiltonian residual helps.
A6 > A3: cross-behavior coupling helps.
A6 > A4: causal direction helps.
A6 > A5: frequency adaptation helps.
A6 > A1 and A6 > A2: memory and spectral coupling are complementary.
```

## 5. Recommended order

1. Tune JD purchase.
2. Tune UB purchase.
3. Select a stable config with good test generalization.
4. Run final JD/UB all behaviors.
5. Run ablation on purchase.
6. Run full-behavior ablation only after purchase ablation looks reasonable.
