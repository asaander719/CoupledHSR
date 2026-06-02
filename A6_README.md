# CoupledHSR A6 full-ranking experiment

Files:
- `CoupledHSRA6.py`: new RecBole model. It defines both `CoupledHSRA6` and a backward-compatible alias `CoupledHSR`.
- `CoupledHSR_A6_Full.yaml`: clean full-ranking config.
- `run_a6_full.py`: runner that avoids MBHT's customized 100+1 candidate evaluation.

## Install / copy

Recommended minimal setup:

```bash
# Replace your current CoupledHSR implementation but keep the model name.
cp CoupledHSRA6.py /path/to/your/recbole/model/sequential_recommender/coupledhsr.py

# Or, if your project keeps custom model files elsewhere:
cp CoupledHSRA6.py /path/to/your/project/coupledhsr.py

cp CoupledHSR_A6_Full.yaml configs/model/CoupledHSR_A6_Full.yaml
cp run_a6_full.py run_a6_full.py
rm -rf __pycache__ recbole/model/sequential_recommender/__pycache__
```

## Run

```bash
python run_a6_full.py --model CoupledHSR --dataset retail_beh --gpu_id 0
python run_a6_full.py --model CoupledHSR --dataset tmall_beh --gpu_id 0
python run_a6_full.py --model CoupledHSR --dataset ijcai_beh --gpu_id 0 --train_batch_size 24 --eval_batch_size 64
```

## What this tests

A6 = A3 + frequency-adaptive causal coupling.

- A3 anchor: masked Transformer, MBHT-style objective, mask-token test readout.
- A6 novelty: causal Hamiltonian residual with frequency-adaptive behavior transfer.

The HNN residual starts near zero (`sigmoid(-5)=0.0067`), so early training behaves like a Transformer baseline rather than a pure physics model.

## Expected first diagnostic

If A6 is still bad, immediately run A3 by setting:

```yaml
use_hnn: False
```

Then compare:
- A3 low: objective/evaluation/data setup issue.
- A3 strong, A6 lower: HNN residual/coupling issue.
- A6 > A3: story is alive.
