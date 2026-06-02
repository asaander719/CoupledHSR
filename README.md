## Repo of CoupledHSR for multi-task multi-behavior seq. recommendation

Re-convert AMPL datasets by 
```bash
python convert_ampl_to_recbole_seq.py \
  --src_dir AMPL/processed_data_JD \
  --dataset_key JD \
  --out_root RecBole/dataset/JD-50 \
  --prefix ampl_jd \
  --max_len 50
```
[ampl_jd_click] train=59,670 valid=4,053 test=2,268 target_behavior=click:0
[ampl_jd_comment] train=24,161 valid=1,841 test=895 target_behavior=comment:1
[ampl_jd_favourite] train=7,358 valid=451 test=243 target_behavior=favourite:2
[ampl_jd_purchase] train=209,107 valid=2,609 test=1,297 target_behavior=purchase:3

```bash
python convert_ampl_to_recbole_seq.py \
  --src_dir AMPL/processed_data_UB \
  --dataset_key UB \
  --out_root RecBole/dataset/UB-50 \
  --prefix ampl_ub \
  --max_len 50
```
[ampl_ub_click] train=104,064 valid=13,901 test=11,375 target_behavior=click:0
[ampl_ub_cart] train=66,334 valid=4,724 test=3,773 target_behavior=cart:1
[ampl_ub_favourite] train=21,657 valid=1,648 test=1,340 target_behavior=favourite:2
[ampl_ub_purchase] train=496,488 valid=7,719 test=6,126 target_behavior=purchase:3

## Run one behavior
For JD purchase:
```bash
python run_ampl_seq_full.py \
  --model CoupledHSR \
  --dataset ampl_jd_purchase \
  --config_files configs/model/CoupledHSR.yaml \
  -g 0
```

For UB purchase:
```bash
python run_ampl_seq_full.py \
  --model CoupledHSR \
  --dataset ampl_ub_purchase \
  --config_files configs/model/CoupledHSR.yaml \
  -g 0
```

## Run all behaviors

JD:
```bash
for b in click comment favourite purchase; do
  python run_ampl_seq_full.py \
    --model CoupledHSR \
    --dataset ampl_jd_${b} \
    --config_files configs/model/CoupledHSR.yaml \
    -g 0
done
```

UB:
```bash
for b in click cart favourite purchase; do
  python run_ampl_seq_full.py \
    --model CoupledHSR \
    --dataset ampl_ub_${b} \
    --config_files configs/model/CoupledHSR.yaml \
    -g 0
done
```

## Reference numbers from AMPL:

| Dataset | Behavior            | SASRec Rec@10 | AMPL Rec@10 | AMPL NDCG@10 |
| ------- | ------------------- | ------------: | ----------: | -----------: |
| JD      | click / examination |        0.0816 |      0.0809 |       0.0371 |
| JD      | comment             |        0.5198 |      0.6056 |       0.4200 |
| JD      | favourite           |        0.1297 |      0.1383 |       0.0634 |
| JD      | purchase            |        0.1507 |      0.1700 |       0.0847 |
| UB      | click / examination |        0.0323 |      0.0408 |       0.0262 |
| UB      | cart                |        0.0226 |      0.0286 |       0.0156 |
| UB      | favourite           |        0.0200 |      0.0284 |       0.0158 |
| UB      | purchase            |        0.0659 |      0.0975 |       0.0662 |


## Important protocol note
AMPL’s Appendix says their preprocessing removes duplicate (user, item, behavior) records, filters low-purchase items and users, sorts by timestamp, uses 80/10/10 temporal splitting, keeps only the first occurrence of each behavior in validation/test, removes users without purchase in train, and removes cold-start items from validation/test .

The converter assumes you already have their processed files, so it does not redo those filters. It only converts AMPL CSVs into RecBole atomic format.


## hyperparameter sweep + performance stats
''run_ampl_seq_sweep.py'' supports：
```yaml
hyper_parameters: ['hidden_size', 'dropout_prob', 'learning_rate']
  hidden_size: [64, 128]
  dropout_prob: [0.3, 0.5]
  learning_rate: [0.001, 0.0005]
```

Performance stats record:
  best_valid_result
  test_result
  fit_sec
  test_sec
  params
  GPU memory
  latency
  optional FLOPs

## For baseline comparison 
AMPL baseline
```bash
python run_ampl_seq_sweep.py \
  --model AMPL \
  --dataset ampl_jd_purchase \
  --config_files configs/model/AMPL.yaml \
  -g 4 \
  --stats_level quick
```

CoupledHSR:
```bash
python run_ampl_seq_sweep.py \
  --model CoupledHSR \
  --dataset ampl_jd_purchase \
  --config_files configs/model/CoupledHSR_AMPL_SEQ.yaml \
  -g 4 \
  --stats_level quick
```

## For full stats + FLOPs
```bash
python run_ampl_seq_sweep.py \
  --model AMPL \
  --dataset ampl_jd_purchase \
  --config_files configs/model/AMPL.yaml \
  -g 4 \
  --stats_level full \
  --compute_flops
```

