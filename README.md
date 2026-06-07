# Repo of TacoRec for multi-task multi-behavior seq. recommendation

Re-convert AMPL datasets by 
```bash
python convert_ampl_to_recbole_seq.py \
  --src_dir AMPL/processed_data_JD \
  --dataset_key JD \
  --out_root RecBole/dataset/JD \
  --prefix ampl_jd \
  --max_len 50 \
  --train_scope union
```

```yaml
[ampl_jd_click] raw_valid=4,053 raw_test=2,268 | seq_train=300,296 seq_valid=4,053 seq_test=2,268 target=click:0 train_scope=union
[ampl_jd_comment] raw_valid=1,841 raw_test=895 | seq_train=300,296 seq_valid=1,841 seq_test=895 target=comment:1 train_scope=union
[ampl_jd_favourite] raw_valid=451 raw_test=243 | seq_train=300,296 seq_valid=451 seq_test=243 target=favourite:2 train_scope=union
[ampl_jd_purchase] raw_valid=2,609 raw_test=1,297 | seq_train=300,296 seq_valid=2,609 seq_test=1,297 target=purchase:3 train_scope=union
[ampl_jd_union] raw_valid=28,648 raw_test=4,703 | seq_train=300,296 seq_valid=28,648 seq_test=4,703
```

```bash
python convert_ampl_to_recbole_seq.py \
  --src_dir AMPL/processed_data_UB \
  --dataset_key UB \
  --out_root RecBole/dataset/UB \
  --prefix ampl_ub \
  --max_len 50 \
  --train_scope union
```
```yaml
[ampl_ub_click] raw_valid=13,901 raw_test=11,375 | seq_train=688,543 seq_valid=13,901 seq_test=11,375 target=click:0 train_scope=union
[ampl_ub_cart] raw_valid=4,724 raw_test=3,773 | seq_train=688,543 seq_valid=4,724 seq_test=3,773 target=cart:1 train_scope=union
[ampl_ub_favourite] raw_valid=1,648 raw_test=1,340 | seq_train=688,543 seq_valid=1,648 seq_test=1,340 target=favourite:2 train_scope=union
[ampl_ub_purchase] raw_valid=7,719 raw_test=6,126 | seq_train=688,543 seq_valid=7,719 seq_test=6,126 target=purchase:3 train_scope=union
[ampl_ub_union] raw_valid=83,545 raw_test=22,614 | seq_train=688,543 seq_valid=83,545 seq_test=22,614
```

## Run one behavior
For JD click:
```bash
python run_ampl_seq_full.py \
  --model TacoRec \
  --dataset ampl_jd_click \
  --config_files configs/model/TacoRec.yaml \
  -g 0
```

For UB click:
```bash
python run_ampl_seq_full.py \
  --model TacoRec \
  --dataset ampl_ub_click \
  --config_files configs/model/TacoRec.yaml \
  -g 0
```

## Run all behaviors

JD:
```bash
python run_tacorec_v5.py \
  --mode final \
  --model TacoRec \
  --dataset_prefix ampl_jd \
  --behaviors auto \
  --config_files configs/model/TacoRec.yaml \
  -g 0
```

UB:
```bash
python run_tacorec_v5.py \
  --mode final \
  --model TacoRec \
  --dataset_prefix ampl_ub \
  --behaviors auto \
  --config_files configs/model/TacoRec.yaml \
  -g 0
```

<!-- ## Reference numbers from AMPL:

| Dataset | Behavior            | SASRec Rec@10 | AMPL Rec@10 | AMPL NDCG@10 |
| ------- | ------------------- | ------------: | ----------: | -----------: |
| JD      | click / examination |        0.0816 |      0.0809 |       0.0371 |
| JD      | comment             |        0.5198 |      0.6056 |       0.4200 |
| JD      | favourite           |        0.1297 |      0.1383 |       0.0634 |
| JD      | purchase            |        0.1507 |      0.1700 |       0.0847 |
| UB      | click / examination |        0.0323 |      0.0408 |       0.0262 |
| UB      | cart                |        0.0226 |      0.0286 |       0.0156 |
| UB      | favourite           |        0.0200 |      0.0284 |       0.0158 |
| UB      | purchase            |        0.0659 |      0.0975 |       0.0662 | -->


## Important protocol note
AMPL’s Appendix says their preprocessing removes duplicate (user, item, behavior) records, filters low-purchase items and users, sorts by timestamp, uses 80/10/10 temporal splitting, keeps only the first occurrence of each behavior in validation/test, removes users without purchase in train, and removes cold-start items from validation/test .

The converter assumes you already have their processed files, so it does not redo those filters. It only converts AMPL CSVs into RecBole atomic format.


### hyperparameter sweep + performance stats
`run_ampl_seq_sweep.py` supports：
```yaml
hyper_parameters: ['hidden_size', 'dropout_prob', 'learning_rate']
hidden_size: [64, 128]
dropout_prob: [0.3, 0.5]
learning_rate: [0.001, 0.0005]
```

Performance stats record:
 -  best_valid_result
 -  test_result
 -  fit_sec
 -  test_sec
 -  params
 -  GPU memory
 -  latency
 -  optional FLOPs


## For baseline comparison 
AMPL baseline
```bash
for b in click comment favourite purchase; do
  python run_ampl_seq_full.py \
    --model AMPL \
    --dataset ampl_jd_${b} \
    --config_files configs/model/AMPL.yaml \
    -g 0
done
```


## For full stats + FLOPs

only on click conduct grid search
```bash
python run_tacorec_v5.py \
  --mode tune \
  --model TacoRec \
  --dataset_prefix ampl_jd \
  --behaviors click \
  --config_files configs/model/TacoRec_v5_GRID_SMALL.yaml \
  -g 0
```

Samely hyperparams grid for baseline AMPL:
```bash
python run_ampl_auto_grid.py \
  --model AMPL \
  --dataset_prefix ampl_jd \
  --config_files configs/model/AMPL.yaml \
  -g 0 \
  --tune_behavior purchase \
  --run_all_after_tune \
  --reuse_tuned_behavior \
  --stats_level quick
```

