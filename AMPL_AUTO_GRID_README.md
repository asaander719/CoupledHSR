# AMPL/CoupledHSR 一键 Grid Fine-tuning

这个 runner 的设计目标是：
1. 先只在 `purchase` behavior 上做 hyperparameter grid search；
2. 自动选择 valid metric 最好的参数；
3. 可选：用最佳参数继续跑其它 behaviors；
4. 所有 run 的 accuracy、memory、latency、params、FLOPs(optional) 都保存到 CSV/JSONL。

## 1. 放置文件

```bash
cp run_ampl_auto_grid.py /home/asaliao/RecBole/run_ampl_auto_grid.py
cp CoupledHSR_AMPL_GRID.yaml /home/asaliao/RecBole/configs/model/CoupledHSR_AMPL_GRID.yaml
cp AMPL_GRID.yaml /home/asaliao/RecBole/configs/model/AMPL_GRID.yaml
```

## 2. 只跑 purchase grid search

```bash
python run_ampl_auto_grid.py \
  --model CoupledHSR \
  --dataset_prefix ampl_jd \
  --config_files configs/model/CoupledHSR_AMPL_GRID.yaml \
  -g 0 \
  --tune_behavior purchase \
  --stats_level quick
```

如果 YAML 是：

```yaml
hyper_parameters: ['learning_rate']
learning_rate: [0.001, 0.0005]
```

那么它会自动跑两次：
- `learning_rate=0.001`
- `learning_rate=0.0005`

## 3. Purchase 找到最佳参数后自动跑所有 behavior

```bash
python run_ampl_auto_grid.py \
  --model CoupledHSR \
  --dataset_prefix ampl_jd \
  --config_files configs/model/CoupledHSR_AMPL_GRID.yaml \
  -g 0 \
  --tune_behavior purchase \
  --run_all_after_tune \
  --reuse_tuned_behavior \
  --stats_level quick
```

JD 默认 behavior 顺序：

```text
click, comment, favourite, purchase
```

UB 默认 behavior 顺序：

```text
click, cart, favourite, purchase
```

UB 运行：

```bash
python run_ampl_auto_grid.py \
  --model CoupledHSR \
  --dataset_prefix ampl_ub \
  --config_files configs/model/CoupledHSR_AMPL_GRID.yaml \
  -g 0 \
  --tune_behavior purchase \
  --run_all_after_tune \
  --reuse_tuned_behavior \
  --stats_level quick
```

## 4. 指定 behavior 列表

```bash
python run_ampl_auto_grid.py \
  --model CoupledHSR \
  --dataset_prefix ampl_jd \
  --config_files configs/model/CoupledHSR_AMPL_GRID.yaml \
  -g 0 \
  --tune_behavior purchase \
  --run_all_after_tune \
  --behaviors click,comment,favourite,purchase \
  --stats_level quick
```

## 5. 查看所有将要跑的组合，但不训练

```bash
python run_ampl_auto_grid.py \
  --model CoupledHSR \
  --dataset_prefix ampl_jd \
  --config_files configs/model/CoupledHSR_AMPL_GRID.yaml \
  -g 0 \
  --dry_run
```

## 6. 输出文件

每次实验会保存到：

```text
saved/ampl_grid/<model>/<dataset_prefix>/<tune_behavior_timestamp>/
```

主要文件：

```text
tune_summary.csv              # purchase grid search 的所有结果
best_hyperparameters.json     # purchase 上最优参数
best_hyperparameters.yaml     # 方便复制到 YAML 的最优参数
best_all_summary.csv          # 用最优参数跑所有 behavior 的结果
summary.csv                   # 全部记录汇总
run_records.jsonl             # 每个 run 的完整 JSON 记录
args.json                     # 命令行参数
```

## 7. 更大的 grid 示例

```yaml
hyper_parameters: ['learning_rate', 'hidden_size', 'num_layers', 'dropout_prob', 'hnn_residual_init']
learning_rate: [0.001, 0.0005]
hidden_size: [50, 64]
num_layers: [1, 2]
dropout_prob: [0.3, 0.5]
hnn_residual_init: [-6.0, -5.0]
```

这个会跑：

```text
2 × 2 × 2 × 2 × 2 = 32 runs
```

## 8. AMPL baseline 也可以同样跑

```bash
python run_ampl_auto_grid.py \
  --model AMPL \
  --dataset_prefix ampl_jd \
  --config_files configs/model/AMPL_GRID.yaml \
  -g 0 \
  --tune_behavior purchase \
  --run_all_after_tune \
  --reuse_tuned_behavior \
  --stats_level quick
```
