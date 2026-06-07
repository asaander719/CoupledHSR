# -*- coding: utf-8 -*-
"""
Auto grid-search runner for AMPL-style RecBole sequence datasets.

目标：
1) 只输入一次命令，在 purchase behavior 上 grid 所有 hyperparameters。
2) 自动选出 best valid score 对应的最佳参数。
3) 可选：用最佳参数继续跑其它 behavior 数据集。
4) 每个 run 都记录 accuracy + params/memory/latency/FLOPs(optional)。

示例：
python run_ampl_auto_grid.py \
  --model CoupledHSR \
  --dataset_prefix ampl_jd \
  --config_files configs/model/CoupledHSR_AMPL_GRID.yaml \
  -g 0 \
  --tune_behavior purchase \
  --run_all_after_tune \
  --stats_level quick

YAML 示例：
hyper_parameters: ['learning_rate']
learning_rate: [0.001, 0.0005]

这样会先用 learning_rate=0.001 训练一次，再用 0.0005 训练一次。
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.trainer import Trainer
from recbole.utils import (
    init_logger,
    init_seed,
    get_model,
    get_environment,
    set_color,
)

try:
    from recbole.utils import get_flops
except Exception:
    get_flops = None

try:
    from recbole.utils.enum_type import ModelType
except Exception:
    from recbole.utils import ModelType


JD_BEHAVIORS = ["click", "comment", "favourite", "purchase"]
UB_BEHAVIORS = ["click", "cart", "favourite", "purchase"]

# RecBole token fields usually reserve 0 for [PAD], so behavior ids become 1..4.
DEFAULT_TARGET_TOKEN = {
    "click": 1,
    "comment": 2,
    "cart": 2,
    "favourite": 3,
    "favorite": 3,
    "purchase": 4,
}


def _config_get(config, key, default=None):
    """Old RecBole Config may not support .get()."""
    try:
        return config[key]
    except Exception:
        return default


def _plain(obj):
    """JSON-safe conversion."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="CoupledHSR")
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        required=True,
        help="Dataset prefix, e.g. ampl_jd or ampl_ub. The script will tune ampl_jd_purchase first.",
    )
    parser.add_argument("--config_files", type=str, default=None)
    parser.add_argument("--gpu_id", "-g", type=int, default=0)
    parser.add_argument("--tune_behavior", type=str, default="purchase")
    parser.add_argument(
        "--behaviors",
        type=str,
        default="auto",
        help="auto or comma list, e.g. click,comment,favourite,purchase",
    )
    parser.add_argument("--run_all_after_tune", action="store_true")
    parser.add_argument(
        "--reuse_tuned_behavior",
        action="store_true",
        help="When run_all_after_tune, do not rerun tune_behavior; reuse its best tune result in summary.",
    )
    parser.add_argument("--target_behavior_token", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--valid_metric", type=str, default=None)
    parser.add_argument("--stats_level", choices=["none", "quick", "full"], default="quick")
    parser.add_argument("--compute_flops", action="store_true")
    parser.add_argument("--results_dir", type=str, default="saved/ampl_grid")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show_progress", type=int, default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the hyperparameter combinations, do not train.",
    )
    return parser.parse_args()


def infer_behaviors(prefix: str, behaviors_arg: str) -> List[str]:
    if behaviors_arg and behaviors_arg.lower() != "auto":
        return [b.strip() for b in behaviors_arg.split(",") if b.strip()]
    low = prefix.lower()
    if "ub" in low:
        return UB_BEHAVIORS
    return JD_BEHAVIORS


def infer_target_token(behavior: str, explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    return int(DEFAULT_TARGET_TOKEN.get(behavior.lower(), 4))


def dataset_name(prefix: str, behavior: str) -> str:
    return f"{prefix}_{behavior}"


def base_config_dict(args, behavior: str) -> Dict[str, Any]:
    target_token = infer_target_token(behavior, args.target_behavior_token)
    cfg = {
        "gpu_id": args.gpu_id,
        "MODEL_TYPE": ModelType.SEQUENTIAL,

        # Sequence benchmark files created by convert_ampl_to_recbole_seq.py
        "benchmark_filename": ["train", "valid", "test"],

        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "LIST_SUFFIX": "_list",
        "MAX_ITEM_LIST_LENGTH": args.max_len,

        "alias_of_item_id": ["item_id_list"],
        "alias_of_item_type": ["item_type_list"],
        "field_separator": "\t",
        "seq_separator": " ",
        "load_col": {
            "inter": [
                "user_id",
                "item_id_list",
                "item_type_list",
                "item_length",
                "item_id",
                "timestamp",
                "item_type",
            ]
        },

        "eval_args": {
            "order": "TO",
            "mode": "full",
        },
        "train_neg_sample_args": None,
        "metrics": ["Recall", "NDCG", "MRR"],
        "topk": [10, 20],
        "valid_metric": args.valid_metric or "Recall@10",

        # Model behavior fields.
        "ITEM_TYPE_SEQ_FIELD": "item_type_list",
        "ITEM_TYPE_FIELD": "item_type",
        "target_behavior_token": target_token,
        "mask_behavior_as_target": True,
        "num_behaviors": 4,
    }
    if args.train_batch_size is not None:
        cfg["train_batch_size"] = args.train_batch_size
    if args.eval_batch_size is not None:
        cfg["eval_batch_size"] = args.eval_batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.show_progress is not None:
        cfg["show_progress"] = bool(args.show_progress)
    return cfg


def get_hyperparameter_grid(config) -> List[Dict[str, Any]]:
    """
    Read:
        hyper_parameters: ['learning_rate', 'dropout_prob']
        learning_rate: [0.001, 0.0005]
        dropout_prob: [0.3, 0.5]
    Return cartesian product:
        0.001 x 0.3, 0.001 x 0.5, 0.0005 x 0.3, 0.0005 x 0.5
    """
    hyper_parameters = _config_get(config, "hyper_parameters", [])
    if hyper_parameters is None or hyper_parameters == "":
        return [{}]
    if not hyper_parameters:
        return [{}]
    if not isinstance(hyper_parameters, (list, tuple)):
        raise ValueError("hyper_parameters must be a list, e.g. ['learning_rate'].")

    hyper_values = []
    for hp in hyper_parameters:
        values = _config_get(config, hp, None)
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"Hyperparameter '{hp}' must be configured as a list. "
                f"Example: {hp}: [0.001, 0.0005]. Current value={values!r}"
            )
        hyper_values.append(list(values))

    grid = []
    for combo_values in itertools.product(*hyper_values):
        grid.append({hp: value for hp, value in zip(hyper_parameters, combo_values)})
    return grid


def sanitize_run_overrides(config, hp_override: Dict[str, Any]) -> Dict[str, Any]:
    """
    For one run, override hyperparameter list values with scalar values.
    Also sets hyper_parameters=[] so list-valued hparam keys are not accidentally
    interpreted by model code.
    """
    out = dict(hp_override)
    out["hyper_parameters"] = []
    return out


def append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_plain(obj), ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, records: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    flat_records = []
    for r in records:
        flat = {}
        for k, v in r.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        for kkk, vvv in vv.items():
                            flat[f"{k}.{kk}.{kkk}"] = _plain(vvv)
                    else:
                        flat[f"{k}.{kk}"] = _plain(vv)
            else:
                flat[k] = _plain(v)
        flat_records.append(flat)
    keys = sorted({k for r in flat_records for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in flat_records:
            writer.writerow(r)


def try_flops(model, dataset, config, logger):
    if get_flops is None:
        return {"error": "recbole.utils.get_flops unavailable"}
    try:
        transform = construct_transform(config)
        flops = get_flops(model, dataset, config["device"], logger, transform)
        return {"flops": _plain(flops)}
    except Exception as e:
        return {"error": str(e)}


def try_performance_stats(model, eval_data, config, logger, level):
    if level == "none":
        return {}
    try:
        if level == "quick":
            from performance_stats import get_quick_stats
            return get_quick_stats(model, eval_data, config, logger)
        else:
            from performance_stats import get_comprehensive_stats
            return get_comprehensive_stats(model, eval_data, config, logger)
    except Exception as e:
        logger.warning(set_color(f"Performance stats failed: {e}", "yellow"))
        return {"error": str(e)}


def get_valid_metric_bigger(config) -> bool:
    val = _config_get(config, "valid_metric_bigger", True)
    if isinstance(val, str):
        return val.lower() not in ("false", "0", "no")
    return bool(val)


def run_one(
    args,
    behavior: str,
    hp_override: Dict[str, Any],
    stage: str,
    run_idx: int,
    total_runs: int,
    result_dir: Path,
    config_file_list: Optional[List[str]],
) -> Dict[str, Any]:
    logger = getLogger()
    ds = dataset_name(args.dataset_prefix, behavior)
    base_dict = base_config_dict(args, behavior)
    run_config_dict = copy.deepcopy(base_dict)
    run_config_dict.update(sanitize_run_overrides(None, hp_override))

    config = Config(
        model=args.model,
        dataset=ds,
        config_file_list=config_file_list,
        config_dict=run_config_dict,
    )
    config.final_config_dict["MODEL_TYPE"] = ModelType.SEQUENTIAL
    config.final_config_dict["target_behavior_token"] = run_config_dict["target_behavior_token"]

    logger.info("\n" + "=" * 100)
    logger.info(set_color(f"{stage} run {run_idx}/{total_runs}", "cyan"))
    logger.info(f"dataset={ds}, behavior={behavior}, hp={hp_override}")
    logger.info(config)

    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    logger.info(model)

    flops_info = {}
    if args.compute_flops:
        flops_info = try_flops(model, dataset, config, logger)
        logger.info(set_color("FLOPs", "blue") + f": {flops_info}")

    trainer = Trainer(config, model)

    fit_start = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    fit_sec = time.time() - fit_start

    test_start = time.time()
    test_result = trainer.evaluate(test_data, show_progress=config["show_progress"])
    test_sec = time.time() - test_start

    logger.info(set_color("best valid score", "yellow") + f": {best_valid_score}")
    logger.info(set_color("best valid result", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    logger.info(set_color("train wall time", "blue") + f": {fit_sec:.2f}s")
    logger.info(set_color("test wall time", "blue") + f": {test_sec:.2f}s")

    perf = try_performance_stats(model, test_data, config, logger, args.stats_level)

    env = get_environment(config)
    logger.info("Environment:\n" + env.draw())

    record = {
        "stage": stage,
        "run_idx": run_idx,
        "total_runs": total_runs,
        "model": args.model,
        "dataset": ds,
        "behavior": behavior,
        "target_behavior_token": run_config_dict["target_behavior_token"],
        "hyperparameters": hp_override,
        "best_valid_score": _plain(best_valid_score),
        "valid_metric_bigger": get_valid_metric_bigger(config),
        "best_valid_result": _plain(best_valid_result),
        "test_result": _plain(test_result),
        "fit_sec": fit_sec,
        "test_sec": test_sec,
        "flops": flops_info,
        "performance": _plain(perf),
        "config_snapshot": {
            "hidden_size": _config_get(config, "hidden_size"),
            "num_layers": _config_get(config, "num_layers"),
            "num_heads": _config_get(config, "num_heads", _config_get(config, "n_heads", None)),
            "dropout_prob": _config_get(config, "dropout_prob"),
            "learning_rate": _config_get(config, "learning_rate"),
            "train_batch_size": _config_get(config, "train_batch_size"),
            "eval_batch_size": _config_get(config, "eval_batch_size"),
            "loss_type": _config_get(config, "loss_type"),
            "coupling_mode": _config_get(config, "coupling_mode", None),
            "hnn_residual_init": _config_get(config, "hnn_residual_init", None),
            "kernel_size": _config_get(config, "kernel_size", None),
        },
    }

    append_jsonl(result_dir / "run_records.jsonl", record)
    return record


def pick_best(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise RuntimeError("No tuning records to select from.")
    bigger = records[0].get("valid_metric_bigger", True)

    def key_fn(r):
        try:
            return float(r["best_valid_score"])
        except Exception:
            return float("-inf") if bigger else float("inf")

    return max(records, key=key_fn) if bigger else min(records, key=key_fn)


def save_best_files(result_dir: Path, best: Dict[str, Any]):
    best_json = {
        "best_hyperparameters": best["hyperparameters"],
        "best_valid_score": best["best_valid_score"],
        "best_valid_result": best["best_valid_result"],
        "test_result_at_best_valid": best["test_result"],
        "dataset": best["dataset"],
        "behavior": best["behavior"],
        "model": best["model"],
    }
    (result_dir / "best_hyperparameters.json").write_text(
        json.dumps(_plain(best_json), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Also write a tiny YAML-style file for easy copy-paste.
    lines = ["# Best hyperparameters selected on purchase validation\n"]
    for k, v in best["hyperparameters"].items():
        if isinstance(v, str):
            lines.append(f"{k}: {v}\n")
        else:
            lines.append(f"{k}: {repr(v)}\n")
    (result_dir / "best_hyperparameters.yaml").write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    config_file_list = args.config_files.strip().split(" ") if args.config_files else None

    tune_ds = dataset_name(args.dataset_prefix, args.tune_behavior)
    initial_config = Config(
        model=args.model,
        dataset=tune_ds,
        config_file_list=config_file_list,
        config_dict=base_config_dict(args, args.tune_behavior),
    )
    initial_config.final_config_dict["MODEL_TYPE"] = ModelType.SEQUENTIAL

    init_logger(initial_config)
    logger = getLogger()
    logger.info("AMPL auto-grid runner")
    logger.info(f"args={args}")

    grid = get_hyperparameter_grid(initial_config)
    result_dir = (
        Path(args.results_dir)
        / args.model
        / args.dataset_prefix
        / f"{args.tune_behavior}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(set_color(f"Total hyperparameter combinations: {len(grid)}", "cyan"))
    for i, hp in enumerate(grid, start=1):
        logger.info(f"[dry-list] {i}/{len(grid)} {hp}")

    if args.dry_run:
        logger.info("dry_run=True, stop before training.")
        return

    # 1) Tune on purchase or specified tune behavior.
    tune_records = []
    for run_idx, hp in enumerate(grid, start=1):
        rec = run_one(
            args=args,
            behavior=args.tune_behavior,
            hp_override=hp,
            stage="tune",
            run_idx=run_idx,
            total_runs=len(grid),
            result_dir=result_dir,
            config_file_list=config_file_list,
        )
        tune_records.append(rec)
        write_summary_csv(result_dir / "tune_summary.csv", tune_records)
        write_summary_csv(result_dir / "summary.csv", tune_records)

    best = pick_best(tune_records)
    save_best_files(result_dir, best)
    logger.info("\n" + "=" * 100)
    logger.info(set_color("BEST HYPERPARAMETERS ON TUNE BEHAVIOR", "green"))
    logger.info(f"best behavior={best['behavior']}")
    logger.info(f"best valid score={best['best_valid_score']}")
    logger.info(f"best hp={best['hyperparameters']}")
    logger.info(f"best files saved to {result_dir}")

    # 2) Optionally run all behaviors with best hp.
    all_records = list(tune_records)
    if args.run_all_after_tune:
        behaviors = infer_behaviors(args.dataset_prefix, args.behaviors)
        eval_records = []
        eval_total = len(behaviors)
        for idx, behavior in enumerate(behaviors, start=1):
            if args.reuse_tuned_behavior and behavior == args.tune_behavior:
                reuse = copy.deepcopy(best)
                reuse["stage"] = "best_all_reused_tune"
                eval_records.append(reuse)
                continue
            rec = run_one(
                args=args,
                behavior=behavior,
                hp_override=best["hyperparameters"],
                stage="best_all",
                run_idx=idx,
                total_runs=eval_total,
                result_dir=result_dir,
                config_file_list=config_file_list,
            )
            eval_records.append(rec)
            all_records.append(rec)
            write_summary_csv(result_dir / "best_all_summary.csv", eval_records)
            write_summary_csv(result_dir / "summary.csv", all_records)

    logger.info("\n" + "=" * 100)
    logger.info(set_color("Finished auto-grid experiment", "green"))
    logger.info(f"Result directory: {result_dir}")
    logger.info(f"Tune summary: {result_dir / 'tune_summary.csv'}")
    logger.info(f"All records: {result_dir / 'run_records.jsonl'}")
    logger.info(f"Best hp: {result_dir / 'best_hyperparameters.json'}")


if __name__ == "__main__":
    main()
