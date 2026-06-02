# -*- coding: utf-8 -*-
"""
Full-ranking RecBole runner for AMPL-converted sequence datasets.

Features:
  1) Works with old RecBole Config, no config.get() required.
  2) Supports hyperparameter sweeps from YAML:
        hyper_parameters: ['hidden_size', 'dropout_prob']
        hidden_size: [30, 50]
        dropout_prob: [0.3, 0.5]
  3) Records every run to JSONL + CSV.
  4) Logs quick/full performance stats: params, memory, latency, optional FLOPs.
  5) Supports AMPL baseline and CoupledHSR A6 in the same pipeline.

Example:
python run_ampl_seq_sweep.py \
  --model AMPL \
  --dataset ampl_jd_purchase \
  --config_files configs/model/AMPL.yaml \
  -g 0 \
  --stats_level quick

python run_ampl_seq_sweep.py \
  --model CoupledHSR \
  --dataset ampl_jd_purchase \
  --config_files configs/model/CoupledHSR_AMPL_SEQ.yaml \
  -g 0 \
  --stats_level full
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
from typing import Any, Dict, List

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


DEFAULT_TARGET = {
    "click": 1,
    "comment": 2,
    "cart": 2,
    "favourite": 3,
    "favorite": 3,
    "purchase": 4,
}


def _config_get(config, key, default=None):
    try:
        return config[key]
    except Exception:
        return default


def _plain(obj):
    """Make RecBole/Tensor/numpy objects JSON serializable."""
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


def infer_target_behavior(dataset_name: str, explicit):
    if explicit is not None:
        return int(explicit)
    suffix = dataset_name.split("_")[-1].lower()
    return DEFAULT_TARGET.get(suffix, 4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="CoupledHSR")
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--config_files", type=str, default=None)
    parser.add_argument("--gpu_id", "-g", type=int, default=0)
    parser.add_argument("--target_behavior_token", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--valid_metric", type=str, default=None)
    parser.add_argument("--stats_level", choices=["none", "quick", "full"], default="quick")
    parser.add_argument("--compute_flops", action="store_true")
    parser.add_argument("--results_dir", type=str, default="saved/ampl_sweeps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show_progress", type=int, default=None)
    return parser.parse_args()


def base_config_dict(args):
    target_behavior = infer_target_behavior(args.dataset, args.target_behavior_token)
    cfg = {
        "gpu_id": args.gpu_id,
        "MODEL_TYPE": ModelType.SEQUENTIAL,

        # Benchmark sequence files created by convert_ampl_to_recbole_seq.py
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

        # Model-specific behavior fields.
        "ITEM_TYPE_SEQ_FIELD": "item_type_list",
        "ITEM_TYPE_FIELD": "item_type",
        "target_behavior_token": target_behavior,
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
    hyper_parameters = _config_get(config, "hyper_parameters", [])
    if hyper_parameters is None or hyper_parameters == "":
        hyper_parameters = []
    if not hyper_parameters:
        return [{}]
    if not isinstance(hyper_parameters, (list, tuple)):
        raise ValueError("hyper_parameters must be a list, e.g. ['hidden_size', 'dropout_prob'].")

    hyper_values = []
    for hp in hyper_parameters:
        values = _config_get(config, hp, None)
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"Hyperparameter '{hp}' must be a list in YAML, but got {values!r}."
            )
        hyper_values.append(list(values))

    grid = []
    for combo_values in itertools.product(*hyper_values):
        grid.append({hp: value for hp, value in zip(hyper_parameters, combo_values)})
    return grid


def append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_plain(obj), ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, records: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    # Flatten one level for common fields.
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


def run_one(args, base_dict, config_file_list, hp_override, run_idx, total_runs, result_dir):
    logger = getLogger()

    run_config_dict = copy.deepcopy(base_dict)
    run_config_dict.update(hp_override)

    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=run_config_dict,
    )
    config.final_config_dict["MODEL_TYPE"] = ModelType.SEQUENTIAL
    config.final_config_dict["target_behavior_token"] = run_config_dict["target_behavior_token"]

    logger.info("\n" + "=" * 100)
    logger.info(set_color(f"Hyperparameter run {run_idx}/{total_runs}", "cyan"))
    logger.info(f"Dataset={args.dataset}, Model={args.model}, hp={hp_override}")
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

    start = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    fit_sec = time.time() - start

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
        "run_idx": run_idx,
        "total_runs": total_runs,
        "model": args.model,
        "dataset": args.dataset,
        "target_behavior_token": run_config_dict["target_behavior_token"],
        "hyperparameters": hp_override,
        "best_valid_score": _plain(best_valid_score),
        "best_valid_result": _plain(best_valid_result),
        "test_result": _plain(test_result),
        "fit_sec": fit_sec,
        "test_sec": test_sec,
        "flops": flops_info,
        "performance": _plain(perf),
        "config_snapshot": {
            "hidden_size": _config_get(config, "hidden_size"),
            "num_layers": _config_get(config, "num_layers"),
            "num_heads": _config_get(config, "num_heads"),
            "dropout_prob": _config_get(config, "dropout_prob"),
            "learning_rate": _config_get(config, "learning_rate"),
            "train_batch_size": _config_get(config, "train_batch_size"),
            "eval_batch_size": _config_get(config, "eval_batch_size"),
            "loss_type": _config_get(config, "loss_type"),
            "coupling_mode": _config_get(config, "coupling_mode", None),
            "hnn_residual_init": _config_get(config, "hnn_residual_init", None),
        },
    }

    append_jsonl(result_dir / "run_records.jsonl", record)
    return record


def main():
    args = parse_args()
    config_file_list = args.config_files.strip().split(" ") if args.config_files else None

    # Build an initial config only to read the hyperparameter grid.
    base_dict = base_config_dict(args)
    initial_config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=base_dict,
    )
    initial_config.final_config_dict["MODEL_TYPE"] = ModelType.SEQUENTIAL

    # Logger after initial config.
    init_logger(initial_config)
    logger = getLogger()
    logger.info("Command-line args are handled by run_ampl_seq_sweep.py, not by RecBole parser.")
    logger.info(f"args={args}")

    grid = get_hyperparameter_grid(initial_config)
    result_dir = Path(args.results_dir) / args.model / args.dataset / time.strftime("%Y%m%d-%H%M%S")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    records = []
    for run_idx, hp_override in enumerate(grid, start=1):
        rec = run_one(
            args=args,
            base_dict=base_dict,
            config_file_list=config_file_list,
            hp_override=hp_override,
            run_idx=run_idx,
            total_runs=len(grid),
            result_dir=result_dir,
        )
        records.append(rec)
        write_summary_csv(result_dir / "summary.csv", records)

    logger.info("\n" + "=" * 100)
    logger.info(set_color("All hyperparameter runs finished", "green"))
    logger.info(f"Records: {result_dir / 'run_records.jsonl'}")
    logger.info(f"Summary: {result_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
