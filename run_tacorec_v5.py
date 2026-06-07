#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TacoRec / CoupledHSR-v5 experiment runner for AMPL-style datasets.

This script supports three modes:

1) tune:
   grid-search on one behavior, usually purchase, and save all runs.

2) final:
   run one selected config on AMPL-style behavior tasks:
   JD: click/comment/favourite/purchase
   UB: click/cart/favourite/purchase
   It writes a paper-ready "ours" table that can fill the blank column.

3) ablation:
   run v5-specific ablations:
   A0 anchor only
   A1 anchor + target-conditioned behavior memory
   A2 anchor + spectral residual
   A3 no cross-behavior coupling
   A4 symmetric coupling
   A5 static causal coupling
   A6 full v5

Example: tune JD purchase
python run_tacorec_v5.py \
  --mode tune \
  --model CoupledHSR \
  --dataset_prefix ampl_jd \
  --behaviors purchase \
  --config_files configs/model/CoupledHSR_v5_GRID.yaml \
  -g 0

Example: run final AMPL-style table on JD and UB
python run_tacorec_v5.py \
  --mode final \
  --model CoupledHSR \
  --dataset_prefix ampl_jd,ampl_ub \
  --behaviors auto \
  --config_files configs/model/CoupledHSR_v5.yaml \
  -g 0

Example: v5 ablation on JD purchase
python run_tacorec_v5.py \
  --mode ablation \
  --model CoupledHSR \
  --dataset_prefix ampl_jd \
  --behaviors purchase \
  --config_files configs/model/CoupledHSR_v5.yaml \
  -g 0
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, get_model, get_environment, set_color

try:
    from recbole.utils.enum_type import ModelType
except Exception:
    from recbole.utils import ModelType


JD_BEHAVIORS = ["click", "comment", "favourite", "purchase"]
UB_BEHAVIORS = ["click", "cart", "favourite", "purchase"]

PRETTY_BEHAVIOR = {
    "click": "Examination",
    "comment": "Comment",
    "cart": "Cart",
    "favourite": "Favorite",
    "favorite": "Favorite",
    "purchase": "Purchase",
    "union": "Union",
}

PAPER_ROWS = [
    ("JD-Examination", "ampl_jd", "click"),
    ("JD-Comment", "ampl_jd", "comment"),
    ("JD-Favorite", "ampl_jd", "favourite"),
    ("JD-Purchase", "ampl_jd", "purchase"),
    ("UB-Examination", "ampl_ub", "click"),
    ("UB-Cart", "ampl_ub", "cart"),
    ("UB-Favorite", "ampl_ub", "favourite"),
    ("UB-Purchase", "ampl_ub", "purchase"),
]

# v5 ablations. These keys must match your model's config names.
ABLATION_VARIANTS: OrderedDict[str, Dict[str, Any]] = OrderedDict({
    "A0_anchor_only": {
        "use_transformer": True,
        "use_hnn": False,
        "use_behavior_memory": False,
        "use_target_residual_gate": False,
        "ablation_note": "Causal sequence anchor only.",
    },
    "A1_anchor_memory": {
        "use_transformer": True,
        "use_hnn": False,
        "use_behavior_memory": True,
        "use_target_residual_gate": False,
        "ablation_note": "Anchor plus target-conditioned behavior memory.",
    },
    "A2_anchor_hsr": {
        "use_transformer": True,
        "use_hnn": True,
        "use_behavior_memory": False,
        "use_target_residual_gate": True,
        "coupling_mode": "causal",
        "ablation_note": "Anchor plus spectral Hamiltonian residual without behavior memory.",
    },
    "A3_no_coupling": {
        "use_transformer": True,
        "use_hnn": True,
        "use_behavior_memory": True,
        "use_target_residual_gate": True,
        "coupling_mode": "none",
        "ablation_note": "Full v5 except cross-behavior coupling is removed.",
    },
    "A4_symmetric_coupling": {
        "use_transformer": True,
        "use_hnn": True,
        "use_behavior_memory": True,
        "use_target_residual_gate": True,
        "coupling_mode": "symmetric",
        "ablation_note": "Replace causal behavior transfer with symmetric coupling.",
    },
    "A5_static_causal": {
        "use_transformer": True,
        "use_hnn": True,
        "use_behavior_memory": True,
        "use_target_residual_gate": True,
        "coupling_mode": "causal",
        "n_coupling_bands": 1,
        "n_coup_bands": 1,
        "ablation_note": "Causal coupling with one band, removing frequency adaptation.",
    },
    "A6_full_v5": {
        "use_transformer": True,
        "use_hnn": True,
        "use_behavior_memory": True,
        "use_target_residual_gate": True,
        "coupling_mode": "causal",
        "ablation_note": "Full v5 with target-conditioned memory and frequency-adaptive causal coupling.",
    },
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["tune", "final", "ablation"], required=True)
    p.add_argument("--model", "-m", default="CoupledHSR")
    p.add_argument("--dataset_prefix", required=True,
                   help="ampl_jd, ampl_ub, or comma list such as ampl_jd,ampl_ub")
    p.add_argument("--behaviors", default="purchase",
                   help="purchase, click,comment,favourite,purchase, union, or auto")
    p.add_argument("--config_files", required=True,
                   help="YAML config file. For tune mode, list-valued keys in hyper_parameters define the grid.")
    p.add_argument("--gpu_id", "-g", type=int, default=0)
    p.add_argument("--results_dir", default="saved/tacorec_v5")
    p.add_argument("--valid_metric", default="Recall@10")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--show_progress", type=int, default=None)
    p.add_argument("--max_runs", type=int, default=None,
                   help="Optional cap for debugging grid search.")
    p.add_argument("--resume_csv", default=None,
                   help="Optional previous summary.csv. Existing config signatures will be skipped.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def is_grid_value(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and not isinstance(v[0], (dict, list))


def make_scalar_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base)
    cfg.update(overrides)
    # RecBole should not receive hyper_parameters as actual tune keys here.
    cfg["hyper_parameters"] = []
    # Enforce v5/RecBole sequential benchmark fields.
    cfg.setdefault("MODEL_TYPE", "SEQUENTIAL")
    cfg.setdefault("benchmark_filename", ["train", "valid", "test"])
    cfg.setdefault("USER_ID_FIELD", "user_id")
    cfg.setdefault("ITEM_ID_FIELD", "item_id")
    cfg.setdefault("TIME_FIELD", "timestamp")
    cfg.setdefault("ITEM_LIST_LENGTH_FIELD", "item_length")
    cfg.setdefault("LIST_SUFFIX", "_list")
    cfg.setdefault("MAX_ITEM_LIST_LENGTH", 50)
    cfg.setdefault("alias_of_item_id", ["item_id_list"])
    cfg.setdefault("alias_of_item_type", ["item_type_list"])
    cfg.setdefault("field_separator", "\t")
    cfg.setdefault("seq_separator", " ")
    cfg.setdefault("ITEM_TYPE_SEQ_FIELD", "item_type_list")
    cfg.setdefault("ITEM_TYPE_FIELD", "item_type")
    cfg.setdefault("num_behaviors", 4)
    cfg.setdefault("mask_behavior_as_target", True)
    cfg.setdefault("use_interaction_target_behavior", True)
    cfg.setdefault("load_col", {
        "inter": [
            "user_id", "item_id_list", "item_type_list", "item_length",
            "item_id", "timestamp", "item_type"
        ]
    })
    cfg.setdefault("eval_args", {"order": "TO", "mode": "full"})
    cfg.setdefault("train_neg_sample_args", None)
    cfg.setdefault("metrics", ["Recall", "NDCG", "MRR"])
    cfg.setdefault("topk", [10, 20])
    cfg.setdefault("valid_metric", "Recall@10")
    return cfg


def grid_from_config(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    hp = base.get("hyper_parameters", [])
    if hp is None:
        hp = []
    if isinstance(hp, str):
        hp = [hp]
    hp = [str(x) for x in hp]

    # Only keys listed in hyper_parameters are treated as grid keys.
    grid_keys = [k for k in hp if is_grid_value(base.get(k))]
    if not grid_keys:
        return [{}]

    values = [base[k] for k in grid_keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(grid_keys, prod)})
    return combos


def parse_prefixes(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def behaviors_for(prefix: str, behavior_arg: str) -> List[str]:
    if behavior_arg == "auto":
        if prefix.endswith("jd"):
            return JD_BEHAVIORS
        if prefix.endswith("ub"):
            return UB_BEHAVIORS
        return ["purchase"]
    return [x.strip() for x in behavior_arg.split(",") if x.strip()]


def dataset_name(prefix: str, behavior: str) -> str:
    return f"{prefix}_{behavior}"


def safe_metric(d: Dict[str, Any], name: str):
    if not isinstance(d, dict):
        return None
    name_lower = name.lower()
    for k, v in d.items():
        if str(k).lower() == name_lower:
            return float(v)
    return None


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def append_jsonl(path: Path, record: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: List[Dict[str, Any]]):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_records = [flatten_dict(r) for r in records]
    keys = sorted({k for row in flat_records for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_records)


def config_signature(prefix: str, behavior: str, variant: str, combo: Dict[str, Any]) -> str:
    payload = {"prefix": prefix, "behavior": behavior, "variant": variant, "combo": combo}
    return json.dumps(payload, sort_keys=True)


def load_done_signatures(path: str | None) -> set:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    done = set()
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sig = row.get("signature")
            if sig:
                done.add(sig)
    return done


def temp_yaml(cfg: Dict[str, Any], run_dir: Path, name: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def run_one(
    args,
    base_config: Dict[str, Any],
    prefix: str,
    behavior: str,
    combo: Dict[str, Any],
    variant_name: str,
    variant_cfg: Dict[str, Any],
    run_dir: Path,
    run_idx: int,
    total_runs: int,
):
    ds = dataset_name(prefix, behavior)

    cfg_dict = make_scalar_config(base_config, {})
    cfg_dict.update(combo)
    cfg_dict.update({k: v for k, v in variant_cfg.items() if k != "ablation_note"})
    cfg_dict["gpu_id"] = args.gpu_id
    cfg_dict["valid_metric"] = args.valid_metric
    if args.seed is not None:
        cfg_dict["seed"] = args.seed
    if args.show_progress is not None:
        cfg_dict["show_progress"] = bool(args.show_progress)

    yaml_path = temp_yaml(cfg_dict, run_dir / "tmp_configs", f"{run_idx:04d}_{ds}_{variant_name}")

    config = Config(
        model=args.model,
        dataset=ds,
        config_file_list=[str(yaml_path)],
        config_dict={"gpu_id": args.gpu_id},
    )
    # Avoid RecBole KeyError: 'sequential'
    config.final_config_dict["MODEL_TYPE"] = ModelType.SEQUENTIAL
    config.final_config_dict["LIST_SUFFIX"] = "_list"
    config.final_config_dict["ITEM_ID_LIST_FIELD"] = "item_id_list"
    config.final_config_dict["ITEM_TYPE_LIST_FIELD"] = "item_type_list"
    config.final_config_dict["alias_of_item_id"] = ["item_id_list"]
    config.final_config_dict["alias_of_item_type"] = ["item_type_list"]

    if run_idx == 1:
        init_logger(config)
    import logging
    logger = logging.getLogger()

    logger.info("=" * 120)
    logger.info(set_color(f"Run {run_idx}/{total_runs}", "cyan"))
    logger.info(f"mode={args.mode} dataset={ds} behavior={behavior} variant={variant_name}")
    logger.info(f"combo={combo}")
    if variant_cfg.get("ablation_note"):
        logger.info(f"note={variant_cfg.get('ablation_note')}")
    logger.info(f"scalar yaml: {yaml_path}")

    init_seed(config["seed"], config["reproducibility"])

    start_total = time.time()
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    fit_start = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    fit_sec = time.time() - fit_start

    test_start = time.time()
    test_result = trainer.evaluate(test_data, show_progress=config["show_progress"])
    test_sec = time.time() - test_start

    total_sec = time.time() - start_total

    # Collect params and GPU memory if available.
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        import torch
        gpu_mem_mb = torch.cuda.max_memory_allocated(config["device"]) / 1024 / 1024 if str(config["device"]).startswith("cuda") else 0.0
    except Exception:
        gpu_mem_mb = None

    record = {
        "signature": config_signature(prefix, behavior, variant_name, combo),
        "mode": args.mode,
        "model": args.model,
        "dataset_prefix": prefix,
        "dataset": ds,
        "behavior": behavior,
        "behavior_pretty": PRETTY_BEHAVIOR.get(behavior, behavior),
        "variant": variant_name,
        "variant_note": variant_cfg.get("ablation_note", ""),
        "combo": combo,
        "best_valid_score": float(best_valid_score),
        "valid_recall10": safe_metric(best_valid_result, "recall@10"),
        "valid_ndcg10": safe_metric(best_valid_result, "ndcg@10"),
        "valid_recall20": safe_metric(best_valid_result, "recall@20"),
        "valid_ndcg20": safe_metric(best_valid_result, "ndcg@20"),
        "test_recall10": safe_metric(test_result, "recall@10"),
        "test_ndcg10": safe_metric(test_result, "ndcg@10"),
        "test_recall20": safe_metric(test_result, "recall@20"),
        "test_ndcg20": safe_metric(test_result, "ndcg@20"),
        "best_valid_result": dict(best_valid_result),
        "test_result": dict(test_result),
        "fit_sec": fit_sec,
        "test_sec": test_sec,
        "total_sec": total_sec,
        "num_params": int(num_params),
        "trainable_params": int(trainable_params),
        "gpu_mem_mb": gpu_mem_mb,
        "config_file": str(yaml_path),
        "key_config": {
            "hidden_size": cfg_dict.get("hidden_size"),
            "num_layers": cfg_dict.get("num_layers"),
            "n_hnn_layers": cfg_dict.get("n_hnn_layers"),
            "num_heads": cfg_dict.get("num_heads"),
            "dropout_prob": cfg_dict.get("dropout_prob"),
            "learning_rate": cfg_dict.get("learning_rate"),
            "loss_type": cfg_dict.get("loss_type"),
            "num_neg": cfg_dict.get("num_neg"),
            "use_hnn": cfg_dict.get("use_hnn"),
            "use_behavior_memory": cfg_dict.get("use_behavior_memory"),
            "use_target_residual_gate": cfg_dict.get("use_target_residual_gate"),
            "coupling_mode": cfg_dict.get("coupling_mode"),
            "n_coupling_bands": cfg_dict.get("n_coupling_bands"),
            "coupling_scale": cfg_dict.get("coupling_scale"),
            "hnn_residual_init": cfg_dict.get("hnn_residual_init"),
            "residual_warmup_steps": cfg_dict.get("residual_warmup_steps"),
            "max_residual_scale": cfg_dict.get("max_residual_scale"),
            "memory_scale_init": cfg_dict.get("memory_scale_init"),
        },
    }

    logger.info(set_color("best_valid_result", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test_result", "yellow") + f": {test_result}")
    logger.info(f"fit_sec={fit_sec:.2f}, test_sec={test_sec:.2f}, params={num_params:,}, gpu_mem_mb={gpu_mem_mb}")

    return record


def latex_ours_table(records: List[Dict[str, Any]], out_path: Path):
    # Choose best by valid recall if there are multiple records for the same row.
    best = {}
    for rec in records:
        key = (rec["dataset_prefix"], rec["behavior"])
        cur = best.get(key)
        if cur is None or (rec.get("valid_recall10") or -1) > (cur.get("valid_recall10") or -1):
            best[key] = rec

    lines = []
    lines.append("% Ours columns generated by run_tacorec_v5.py")
    lines.append("% Fill these two numbers into the R@10 / N@10 columns of Table II.")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Behavior & R@10 & N@10 \\\\")
    lines.append("\\midrule")
    for display, prefix, beh in PAPER_ROWS:
        rec = best.get((prefix, beh))
        if rec is None:
            r = "--"
            n = "--"
        else:
            r = f"{rec.get('test_recall10', 0):.4f}"
            n = f"{rec.get('test_ndcg10', 0):.4f}"
        if display == "UB-Examination":
            lines.append("\\midrule")
        lines.append(f"{display} & {r} & {n} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    base_config = load_yaml(args.config_files)
    prefixes = parse_prefixes(args.dataset_prefix)

    if args.mode == "tune":
        combos = grid_from_config(base_config)
        variants = OrderedDict({"tune": {}})
    elif args.mode == "final":
        combos = [{}]
        variants = OrderedDict({"final": {}})
    else:
        combos = [{}]
        variants = ABLATION_VARIANTS

    plan = []
    for prefix in prefixes:
        for behavior in behaviors_for(prefix, args.behaviors):
            for variant_name, variant_cfg in variants.items():
                for combo in combos:
                    plan.append((prefix, behavior, combo, variant_name, variant_cfg))

    if args.max_runs is not None:
        plan = plan[: args.max_runs]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.results_dir) / args.mode / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (run_dir / "base_config_snapshot.yaml").write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    if args.mode == "ablation":
        (run_dir / "ablation_variants.json").write_text(json.dumps(ABLATION_VARIANTS, indent=2), encoding="utf-8")

    done = load_done_signatures(args.resume_csv)

    print(f"[plan] mode={args.mode} total_runs={len(plan)} output={run_dir}")
    for i, (prefix, behavior, combo, variant_name, variant_cfg) in enumerate(plan, 1):
        print(f"[plan] {i:03d}/{len(plan):03d} {dataset_name(prefix, behavior)} variant={variant_name} combo={combo}")

    if args.dry_run:
        return

    records = []
    jsonl_path = run_dir / "records.jsonl"
    csv_path = run_dir / "summary.csv"

    for i, (prefix, behavior, combo, variant_name, variant_cfg) in enumerate(plan, 1):
        sig = config_signature(prefix, behavior, variant_name, combo)
        if sig in done:
            print(f"[skip] already done: {sig}")
            continue
        try:
            rec = run_one(args, base_config, prefix, behavior, combo, variant_name, variant_cfg, run_dir, i, len(plan))
            records.append(rec)
            append_jsonl(jsonl_path, rec)
            write_csv(csv_path, records)
        except Exception as e:
            err = {
                "signature": sig,
                "mode": args.mode,
                "dataset_prefix": prefix,
                "dataset": dataset_name(prefix, behavior),
                "behavior": behavior,
                "variant": variant_name,
                "combo": combo,
                "error": repr(e),
            }
            records.append(err)
            append_jsonl(jsonl_path, err)
            write_csv(csv_path, records)
            print(f"[error] {dataset_name(prefix, behavior)} variant={variant_name}: {e}")
            raise

    # Best records by behavior.
    ok_records = [r for r in records if "error" not in r]
    if ok_records:
        best = {}
        for rec in ok_records:
            key = (rec["dataset_prefix"], rec["behavior"])
            cur = best.get(key)
            if cur is None or (rec.get("valid_recall10") or -1) > (cur.get("valid_recall10") or -1):
                best[key] = rec
        best_records = list(best.values())
        write_csv(run_dir / "best_by_behavior.csv", best_records)
        latex_ours_table(ok_records, run_dir / "ours_table_fillin.tex")

    print(f"[done] records: {jsonl_path}")
    print(f"[done] summary: {csv_path}")
    print(f"[done] best: {run_dir / 'best_by_behavior.csv'}")
    print(f"[done] latex ours table: {run_dir / 'ours_table_fillin.tex'}")


if __name__ == "__main__":
    main()
