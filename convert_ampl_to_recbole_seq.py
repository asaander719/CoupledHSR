#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMPL official-style converter to RecBole SequentialDataset benchmark files.

Key correction v2:
- AMPL Table 1 reports raw interaction counts in each split.
- The behavior-specific valid/test files should NOT be reduced to one row per user by default.
- The previous converter's "first target per user" option caused valid/test counts to be far smaller
  than AMPL Table 1. This version keeps all provided behavior-specific valid/test rows by default.

Input AMPL processed directory:
    union_train.csv
    union_valid.csv
    click_valid.csv / click_test.csv
    comment_valid.csv / comment_test.csv      for JD
    cart_valid.csv / cart_test.csv            for UB
    favourite_valid.csv / favourite_test.csv
    purchase_valid.csv / purchase_test.csv

Input CSV format:
    user_id,item_id,timestamp,behavior_id
without header.

Output RecBole sequence benchmark columns:
    user_id:token
    item_id_list:token_seq
    item_type_list:token_seq
    item_length:float
    item_id:token
    timestamp:float
    item_type:token

Recommended commands:
python convert_ampl_official_to_recbole_v2.py \
  --src_dir /home/asaliao/AMPL/processed_data_JD \
  --dataset_key JD \
  --out_root /home/asaliao/RecBole/dataset \
  --prefix ampl_jd \
  --max_len 50 \
  --train_scope union

python convert_ampl_official_to_recbole_v2.py \
  --src_dir /home/asaliao/AMPL/processed_data_UB \
  --dataset_key UB \
  --out_root /home/asaliao/RecBole/dataset \
  --prefix ampl_ub \
  --max_len 50 \
  --train_scope union
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_BEHAVIORS = {
    "JD": {
        "click": 0,       # examination
        "comment": 1,
        "favourite": 2,
        "purchase": 3,
    },
    "UB": {
        "click": 0,       # examination
        "cart": 1,
        "favourite": 2,
        "purchase": 3,
    },
}

OUT_COLS = [
    "user_id:token",
    "item_id_list:token_seq",
    "item_type_list:token_seq",
    "item_length:float",
    "item_id:token",
    "timestamp:float",
    "item_type:token",
]


def parse_behavior_map(s: Optional[str], dataset_key: str) -> Dict[str, int]:
    if not s:
        return dict(DEFAULT_BEHAVIORS[dataset_key.upper()])
    out = {}
    for part in s.split(","):
        name, value = part.split(":")
        out[name.strip()] = int(value)
    return out


def read_ampl_csv(path: Path, behavior_id: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 4:
        raise ValueError(f"{path} should have at least 4 columns: user,item,timestamp,behavior")
    df = df.iloc[:, :4].copy()
    df.columns = ["user_id", "item_id", "timestamp", "item_type"]
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype("int64")
    if behavior_id is None:
        df["item_type"] = pd.to_numeric(df["item_type"], errors="coerce").fillna(0).astype("int64")
    else:
        # Enforce target behavior for behavior-specific files.
        df["item_type"] = int(behavior_id)
    return df.sort_values(["user_id", "timestamp", "item_id", "item_type"]).reset_index(drop=True)


def dedup_earliest_user_item_behavior(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["user_id", "item_type", "item_id", "timestamp"])
        .drop_duplicates(["user_id", "item_id", "item_type"], keep="first")
        .sort_values(["user_id", "timestamp", "item_id", "item_type"])
        .reset_index(drop=True)
    )


def make_row(
    user_id: str,
    target_item: str,
    target_ts: int,
    target_type: int,
    history_events: List[Tuple[int, str, int]],
    max_len: int,
):
    if not history_events:
        return None
    hist = history_events[-max_len:]
    return {
        "user_id:token": str(user_id),
        "item_id_list:token_seq": " ".join(str(it) for _, it, _ in hist),
        "item_type_list:token_seq": " ".join(str(bt) for _, _, bt in hist),
        "item_length:float": float(len(hist)),
        "item_id:token": str(target_item),
        "timestamp:float": float(target_ts),
        "item_type:token": str(int(target_type)),
    }


def build_train_instances(
    union_train: pd.DataFrame,
    max_len: int,
    target_behavior: Optional[int] = None,
) -> pd.DataFrame:
    """
    RecBole sequence benchmark training rows.

    Note:
    If target_behavior is None, all union_train events except the first event of each user
    become next-event training targets. Therefore the row count is usually:
        len(union_train) - number_of_train_users
    This is expected and differs from AMPL Table 1, which reports raw interaction counts.
    """
    rows = []
    for u, g in union_train.sort_values(["user_id", "timestamp", "item_id", "item_type"]).groupby("user_id", sort=False):
        hist = []
        for _, r in g.iterrows():
            bt = int(r["item_type"])
            if target_behavior is None or bt == int(target_behavior):
                row = make_row(
                    user_id=str(u),
                    target_item=str(r["item_id"]),
                    target_ts=int(r["timestamp"]),
                    target_type=bt,
                    history_events=hist,
                    max_len=max_len,
                )
                if row is not None:
                    rows.append(row)
            hist.append((int(r["timestamp"]), str(r["item_id"]), bt))
    return pd.DataFrame(rows, columns=OUT_COLS)


def build_history_index(history_pool: pd.DataFrame) -> Dict[str, List[Tuple[int, str, int]]]:
    index = {}
    for u, g in history_pool.sort_values(["user_id", "timestamp", "item_id", "item_type"]).groupby("user_id", sort=False):
        index[str(u)] = list(
            zip(
                g["timestamp"].astype(int).tolist(),
                g["item_id"].astype(str).tolist(),
                g["item_type"].astype(int).tolist(),
            )
        )
    return index


def build_eval_instances(
    history_pool: pd.DataFrame,
    targets: pd.DataFrame,
    max_len: int,
    train_items: Optional[set],
    remove_cold_items: bool,
) -> pd.DataFrame:
    if remove_cold_items and train_items is not None:
        targets = targets[targets["item_id"].astype(str).isin(train_items)].copy()

    hist_index = build_history_index(history_pool)
    rows = []
    for _, r in targets.sort_values(["user_id", "timestamp", "item_id", "item_type"]).iterrows():
        u = str(r["user_id"])
        ts = int(r["timestamp"])
        hist = [e for e in hist_index.get(u, []) if e[0] < ts]
        row = make_row(
            user_id=u,
            target_item=str(r["item_id"]),
            target_ts=ts,
            target_type=int(r["item_type"]),
            history_events=hist,
            max_len=max_len,
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows, columns=OUT_COLS)


def write_inter(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        pd.DataFrame(columns=OUT_COLS).to_csv(path, sep="\t", index=False)
    else:
        df[OUT_COLS].to_csv(path, sep="\t", index=False)


def stats_raw(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "users": 0, "items": 0}
    return {
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
    }


def stats_seq(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "users": 0, "items": 0, "avg_hist_len": 0.0}
    return {
        "rows": int(len(df)),
        "users": int(df["user_id:token"].nunique()),
        "items": int(df["item_id:token"].nunique()),
        "avg_hist_len": float(df["item_length:float"].mean()),
    }


def convert(
    src_dir: Path,
    out_root: Path,
    prefix: str,
    behavior_map: Dict[str, int],
    max_len: int,
    train_scope: str,
    dedup_train_valid: bool,
    dedup_behavior_eval: bool,
    remove_cold_items: bool,
):
    union_train = read_ampl_csv(src_dir / "union_train.csv")
    union_valid = read_ampl_csv(src_dir / "union_valid.csv")

    if dedup_train_valid:
        union_train = dedup_earliest_user_item_behavior(union_train)
        union_valid = dedup_earliest_user_item_behavior(union_valid)

    train_items = set(union_train["item_id"].astype(str).unique())

    # AMPL-faithful training: all behaviors in union_train.
    train_union_df = build_train_instances(union_train, max_len=max_len, target_behavior=None)

    report = {
        "prefix": prefix,
        "src_dir": str(src_dir),
        "max_len": max_len,
        "train_scope": train_scope,
        "dedup_train_valid": dedup_train_valid,
        "dedup_behavior_eval": dedup_behavior_eval,
        "remove_cold_items": remove_cold_items,
        "raw_union_train": stats_raw(union_train),
        "raw_union_valid": stats_raw(union_valid),
        "datasets": {},
        "notes": [
            "AMPL Table 1 reports raw interaction counts.",
            "RecBole train rows are sequence target instances, so the first event of each user has no history and is not used as a training row.",
            "Behavior-specific valid/test rows are kept as provided by AMPL by default.",
            "union.test is concatenated from behavior-specific test files because AMPL provides no union_test.csv.",
        ],
    }

    behavior_test_targets = []

    for behavior_name, behavior_id in behavior_map.items():
        valid_targets = read_ampl_csv(src_dir / f"{behavior_name}_valid.csv", behavior_id=behavior_id)
        test_targets = read_ampl_csv(src_dir / f"{behavior_name}_test.csv", behavior_id=behavior_id)

        if dedup_behavior_eval:
            valid_targets = dedup_earliest_user_item_behavior(valid_targets)
            test_targets = dedup_earliest_user_item_behavior(test_targets)

        behavior_test_targets.append(test_targets)

        if train_scope == "union":
            train_df = train_union_df.copy()
        elif train_scope == "target":
            train_df = build_train_instances(union_train, max_len=max_len, target_behavior=behavior_id)
        else:
            raise ValueError("train_scope must be union or target")

        valid_df = build_eval_instances(
            history_pool=union_train,
            targets=valid_targets,
            max_len=max_len,
            train_items=train_items,
            remove_cold_items=remove_cold_items,
        )
        test_df = build_eval_instances(
            history_pool=pd.concat([union_train, union_valid], ignore_index=True),
            targets=test_targets,
            max_len=max_len,
            train_items=train_items,
            remove_cold_items=remove_cold_items,
        )

        ds = f"{prefix}_{behavior_name}"
        write_inter(train_df, out_root / ds / f"{ds}.train.inter")
        write_inter(valid_df, out_root / ds / f"{ds}.valid.inter")
        write_inter(test_df, out_root / ds / f"{ds}.test.inter")

        report["datasets"][ds] = {
            "target_behavior": behavior_name,
            "target_behavior_raw_id": int(behavior_id),
            "raw_valid_targets": stats_raw(valid_targets),
            "raw_test_targets": stats_raw(test_targets),
            "seq_train": stats_seq(train_df),
            "seq_valid": stats_seq(valid_df),
            "seq_test": stats_seq(test_df),
        }

        print(
            f"[{ds}] raw_valid={len(valid_targets):,} raw_test={len(test_targets):,} | "
            f"seq_train={len(train_df):,} seq_valid={len(valid_df):,} seq_test={len(test_df):,} "
            f"target={behavior_name}:{behavior_id} train_scope={train_scope}"
        )

    # Union/all dataset.
    union_valid_targets = union_valid.copy()
    union_test_targets = pd.concat(behavior_test_targets, ignore_index=True)
    union_test_targets = union_test_targets.sort_values(["user_id", "timestamp", "item_id", "item_type"]).reset_index(drop=True)

    union_valid_df = build_eval_instances(
        history_pool=union_train,
        targets=union_valid_targets,
        max_len=max_len,
        train_items=train_items,
        remove_cold_items=remove_cold_items,
    )
    union_test_df = build_eval_instances(
        history_pool=pd.concat([union_train, union_valid], ignore_index=True),
        targets=union_test_targets,
        max_len=max_len,
        train_items=train_items,
        remove_cold_items=remove_cold_items,
    )

    ds = f"{prefix}_union"
    write_inter(train_union_df, out_root / ds / f"{ds}.train.inter")
    write_inter(union_valid_df, out_root / ds / f"{ds}.valid.inter")
    write_inter(union_test_df, out_root / ds / f"{ds}.test.inter")

    report["datasets"][ds] = {
        "target_behavior": "union",
        "target_behavior_raw_id": None,
        "raw_valid_targets": stats_raw(union_valid_targets),
        "raw_test_targets": stats_raw(union_test_targets),
        "seq_train": stats_seq(train_union_df),
        "seq_valid": stats_seq(union_valid_df),
        "seq_test": stats_seq(union_test_df),
    }

    print(
        f"[{ds}] raw_valid={len(union_valid_targets):,} raw_test={len(union_test_targets):,} | "
        f"seq_train={len(train_union_df):,} seq_valid={len(union_valid_df):,} seq_test={len(union_test_df):,}"
    )

    report_path = out_root / f"{prefix}_conversion_report_v2.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--dataset_key", required=True, choices=["JD", "UB", "jd", "ub"])
    parser.add_argument("--out_root", default="dataset")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--train_scope", choices=["union", "target"], default="union")
    parser.add_argument("--behavior_map", default=None)

    # Default follows AMPL released processed files: keep rows as provided.
    parser.add_argument("--dedup_train_valid", action="store_true",
                        help="Apply earliest duplicate user-item-behavior filtering to union_train/union_valid. Off by default because AMPL processed files are assumed already processed.")
    parser.add_argument("--dedup_behavior_eval", action="store_true",
                        help="Apply earliest duplicate user-item-behavior filtering to behavior-specific valid/test files. Off by default.")
    parser.add_argument("--keep_cold_items", action="store_true",
                        help="Keep validation/test targets whose items are not in union_train. Default removes cold-start items.")

    args = parser.parse_args()
    behavior_map = parse_behavior_map(args.behavior_map, args.dataset_key)

    convert(
        src_dir=Path(args.src_dir),
        out_root=Path(args.out_root),
        prefix=args.prefix,
        behavior_map=behavior_map,
        max_len=args.max_len,
        train_scope=args.train_scope,
        dedup_train_valid=args.dedup_train_valid,
        dedup_behavior_eval=args.dedup_behavior_eval,
        remove_cold_items=not args.keep_cold_items,
    )


if __name__ == "__main__":
    main()
