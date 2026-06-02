#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert AMPL processed_data_JD / processed_data_UB CSVs into RecBole
SequentialDataset benchmark-format .inter files.

Input AMPL files are expected to be headerless CSV:
    user_id,item_id,timestamp,behavior_id

Example:
python convert_ampl_to_recbole_seq.py \
  --src_dir /home/asaliao/AMPL/processed_data_JD \
  --dataset_key JD \
  --out_root /home/asaliao/RecBole/dataset \
  --prefix ampl_jd \
  --max_len 50

This creates:
  dataset/ampl_jd_purchase/ampl_jd_purchase.train.inter
  dataset/ampl_jd_purchase/ampl_jd_purchase.valid.inter
  dataset/ampl_jd_purchase/ampl_jd_purchase.test.inter
and similarly for other behaviors.

Each row is a next-target-behavior prediction instance:
    history = all previous behaviors/items
    label   = next item under the target behavior
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_BEHAVIORS = {
    # AMPL paper keeps examination/click, comment, favorite, purchase for JD.
    # AMPL CSVs use integer behavior ids. This is the common order in the released files.
    "JD": {
        "click": 0,
        "comment": 1,
        "favourite": 2,
        "purchase": 3,
    },
    # AMPL paper keeps examination/click, cart, favorite, purchase for UB.
    "UB": {
        "click": 0,
        "cart": 1,
        "favourite": 2,
        "purchase": 3,
    },
}


def parse_behavior_map(s: str | None, dataset_key: str) -> Dict[str, int]:
    if not s:
        return dict(DEFAULT_BEHAVIORS[dataset_key.upper()])
    out = {}
    for part in s.split(","):
        name, value = part.split(":")
        out[name.strip()] = int(value)
    return out


def read_ampl_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing AMPL file: {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 4:
        raise ValueError(f"{path} should have at least 4 columns: user,item,timestamp,behavior")
    df = df.iloc[:, :4].copy()
    df.columns = ["user_id", "item_id", "timestamp", "item_type"]
    # Keep raw ids as strings/tokens for RecBole, but sort timestamps numerically.
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype("int64")
    df["item_type"] = pd.to_numeric(df["item_type"], errors="coerce").fillna(0).astype("int64")
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def build_history_index(history_df: pd.DataFrame) -> Dict[str, List[Tuple[int, str, int]]]:
    """user -> sorted list of (timestamp, item_id, behavior_id)."""
    hist = {}
    for u, g in history_df.sort_values(["user_id", "timestamp"]).groupby("user_id", sort=False):
        hist[u] = list(zip(g["timestamp"].tolist(), g["item_id"].tolist(), g["item_type"].tolist()))
    return hist


def make_row(u: str, target_item: str, target_ts: int, target_type: int,
             hist_events: List[Tuple[int, str, int]], max_len: int):
    if len(hist_events) <= 0:
        return None
    hist_events = hist_events[-max_len:]
    items = [it for _, it, _ in hist_events]
    types = [str(bt) for _, _, bt in hist_events]
    return {
        "user_id:token": u,
        "item_id_list:token_seq": " ".join(items),
        "item_type_list:token_seq": " ".join(types),
        "item_length:float": len(items),
        "item_id:token": target_item,
        "timestamp:float": int(target_ts),
        "item_type:token": str(int(target_type)),
    }


def build_train_instances(union_train: pd.DataFrame, target_type: int, max_len: int) -> pd.DataFrame:
    rows = []
    for u, g in union_train.sort_values(["user_id", "timestamp"]).groupby("user_id", sort=False):
        hist = []
        for _, r in g.iterrows():
            cur = (int(r["timestamp"]), str(r["item_id"]), int(r["item_type"]))
            if int(r["item_type"]) == int(target_type):
                row = make_row(
                    u=u,
                    target_item=str(r["item_id"]),
                    target_ts=int(r["timestamp"]),
                    target_type=target_type,
                    hist_events=hist,
                    max_len=max_len,
                )
                if row is not None:
                    rows.append(row)
            hist.append(cur)
    return pd.DataFrame(rows)


def build_eval_instances(history_pool: pd.DataFrame, target_df: pd.DataFrame,
                         target_type: int, max_len: int) -> pd.DataFrame:
    rows = []
    hist_index = build_history_index(history_pool)
    for _, r in target_df.sort_values(["user_id", "timestamp"]).iterrows():
        u = str(r["user_id"])
        ts = int(r["timestamp"])
        events = hist_index.get(u, [])
        # Use only events before the target timestamp to avoid leakage.
        # Linear scan is acceptable for these processed datasets.
        hist = [e for e in events if e[0] < ts]
        row = make_row(
            u=u,
            target_item=str(r["item_id"]),
            target_ts=ts,
            target_type=target_type,
            hist_events=hist,
            max_len=max_len,
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def write_inter(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "user_id:token",
        "item_id_list:token_seq",
        "item_type_list:token_seq",
        "item_length:float",
        "item_id:token",
        "timestamp:float",
        "item_type:token",
    ]
    if df.empty:
        # Still write a valid header so RecBole errors clearly later.
        pd.DataFrame(columns=cols).to_csv(path, sep="\t", index=False)
    else:
        df[cols].to_csv(path, sep="\t", index=False)


def convert_one(src_dir: Path, out_root: Path, prefix: str,
                behavior_name: str, behavior_id: int, max_len: int):
    union_train = read_ampl_csv(src_dir / "union_train.csv")
    union_valid = read_ampl_csv(src_dir / "union_valid.csv")
    valid_file = src_dir / f"{behavior_name}_valid.csv"
    test_file = src_dir / f"{behavior_name}_test.csv"

    valid_targets = read_ampl_csv(valid_file)
    test_targets = read_ampl_csv(test_file)

    # Make the target id explicit; this also guards against files whose behavior column is not normalized.
    valid_targets["item_type"] = int(behavior_id)
    test_targets["item_type"] = int(behavior_id)

    train_df = build_train_instances(union_train, behavior_id, max_len=max_len)
    valid_df = build_eval_instances(union_train, valid_targets, behavior_id, max_len=max_len)
    test_history = pd.concat([union_train, union_valid], ignore_index=True)
    test_df = build_eval_instances(test_history, test_targets, behavior_id, max_len=max_len)

    dataset_name = f"{prefix}_{behavior_name}"
    out_dir = out_root / dataset_name
    write_inter(train_df, out_dir / f"{dataset_name}.train.inter")
    write_inter(valid_df, out_dir / f"{dataset_name}.valid.inter")
    write_inter(test_df, out_dir / f"{dataset_name}.test.inter")

    print(f"[{dataset_name}] train={len(train_df):,} valid={len(valid_df):,} test={len(test_df):,} "
          f"target_behavior={behavior_name}:{behavior_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="AMPL processed_data_JD or processed_data_UB directory")
    parser.add_argument("--dataset_key", required=True, choices=["JD", "UB", "jd", "ub"])
    parser.add_argument("--out_root", default="dataset", help="RecBole dataset directory")
    parser.add_argument("--prefix", required=True, help="Output dataset prefix, e.g. ampl_jd or ampl_ub")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument(
        "--behavior_map",
        default=None,
        help="Optional override, e.g. click:0,comment:1,favourite:2,purchase:3",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_root = Path(args.out_root)
    behavior_map = parse_behavior_map(args.behavior_map, args.dataset_key)

    for name, bid in behavior_map.items():
        convert_one(src_dir, out_root, args.prefix, name, bid, args.max_len)


if __name__ == "__main__":
    main()
