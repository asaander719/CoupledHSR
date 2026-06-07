#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-command ablation runner for CoupledHSR on AMPL-style RecBole sequence datasets.

Example:
python run_coupledhsr_ablation.py \
  --model CoupledHSR --dataset_prefix ampl_jd --behaviors purchase \
  --base_config configs/model/CoupledHSR_ABLATION_BASE.yaml -g 0 --stats_level quick

Default variants:
A0_anchor_only, A2_no_coupling, A3_symmetric_coupling,
A4_static_causal, A5_full_coupledhsr
"""
from __future__ import annotations
import argparse, copy, csv, json, time
from pathlib import Path
from logging import getLogger
from typing import Any, Dict, List

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, get_model, get_environment, set_color

try:
    from recbole.utils.enum_type import ModelType
except Exception:
    from recbole.utils import ModelType

VARIANTS: Dict[str, Dict[str, Any]] = {
    "A0_anchor_only": {
        "use_transformer": True, "use_hnn": False, "coupling_mode": "none",
        "n_coupling_bands": 1, "hnn_residual_init": -5.0,
        "note": "Causal sequence anchor only; removes Hamiltonian residual branch."
    },
    "A1_hnn_only": {
        "use_transformer": False, "use_hnn": True, "coupling_mode": "causal",
        "n_coupling_bands": 8, "hnn_residual_init": -5.0,
        "note": "Frequency-coupled HNN branch without Transformer anchor."
    },
    "A2_no_coupling": {
        "use_transformer": True, "use_hnn": True, "coupling_mode": "none",
        "n_coupling_bands": 1, "hnn_residual_init": -5.0,
        "note": "Keeps HNN residual but removes cross-behavior coupling."
    },
    "A3_symmetric_coupling": {
        "use_transformer": True, "use_hnn": True, "coupling_mode": "symmetric",
        "n_coupling_bands": 8, "hnn_residual_init": -5.0,
        "note": "Replaces causal behavior transfer with symmetric coupling."
    },
    "A4_static_causal": {
        "use_transformer": True, "use_hnn": True, "coupling_mode": "causal",
        "n_coupling_bands": 1, "hnn_residual_init": -5.0,
        "note": "Causal coupling with one spectral band; removes frequency adaptation."
    },
    "A5_full_coupledhsr": {
        "use_transformer": True, "use_hnn": True, "coupling_mode": "causal",
        "n_coupling_bands": 8, "hnn_residual_init": -5.0,
        "note": "Full frequency-adaptive causal Hamiltonian residual."
    },
    "A6_stronger_residual": {
        "use_transformer": True, "use_hnn": True, "coupling_mode": "causal",
        "n_coupling_bands": 8, "hnn_residual_init": -4.0,
        "note": "Full model with a stronger initial residual gate."
    },
}
DEFAULT_VARIANTS = ["A0_anchor_only","A2_no_coupling","A3_symmetric_coupling","A4_static_causal","A5_full_coupledhsr"]
TARGET_TOKEN = {"click":1, "comment":2, "cart":2, "favourite":3, "favorite":3, "purchase":4, "union":4}

def cfg_get(config, key, default=None):
    try: return config[key]
    except Exception: return default

def plain(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item() if x.numel()==1 else x.detach().cpu().tolist()
    except Exception: pass
    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
    except Exception: pass
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [plain(v) for v in x]
    try:
        json.dumps(x); return x
    except Exception:
        return str(x)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--model","-m",default="CoupledHSR")
    p.add_argument("--dataset_prefix",required=True)
    p.add_argument("--behaviors",default="purchase")
    p.add_argument("--base_config",required=True)
    p.add_argument("--gpu_id","-g",type=int,default=0)
    p.add_argument("--variants",default="default")
    p.add_argument("--max_len",type=int,default=50)
    p.add_argument("--valid_metric",default="Recall@10")
    p.add_argument("--stats_level",choices=["none","quick","full"],default="quick")
    p.add_argument("--results_dir",default="saved/coupledhsr_ablation")
    p.add_argument("--epochs",type=int,default=None)
    p.add_argument("--stopping_step",type=int,default=None)
    p.add_argument("--seed",type=int,default=None)
    p.add_argument("--show_progress",type=int,default=None)
    p.add_argument("--dry_run",action="store_true")
    return p.parse_args()

def dataset_name(prefix, behavior): return f"{prefix}_{behavior}"

def base_dict(args, behavior):
    d={
        "gpu_id":args.gpu_id, "MODEL_TYPE":ModelType.SEQUENTIAL,
        "benchmark_filename":["train","valid","test"],
        "USER_ID_FIELD":"user_id", "ITEM_ID_FIELD":"item_id", "TIME_FIELD":"timestamp",
        "ITEM_LIST_LENGTH_FIELD":"item_length", "LIST_SUFFIX":"_list", "MAX_ITEM_LIST_LENGTH":args.max_len,
        "alias_of_item_id":["item_id_list"], "alias_of_item_type":["item_type_list"],
        "ITEM_ID_LIST_FIELD":"item_id_list", "ITEM_TYPE_LIST_FIELD":"item_type_list",
        "field_separator":"\t", "seq_separator":" ",
        "load_col":{"inter":["user_id","item_id_list","item_type_list","item_length","item_id","timestamp","item_type"]},
        "ITEM_TYPE_SEQ_FIELD":"item_type_list", "ITEM_TYPE_FIELD":"item_type",
        "target_behavior_token":TARGET_TOKEN.get(behavior.lower(),4),
        "mask_behavior_as_target":True, "num_behaviors":4,
        "eval_args":{"order":"TO","mode":"full"},
        "train_neg_sample_args":None, "metrics":["Recall","NDCG","MRR"], "topk":[10,20],
        "valid_metric":args.valid_metric, "hyper_parameters":[]
    }
    if args.epochs is not None: d["epochs"]=args.epochs
    if args.stopping_step is not None: d["stopping_step"]=args.stopping_step
    if args.seed is not None: d["seed"]=args.seed
    if args.show_progress is not None: d["show_progress"]=bool(args.show_progress)
    return d

def perf_stats(model, eval_data, config, logger, level):
    if level=="none": return {}
    try:
        if level=="quick":
            from performance_stats import get_quick_stats
            return get_quick_stats(model, eval_data, config, logger)
        from performance_stats import get_comprehensive_stats
        return get_comprehensive_stats(model, eval_data, config, logger)
    except Exception as e:
        logger.warning(set_color(f"Performance stats failed: {e}", "yellow"))
        return {"error":str(e)}

def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(plain(obj), ensure_ascii=False)+"\n")

def flatten(d, prefix=""):
    out={}
    for k,v in d.items():
        kk=f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict): out.update(flatten(v, kk))
        else: out[kk]=plain(v)
    return out

def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows=[flatten(r) for r in records]
    keys=sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

def run_one(args, behavior, vname, vcfg, outdir, idx, total, config_files):
    logger=getLogger()
    d=base_dict(args, behavior); d.update(copy.deepcopy(vcfg))
    d["ablation_variant"]=vname; d["hyper_parameters"]=[]
    config=Config(model=args.model, dataset=dataset_name(args.dataset_prefix, behavior),
                  config_file_list=config_files, config_dict=d)
    config.final_config_dict["MODEL_TYPE"]=ModelType.SEQUENTIAL
    config.final_config_dict["LIST_SUFFIX"]="_list"
    config.final_config_dict["ITEM_ID_LIST_FIELD"]="item_id_list"
    config.final_config_dict["ITEM_TYPE_LIST_FIELD"]="item_type_list"
    config.final_config_dict["alias_of_item_id"]=["item_id_list"]
    config.final_config_dict["alias_of_item_type"]=["item_type_list"]
    config.final_config_dict["ablation_variant"]=vname
    config.final_config_dict["target_behavior_token"]=d["target_behavior_token"]
    init_seed(config["seed"], config["reproducibility"])
    logger.info("\n"+"="*90)
    logger.info(set_color(f"Ablation {idx}/{total}: {behavior} | {vname}", "cyan"))
    logger.info(vcfg); logger.info(config)
    dataset=create_dataset(config); logger.info(dataset)
    train_data, valid_data, test_data=data_preparation(config, dataset)
    model=get_model(config["model"])(config, train_data.dataset).to(config["device"])
    logger.info(model)
    trainer=Trainer(config, model)
    t0=time.time()
    best_score,best_valid=trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])
    fit_sec=time.time()-t0
    t1=time.time()
    test_result=trainer.evaluate(test_data, show_progress=config["show_progress"])
    test_sec=time.time()-t1
    stats=perf_stats(model, test_data, config, logger, args.stats_level)
    logger.info(set_color("best valid result", "yellow")+f": {best_valid}")
    logger.info(set_color("test result", "yellow")+f": {test_result}")
    logger.info("Environment:\n"+get_environment(config).draw())
    rec={
        "model":args.model, "dataset":dataset_name(args.dataset_prefix, behavior),
        "behavior":behavior, "variant":vname, "note":vcfg.get("note",""),
        "variant_config":vcfg, "best_valid_score":plain(best_score),
        "best_valid_result":plain(best_valid), "test_result":plain(test_result),
        "fit_sec":fit_sec, "test_sec":test_sec, "performance":plain(stats),
        "config_snapshot":{
            "hidden_size":cfg_get(config,"hidden_size"), "num_layers":cfg_get(config,"num_layers"),
            "learning_rate":cfg_get(config,"learning_rate"), "use_transformer":cfg_get(config,"use_transformer"),
            "use_hnn":cfg_get(config,"use_hnn"), "coupling_mode":cfg_get(config,"coupling_mode"),
            "n_coupling_bands":cfg_get(config,"n_coupling_bands"),
            "hnn_residual_init":cfg_get(config,"hnn_residual_init")
        }
    }
    append_jsonl(outdir/"ablation_records.jsonl", rec)
    return rec

def main():
    args=parse_args()
    behaviors=[x.strip() for x in args.behaviors.split(",") if x.strip()]
    if args.variants=="default": vnames=DEFAULT_VARIANTS
    else: vnames=[x.strip() for x in args.variants.split(",") if x.strip()]
    variants={k:VARIANTS[k] for k in vnames}
    config_files=args.base_config.strip().split()
    init_config=Config(model=args.model, dataset=dataset_name(args.dataset_prefix, behaviors[0]),
                       config_file_list=config_files, config_dict=base_dict(args, behaviors[0]))
    init_config.final_config_dict["MODEL_TYPE"]=ModelType.SEQUENTIAL
    init_config.final_config_dict["LIST_SUFFIX"]="_list"
    init_config.final_config_dict["ITEM_ID_LIST_FIELD"]="item_id_list"
    init_config.final_config_dict["ITEM_TYPE_LIST_FIELD"]="item_type_list"
    init_config.final_config_dict["alias_of_item_id"]=["item_id_list"]
    init_config.final_config_dict["alias_of_item_type"]=["item_type_list"]
    init_logger(init_config)
    logger=getLogger()
    outdir=Path(args.results_dir)/args.model/args.dataset_prefix/time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"args.json").write_text(json.dumps(vars(args),indent=2,ensure_ascii=False),encoding="utf-8")
    (outdir/"variants.json").write_text(json.dumps(plain(variants),indent=2,ensure_ascii=False),encoding="utf-8")
    plan=[(b,v,variants[v]) for b in behaviors for v in vnames]
    logger.info(f"Total runs: {len(plan)}")
    for i,(b,v,cfg) in enumerate(plan,1): logger.info(f"[plan] {i}: {dataset_name(args.dataset_prefix,b)} {v}")
    if args.dry_run: return
    records=[]
    for i,(b,v,cfg) in enumerate(plan,1):
        records.append(run_one(args,b,v,cfg,outdir,i,len(plan),config_files))
        write_csv(outdir/"ablation_summary.csv", records)
    logger.info(set_color("Ablation finished", "green"))
    logger.info(f"Result directory: {outdir}")

if __name__=="__main__":
    main()
