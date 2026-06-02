# -*- coding: utf-8 -*-
"""
Run RecBole full-ranking evaluation on AMPL-converted sequential benchmark files.

Example:
python run_ampl_seq_full.py \
  --model CoupledHSR \
  --dataset ampl_jd_purchase \
  --config_files configs/model/CoupledHSR_AMPL_SEQ.yaml \
  -g 0
"""
import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, get_model, get_environment, set_color

try:
    from recbole.utils.enum_type import ModelType
except Exception:
    from recbole.utils import ModelType


DEFAULT_TARGET = {
    "click": 0,
    "comment": 1,
    "cart": 1,
    "favourite": 2,
    "favorite": 2,
    "purchase": 3,
}


def infer_target_behavior(dataset_name: str, explicit):
    if explicit is not None:
        return int(explicit)
    suffix = dataset_name.split("_")[-1].lower()
    return DEFAULT_TARGET.get(suffix, 3)


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
    parser.add_argument("--valid_metric", type=str, default="Recall@10")
    return parser.parse_args()


def main():
    args = parse_args()
    config_file_list =(args.config_files.strip().split(" ") if args.config_files else [f"configs/model/{args.model}.yaml"])
    target_behavior = infer_target_behavior(args.dataset, args.target_behavior_token)

    config_dict = {
        "gpu_id": args.gpu_id,
        "MODEL_TYPE": ModelType.SEQUENTIAL,

        # Benchmark files already contain sequence features.
        "benchmark_filename": ["train", "valid", "test"],

        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "LIST_SUFFIX": "_list",
        "MAX_ITEM_LIST_LENGTH": args.max_len,

        # Critical for RecBole SequentialDataset benchmark presets.
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
        "valid_metric": args.valid_metric,

        # CoupledHSR A6 target behavior for mask-token type.
        "target_behavior_token": target_behavior,
        "mask_behavior_as_target": True,
        "num_behaviors": 4,
    }
    if args.train_batch_size is not None:
        config_dict["train_batch_size"] = args.train_batch_size
    if args.eval_batch_size is not None:
        config_dict["eval_batch_size"] = args.eval_batch_size

    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    config.final_config_dict["MODEL_TYPE"] = ModelType.SEQUENTIAL
    config.final_config_dict["target_behavior_token"] = target_behavior

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(f"AMPL sequential benchmark dataset={args.dataset}, target_behavior={target_behavior}")
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    logger.info(model)

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    test_result = trainer.evaluate(test_data, show_progress=config["show_progress"])

    logger.info(set_color("best valid score", "yellow") + f": {best_valid_score}")
    logger.info(set_color("best valid result", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    logger.info("Environment:\n" + get_environment(config).draw())


if __name__ == "__main__":
    main()
