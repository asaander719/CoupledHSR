# -*- coding: utf-8 -*-
"""
Clean full-ranking runner for CoupledHSR A6.

Usage examples:
    python run_a6_full.py --dataset retail_beh --gpu_id 0
    python run_a6_full.py --dataset tmall_beh --gpu_id 0
    python run_a6_full.py --dataset ijcai_beh --gpu_id 0 --train_batch_size 24 --eval_batch_size 64

This runner intentionally avoids MBHT's customized 100+1 candidate protocol.
It uses full_sort_predict() and RecBole's full-ranking evaluator.
"""

import argparse
import copy
import itertools
import sys
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import create_samplers, get_dataloader
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color, get_environment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="CoupledHSR",
                        help="Use CoupledHSR if you copied CoupledHSRA6.py over coupledhsr.py.")
    parser.add_argument("--dataset", "-d", type=str, default="retail_beh")
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument("--gpu_id", "-g", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--validation", action="store_true",
                        help="Split train into train/valid; otherwise fit with test set as RecBole valid_data, like MBHT runner.")
    parser.add_argument("--valid_portion", type=float, default=0.1)
    return parser.parse_args()


def hyper_grid(config):
    hp = config["hyper_parameters"] if "hyper_parameters" in config else None
    if not hp:
        return [None]
    values = []
    for k in hp:
        v = config[k]
        values.append(v if isinstance(v, (list, tuple)) else [v])
    return [dict(zip(hp, comb)) for comb in itertools.product(*values)]


def main():
    args = parse_args()
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else [f"configs/model/{args.model}.yaml"]
    )


    override = {
        "gpu_id": args.gpu_id,
        # Clean full-ranking, not customized 100+1.
        "eval_args": {"mode": "full", "order": "TO"},
        "customized_eval": 0,
        "benchmark_filename": ["train", "test"],
        "USER_ID_FIELD": "session_id",
        "alias_of_item_id": ["item_id_list"],
        "load_col": None,
        "train_neg_sample_args": None,
        "neg_sampling": None,
    }
    if args.train_batch_size is not None:
        override["train_batch_size"] = args.train_batch_size
    if args.eval_batch_size is not None:
        override["eval_batch_size"] = args.eval_batch_size

    base_config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=override,
    )

    init_logger(base_config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(base_config)

    for run_idx, hp in enumerate(hyper_grid(base_config), 1):
        run_override = copy.deepcopy(override)
        if hp:
            run_override.update(hp)

        config = Config(
            model=args.model,
            dataset=args.dataset,
            config_file_list=config_file_list,
            config_dict=run_override,
        )
        init_seed(config["seed"], config["reproducibility"])

        logger.info("\\n" + "=" * 80)
        logger.info(set_color(f"A6 full-ranking run {run_idx}", "cyan"))
        logger.info(f"Hyperparameters: {hp}")
        logger.info(config)

        dataset = create_dataset(config)
        logger.info(dataset)

        built = dataset.build()
        if len(built) == 2:
            train_dataset, test_dataset = built
            train_sampler, valid_sampler, test_sampler = create_samplers(
                config, dataset, [train_dataset, test_dataset]
            )
            if args.validation:
                train_dataset.shuffle()
                train_dataset, valid_dataset = train_dataset.split_by_ratio(
                    [1 - args.valid_portion, args.valid_portion]
                )
                train_data = get_dataloader(config, "train")(config, train_dataset, None, shuffle=True)
                valid_data = get_dataloader(config, "valid")(config, valid_dataset, None, shuffle=False)
                test_data = get_dataloader(config, "test")(config, test_dataset, test_sampler, shuffle=False)
            else:
                train_data = get_dataloader(config, "train")(config, train_dataset, train_sampler, shuffle=True)
                valid_data = get_dataloader(config, "valid")(config, test_dataset, test_sampler, shuffle=False)
                test_data = valid_data
        else:
            # Fallback for datasets with train/valid/test splits already prepared.
            train_dataset, valid_dataset, test_dataset = built
            train_sampler, valid_sampler, test_sampler = create_samplers(
                config, dataset, [train_dataset, valid_dataset, test_dataset]
            )
            train_data = get_dataloader(config, "train")(config, train_dataset, train_sampler, shuffle=True)
            valid_data = get_dataloader(config, "valid")(config, valid_dataset, valid_sampler, shuffle=False)
            test_data = get_dataloader(config, "test")(config, test_dataset, test_sampler, shuffle=False)

        model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
        logger.info(model)

        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=config["show_progress"]
        )
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])

        logger.info(set_color("best valid score", "yellow") + f": {best_valid_score}")
        logger.info(set_color("best valid result", "yellow") + f": {best_valid_result}")
        logger.info(set_color("test result", "yellow") + f": {test_result}")
        logger.info("Environment:\\n" + get_environment(config).draw())


if __name__ == "__main__":
    main()
