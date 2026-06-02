import sys
import os
import argparse
import copy
import itertools
from logging import getLogger
from recbole.data.utils import get_dataloader, create_samplers
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.utils import get_dataloader
from recbole.data.transform import construct_transform
from recbole.trainer import Trainer
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CoupledHSR', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='retail_beh', help='Benchmarks for session-based rec.')
    parser.add_argument('--validation', action='store_true', help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    parser.add_argument('--gpu_id','-g', type=int, default=0)
    parser.add_argument('--batch_size','-bs', type=int, default=2048)
    parser.add_argument('--enable_hg', type=int, default=1)
    parser.add_argument('--enable_ms', type=int, default=1)
    parser.add_argument('--enable_en', type=int, default=1)
    parser.add_argument('--customized_eval', type=int, default=1)
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument("--coupling_mode", type=str, default="symmetric", help="none or symmetric or causal")
    return parser.parse_known_args()[0]


def get_hyperparameter_grid(config):
    hyper_parameters = config['hyper_parameters']
    if not hyper_parameters:
        return [None]
    if not isinstance(hyper_parameters, (list, tuple)):
        raise ValueError("hyper_parameters must be a list of config keys")

    hyper_values = []
    for hp in hyper_parameters:
        if hp not in config:
            raise ValueError(f"Hyperparameter '{hp}' is not defined in the config")
        values = config[hp]
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"Hyperparameter '{hp}' must be configured as a list of values to sweep"
            )
        hyper_values.append(list(values))

    grid = []
    for reversed_values in itertools.product(*hyper_values[::-1]):
        combo = {
            hp: value
            for hp, value in zip(reversed(hyper_parameters), reversed_values)
        }
        grid.append(combo)
    return grid


def train_model(config, args):
    logger = getLogger()
    valid_data = None

    if args.model in ("MBHT", "HSR", "CoupledHSR"):
        dataset = create_dataset(config)
        logger.info(dataset)

        train_dataset, test_dataset = dataset.build()
        train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, [train_dataset, test_dataset])
        if args.validation:
            train_dataset.shuffle()
            new_train_dataset, new_test_dataset = train_dataset.split_by_ratio(
                [1 - args.valid_portion, args.valid_portion]
            )
            train_data = get_dataloader(config, 'train')(
                config, new_train_dataset, None, shuffle=True
            )
            test_data = get_dataloader(config, 'test')(
                config, new_test_dataset, None, shuffle=False
            )
        else:
            train_data = get_dataloader(config, 'train')(
                config, train_dataset, train_sampler, shuffle=True
            )
            test_data = get_dataloader(config, 'test')(
                config, test_dataset, test_sampler, shuffle=False
            )

        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        logger.info(model)

        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        test_score, test_result = trainer.fit(
            train_data, test_data, saved=True, show_progress=config['show_progress']
        )
    else:
        dataset = create_dataset(config)
        logger.info(dataset)

        train_data, valid_data, test_data = data_preparation(config, dataset)

        try:
            local_rank = config['local_rank']
        except Exception:
            local_rank = 0
        if local_rank is None:
            local_rank = 0
        init_seed(config['seed'] + local_rank, config['reproducibility'])

        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        logger.info(model)

        transform = construct_transform(config)
        flops = get_flops(model, dataset, config['device'], logger, transform)
        logger.info(set_color('FLOPs', 'blue') + f": {flops}")

        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=config['show_progress']
        )
        test_result = trainer.evaluate(test_data, show_progress=config['show_progress'])

    return model, test_data, valid_data, test_result


'''
RETAIL-BEH dataset behavior distribution :
ID 0:      0  → [PAD] / masked target
ID 2: 337,651 → view           (most common)
ID 1:  99,011 → addtocart      (second)
ID 3:  48,184 → fav / collect   (third)
ID 4:   6,708 → transaction    (rarest = buy)
'''

if __name__ == '__main__':
    args = get_args()
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else [f"configs/model/{args.model}.yaml"]
    )

    # configurations initialization
    config_dict = {
        'USER_ID_FIELD': 'session_id',
        'load_col': None,
        # 'neg_sampling': {'uniform':1},
        'neg_sampling': None,
        'benchmark_filename': ['train', 'test'],
        'alias_of_item_id': ['item_id_list'],
        'topk': [5, 10, 20],
        'metrics': ['Recall', 'NDCG', 'MRR'],
        'valid_metric': 'NDCG@10',
        'eval_args':{
            'mode':'full',
            'order':'TO'
            },
        'gpu_id':args.gpu_id,
        "MAX_ITEM_LIST_LENGTH":200,
        "train_batch_size": 24 if args.dataset == "ijcai_beh" else 64, #36
        "eval_batch_size":24 if args.dataset == "ijcai_beh" else 128,
        "hyper_len":10 if args.dataset == "ijcai_beh" else 6,
        "scales":[10, 4, 20],
        "enable_hg":1,  # 1,
        "enable_ms":1,  # 1,
        "enable_en":0,  # 1,
        "customized_eval":1,  # 1,
        "abaltion":"",
        "train_neg_sample_args": None,
    }

    if args.dataset == "retail_beh":
        config_dict['scales'] = [5, 4, 20]
        config_dict['hyper_len'] = 6

    # load config for the chosen model from configs/model/<model>.yaml unless overridden
    config = Config(model=args.model, dataset=f'{args.dataset}', config_file_list=config_file_list, config_dict=config_dict)

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    total_runs = len(get_hyperparameter_grid(config))
    for run_idx, hp_override in enumerate(get_hyperparameter_grid(config), start=1):
        run_config_dict = copy.deepcopy(config_dict)
        if hp_override:
            run_config_dict.update(hp_override)

        run_config = Config(model=args.model, dataset=f'{args.dataset}', config_file_list=config_file_list, config_dict=run_config_dict)
        logger.info("\n" + "=" * 80)
        logger.info(set_color(f"Hyperparameter run {run_idx}/{total_runs}", "cyan"))
        logger.info(f"Hyperparameter values: {hp_override}")
        logger.info(run_config)

        model, test_data, valid_data, test_result = train_model(run_config, args)
        eval_data = valid_data if valid_data is not None else test_data

        logger.info("\n" + "=" * 80)
        logger.info(set_color("Quick Performance Check", "cyan"))
        logger.info("=" * 80)
        logger.info("Running quick performance check after training...")
        from performance_stats import get_quick_stats
        try:
            quick_stats = get_quick_stats(model, eval_data, run_config, logger)
            logger.info(f"✓ Model Parameters: {quick_stats['parameters']['total_params_m']:.2f}M")
            if 'allocated_gb' in quick_stats['memory'] and quick_stats['memory']['allocated_gb'] > 0:
                logger.info(f"✓ GPU Memory: {quick_stats['memory']['allocated_gb']:.2f} GB")
            elif 'note' in quick_stats['memory']:
                logger.info(f"✓ {quick_stats['memory']['note']}")
            if quick_stats['inference_latency']['samples_tested'] > 0:
                logger.info(f"✓ Quick Inference Test: {quick_stats['inference_latency']['avg_latency_ms']:.3f} ms/user ({quick_stats['inference_latency']['samples_tested']} samples)")
            logger.info(set_color("Quick check passed!", "green"))
        except Exception as e:
            logger.warning(set_color(f"Quick check failed: {e}", "yellow"))
            logger.warning("Continuing with training anyway...")
        logger.info("=" * 80 + "\n")

        environment_tb = get_environment(run_config)
        logger.info(
            "The running environment of this training is as follows:\n"
            + environment_tb.draw()
        )

        logger.info(set_color("test result", "yellow") + f": {test_result}")

        logger.info("\n" + "=" * 80)
        logger.info(set_color("Performance Statistics", "green"))
        logger.info("=" * 80)
        # from performance_stats import get_comprehensive_stats
        # performance_stats = get_comprehensive_stats(model, test_data, run_config, logger)

        # logger.info("\n" + "=" * 80)
        # logger.info(set_color("Performance Summary", "green"))
        # logger.info("=" * 80)
        # logger.info(f"Model Parameters: {performance_stats['parameters']['total_params_m']:.2f}M")
        # if 'allocated_gb' in performance_stats['memory']:
        #     logger.info(f"GPU Memory: {performance_stats['memory']['allocated_gb']:.2f} GB")
        # logger.info(f"Average Inference Latency: {performance_stats['inference_latency']['avg_latency_ms']:.3f} ms/user")
        # logger.info(f"Throughput (QPS): {performance_stats['inference_latency']['qps']:.2f} queries/sec")
        # logger.info(f"Full Sort Latency: {performance_stats['full_sort_latency']['avg_latency_ms']:.3f} ms/user")
        # if 'speedup_ratio' in performance_stats['speedup_analysis']:
        #     logger.info(f"Speedup Ratio (Inference/Train): {performance_stats['speedup_analysis']['speedup_ratio']:.2f}x")
        # logger.info("=" * 80)

        # if run_idx < total_runs:
        #     logger.info(set_color(f"Completed hyperparameter run {run_idx}/{total_runs}. Moving to next set.", "blue"))
