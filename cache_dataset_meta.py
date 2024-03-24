import argparse
import random

from accelerate.utils import set_seed

import library.train_util as train_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def make_dataset(args):
    train_util.prepare_dataset_args(args, True)
    setup_logging(args, reset=True)

    use_dreambooth_method = args.in_json is None
    use_user_config = args.dataset_config is not None

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(
            ConfigSanitizer(True, True, False, True)
        )
        if use_user_config:
            logger.info(f"Loading dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "reg_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=None)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(
            blueprint.dataset_group
        )
    else:
        # use arbitrary dataset class
        train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer=None)
    return train_dataset_group


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args, unknown = parser.parse_known_args()
    args = train_util.read_config_from_file(args, parser)
    if args.max_token_length is None:
        args.max_token_length = 75
    args.cache_meta = True

    dataset_group = make_dataset(args)
