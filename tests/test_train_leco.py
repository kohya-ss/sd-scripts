import train_leco
from library import deepspeed_utils, train_util


def test_syntax():
    assert train_leco is not None


def test_setup_parser_supports_shared_training_validation():
    args = train_leco.setup_parser().parse_args(["--prompts_file", "slider.yaml"])

    train_util.verify_training_args(args)

    assert args.min_snr_gamma is None
    assert deepspeed_utils.prepare_deepspeed_plugin(args) is None
