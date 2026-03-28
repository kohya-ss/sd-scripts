import sdxl_train_leco
from library import deepspeed_utils, sdxl_train_util, train_util


def test_syntax():
    assert sdxl_train_leco is not None


def test_setup_parser_supports_shared_training_validation():
    args = sdxl_train_leco.setup_parser().parse_args(["--prompts_file", "slider.yaml"])

    train_util.verify_training_args(args)
    sdxl_train_util.verify_sdxl_training_args(args, support_text_encoder_caching=False)

    assert args.min_snr_gamma is None
    assert deepspeed_utils.prepare_deepspeed_plugin(args) is None
