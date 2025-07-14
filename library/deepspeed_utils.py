import os
import argparse
import torch
from accelerate import DeepSpeedPlugin
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def add_deepspeed_arguments(parser: argparse.ArgumentParser):
    # DeepSpeed Arguments. https://huggingface.co/docs/accelerate/usage_guides/deepspeed
    parser.add_argument("--deepspeed", action="store_true", help="enable deepspeed training")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3], help="Possible options are 0,1,2,3.")
    parser.add_argument(
        "--offload_optimizer_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme. Only applicable with ZeRO Stages 2 and 3.",
    )
    parser.add_argument(
        "--offload_optimizer_nvme_path",
        type=str,
        default=None,
        help="Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--offload_param_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--offload_param_nvme_path",
        type=str,
        default=None,
        help="Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--zero3_init_flag",
        action="store_true",
        help="Flag to indicate whether to enable `deepspeed.zero.Init` for constructing massive models."
        "Only applicable with ZeRO Stage-3.",
    )
    parser.add_argument(
        "--zero3_save_16bit_model",
        action="store_true",
        help="Flag to indicate whether to save 16-bit model. Only applicable with ZeRO Stage-3.",
    )
    parser.add_argument(
        "--fp16_master_weights_and_gradients",
        action="store_true",
        help="fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32.",
    )



def prepare_deepspeed_plugin(args: argparse.Namespace):
    if not args.deepspeed:
        return None

    try:
        import deepspeed
    except ImportError as e:
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command. DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_stage,
        #gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_accumulation_steps=1,  # Overriding to a fixed value for compatibility with accelerate
        gradient_clipping=args.max_grad_norm,
        offload_optimizer_device=args.offload_optimizer_device,
        offload_optimizer_nvme_path=args.offload_optimizer_nvme_path,
        offload_param_device=args.offload_param_device,
        offload_param_nvme_path=args.offload_param_nvme_path,
        zero3_init_flag=args.zero3_init_flag,
        zero3_save_16bit_model=args.zero3_save_16bit_model,
    )
    #  Initialize a dictionary for DeepSpeed config
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size * args.gradient_accumulation_steps

    deepspeed_plugin.deepspeed_config["train_batch_size"] = (
        args.train_batch_size * args.gradient_accumulation_steps * int(os.environ["WORLD_SIZE"])
    )
    
    deepspeed_plugin.set_mixed_precision(args.mixed_precision)

    # Generalize the AIO configuration to apply if ANY NVMe offloading is used in Stage 3.
    is_optimizer_nvme_offload = args.zero_stage == 3 and args.offload_optimizer_device == "nvme"
    is_param_nvme_offload = args.zero_stage == 3 and args.offload_param_device == "nvme"

    if is_optimizer_nvme_offload or is_param_nvme_offload:
        deepspeed_plugin.deepspeed_config["aio"] = {}
        deepspeed_plugin.deepspeed_config["aio"]["single_submit"] = False
        deepspeed_plugin.deepspeed_config["aio"]["overlap_events"] = True
        deepspeed_plugin.deepspeed_config["aio"]["thread_count"] = 1
        deepspeed_plugin.deepspeed_config["aio"]["queue_depth"] = 128
        deepspeed_plugin.deepspeed_config["aio"]["block_size"] = 8388608
        logger.info("[DeepSpeed] NVMe parameter offloading configured.")

    #deepspeed_plugin.deepspeed_config["scheduler"] = {}
    
    deepspeed_plugin.deepspeed_config["optimizer"] = {}
    deepspeed_plugin.deepspeed_config["optimizer"]["type"] = "Adam"
    deepspeed_plugin.deepspeed_config["optimizer"]["params"] = {}
    deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"] = getattr(args, "unet_lr", args.learning_rate)
    deepspeed_plugin.deepspeed_config["optimizer"]["params"]["betas"] = [0.9, 0.999]
    deepspeed_plugin.deepspeed_config["optimizer"]["params"]["weight_decay"] = 0.01


    if args.mixed_precision.lower() == "fp16":
        deepspeed_plugin.deepspeed_config["fp16"]["initial_scale_power"] = 0

    if args.full_fp16 or args.fp16_master_weights_and_gradients:
      if args.offload_optimizer_device == "cpu" and args.zero_stage == 2:
        deepspeed_plugin.deepspeed_config["fp16"]["fp16_master_weights_and_grads"] = True
        logger.info("[DeepSpeed] full fp16 enable.")
      else:
        logger.info("[DeepSpeed]full fp16, fp16_master_weights_and_grads currently only supported using ZeRO-Offload with DeepSpeedCPUAdam on ZeRO-2 stage.")

    deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_optimizer"]["pin_memory"] = True
    deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["pin_memory"] = True


    if args.offload_optimizer_device is not None:
        logger.info("[DeepSpeed] start to manually build cpu_adam.")
        deepspeed.ops.op_builder.CPUAdamBuilder().load()
        logger.info("[DeepSpeed] building cpu_adam done.")

    return deepspeed_plugin


def finalize_deepspeed_config(deepspeed_plugin: DeepSpeedPlugin, args: argparse.Namespace, num_update_steps_per_epoch: int):

    if args.max_train_epochs is not None:
        max_train_steps = args.max_train_epochs * num_update_steps_per_epoch
    else:
        max_train_steps = args.max_train_steps  # Use the provided max_train_steps

    # Now we can calculate the true total batch size and warmup steps
    train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * int(os.environ["WORLD_SIZE"])
    warmup_num_steps = int(max_train_steps * (args.lr_warmup_steps / 100.0))

    deepspeed_plugin.deepspeed_config["train_batch_size"] = train_batch_size

    """ deepspeed_plugin.deepspeed_config["scheduler"]["type"] = "WarmupLR"
    deepspeed_plugin.deepspeed_config["scheduler"]["params"] = {
        "warmup_min_lr": getattr(args, "unet_lr", args.learning_rate) if warmup_num_steps == 0 else 0,
        "warmup_max_lr": getattr(args, "unet_lr", args.learning_rate),
        "warmup_num_steps": warmup_num_steps,
    } """
    
# Accelerate library does not support multiple models for deepspeed. So, we need to wrap multiple models into a single model.
def prepare_deepspeed_model(args: argparse.Namespace, **models):
    # remove None from models
    models = {k: v for k, v in models.items() if v is not None}

    class DeepSpeedWrapper(torch.nn.Module):
        def __init__(self, **kw_models) -> None:
            super().__init__()
            self.models = torch.nn.ModuleDict()

            for key, model in kw_models.items():
                if isinstance(model, list):
                    model = torch.nn.ModuleList(model)
                assert isinstance(
                    model, torch.nn.Module
                ), f"model must be an instance of torch.nn.Module, but got {key} is {type(model)}"
                self.models.update(torch.nn.ModuleDict({key: model}))

        def get_models(self):
            return self.models

    ds_model = DeepSpeedWrapper(**models)
    return ds_model

def prepare_deepspeed_args(args: argparse.Namespace):
    if not args.deepspeed:
        return

    # To avoid RuntimeError: DataLoader worker exited unexpectedly with exit code 1.
    args.max_data_loader_n_workers = 1