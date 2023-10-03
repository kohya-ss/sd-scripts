# training with captions
# heavily based on https://github.com/kashif/diffusers

import argparse
import gc
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm
import torch
from torchvision.models import efficientnet_v2_l, efficientnet_v2_s
from torchvision import transforms

try:
    import intel_extension_for_pytorch as ipex

    if torch.xpu.is_available():
        from library.ipex import ipex_init

        ipex_init()
except Exception:
    pass

from accelerate.utils import set_seed
from transformers import CLIPTextModel, PreTrainedTokenizerFast
from diffusers import AutoPipelineForText2Image, DDPMWuerstchenScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.wuerstchen.modeling_wuerstchen_prior import WuerstchenPrior
from huggingface_hub import hf_hub_download

import library.train_util as train_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)


class EfficientNetEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, c_latent=16, c_cond=1280, effnet="efficientnet_v2_s"):
        super().__init__()

        if effnet == "efficientnet_v2_s":
            self.backbone = efficientnet_v2_s(weights="DEFAULT").features
        else:
            self.backbone = efficientnet_v2_l(weights="DEFAULT").features
        self.mapper = torch.nn.Sequential(
            torch.nn.Conv2d(c_cond, c_latent, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))


class DatasetWrapper(train_util.DatasetGroup):
    r"""
    Wrapper for datasets to be used with DataLoader.
    add effnet_pixel_values and text_mask to dataset.
    """

    # なんかうまいことやればattributeをコピーしなくてもいい気がする

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.image_data = dataset.image_data
        self.tokenizer = tokenizer
        self.num_train_images = dataset.num_train_images
        self.datasets = dataset.datasets

        # images are already resized
        self.effnet_transforms = transforms.Compose(
            [
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # create attention mask by input_ids
        input_ids = item["input_ids"]
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        text_mask = attention_mask.bool()
        item["text_mask"] = text_mask

        # create effnet input
        images = item["images"]
        # effnet_pixel_values = [self.effnet_transforms(image) for image in images]
        # effnet_pixel_values = torch.stack(effnet_pixel_values, dim=0)
        effnet_pixel_values = self.effnet_transforms(((images) + 1.0) / 2.0)
        effnet_pixel_values = effnet_pixel_values.to(memory_format=torch.contiguous_format)
        item["effnet_pixel_values"] = effnet_pixel_values

        return item

    def __len__(self):
        return len(self.dataset)

    def add_replacement(self, str_from, str_to):
        self.dataset.add_replacement(str_from, str_to)

    def enable_XTI(self, *args, **kwargs):
        self.dataset.enable_XTI(*args, **kwargs)

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        self.dataset.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, weight_dtype, cache_to_disk=False, is_main_process=True
    ):
        self.dataset.cache_text_encoder_outputs(tokenizers, text_encoders, device, weight_dtype, cache_to_disk, is_main_process)

    def set_caching_mode(self, caching_mode):
        self.dataset.set_caching_mode(caching_mode)

    def verify_bucket_reso_steps(self, min_steps: int):
        self.dataset.verify_bucket_reso_steps(min_steps)

    def is_latent_cacheable(self) -> bool:
        return self.dataset.is_latent_cacheable()

    def is_text_encoder_output_cacheable(self) -> bool:
        return self.dataset.is_text_encoder_output_cacheable()

    def set_current_epoch(self, epoch):
        self.dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        self.dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        self.dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        self.dataset.disable_token_padding()


def get_hidden_states(args: argparse.Namespace, input_ids, text_mask, tokenizer, text_encoder, weight_dtype=None):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids, attention_mask=text_mask)[0]

    # input_ids: b,n,77
    b_size = input_ids.size()[0]
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77
    text_mask = text_mask.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77

    if args.clip_skip is None:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out["hidden_states"][-args.clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    # bs*3, 77, 768 or 1024
    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if args.max_token_length is not None:
        # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, args.max_token_length, tokenizer.model_max_length):
            states_list.append(encoder_hidden_states[:, i : i + tokenizer.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
        encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states


def train(args):
    # TODO: add checking for unsupported args
    # TODO: cache image encoder outputs instead of latents

    # train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)

    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    print("prepare tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="tokenizer")

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
        if args.dataset_config is not None:
            print(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                print(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                print("Using DreamBooth method.")
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
                print("Training with captions.")
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

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    # wrap for wuestchen
    train_dataset_group = DatasetWrapper(train_dataset_group, tokenizer)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        print(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    # acceleratorを準備する
    print("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, _ = train_util.prepare_dtype(args)

    # Load scheduler, effnet, tokenizer, clip_model
    print("prepare scheduler, effnet, clip_model")
    noise_scheduler = DDPMWuerstchenScheduler()

    # TODO support explicit local caching for faster loading
    pretrained_checkpoint_file = hf_hub_download("dome272/wuerstchen", filename="model_v2_stage_b.pt")
    state_dict = torch.load(pretrained_checkpoint_file, map_location="cpu")
    image_encoder = EfficientNetEncoder()
    image_encoder.load_state_dict(state_dict["effnet_state_dict"])
    image_encoder.eval()

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    )

    # Freeze text_encoder and image_encoder
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # load prior model
    prior: WuerstchenPrior = WuerstchenPrior.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="prior")

    # EMA is not supported yet

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        accelerator.print("Use xformers by Diffusers")
        set_diffusers_xformers_flag(prior, True)

    # 学習を準備する

    # 学習を準備する：モデルを適切な状態にする
    training_models = []
    if args.gradient_checkpointing:
        # prior.enable_gradient_checkpointing()
        print("*" * 80)
        print("*** Prior model does not support gradient checkpointing. ***")
        print("*" * 80)
    training_models.append(prior)

    text_encoder.requires_grad_(False)
    text_encoder.eval()

    for m in training_models:
        m.requires_grad_(True)

    params = []
    for m in training_models:
        params.extend(m.parameters())
    params_to_optimize = params

    # calculate number of trainable parameters
    n_params = 0
    for p in params:
        n_params += p.numel()

    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # dataloaderを準備する

    # DataLoaderのプロセス数：0はメインプロセスになる
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        prior.to(weight_dtype)
        text_encoder.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        prior.to(weight_dtype)
        text_encoder.to(weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい
    prior, image_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        prior, image_encoder, optimizer, train_dataloader, lr_scheduler
    )
    (prior, image_encoder) = train_util.transform_models_if_DDP([prior, image_encoder])

    text_encoder.to(weight_dtype)
    text_encoder.to(accelerator.device)
    image_encoder.to(weight_dtype)
    image_encoder.to(accelerator.device)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "wuerstchen_finetuning" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs
        )

    # workaround for DDPMWuerstchenScheduler
    def add_noise(
        scheduler: DDPMWuerstchenScheduler,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod_timesteps = scheduler._alpha_cumprod(timesteps, original_samples.device)
        sqrt_alpha_prod = alphas_cumprod_timesteps**0.5

        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod_timesteps) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        loss_total = 0
        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(training_models[0]):  # 複数モデルに対応していない模様だがとりあえずこうしておく
                input_ids = batch["input_ids"]
                text_mask = batch["text_mask"]
                effnet_pixel_values = batch["effnet_pixel_values"]

                with torch.no_grad():
                    input_ids = input_ids.to(accelerator.device)
                    text_mask = text_mask.to(accelerator.device)
                    prompt_embeds = get_hidden_states(
                        args, input_ids, text_mask, tokenizer, text_encoder, None if not args.full_fp16 else weight_dtype
                    )

                    image_embeds = image_encoder(effnet_pixel_values)
                    image_embeds = image_embeds.add(1.0).div(42.0)  # scale

                    # Sample noise that we'll add to the image_embeds
                    noise = torch.randn_like(image_embeds)
                    bsz = image_embeds.shape[0]

                    # Sample a random timestep for each image
                    # TODO support mul/add/clump
                    timesteps = torch.rand((bsz,), device=image_embeds.device, dtype=weight_dtype)

                    # add noise to latent: This is same to Diffuzz.diffuse in diffuzz.py
                    # noisy_latents = noise_scheduler.add_noise(image_embeds, noise, timesteps)
                    noisy_latents = add_noise(noise_scheduler, image_embeds, noise, timesteps)

                # Predict the noise residual
                with accelerator.autocast():
                    noise_pred = prior(noisy_latents, timesteps, prompt_embeds)

                target = noise

                # TODO add consistency loss

                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = []
                    for m in training_models:
                        params_to_clip.extend(m.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # TODO ここでサンプルを生成する
                # sample_images(
                #     accelerator,
                #     args,
                #     None,
                #     global_step,
                #     accelerator.device,
                #     vae,
                #     [tokenizer1, tokenizer2],
                #     [text_encoder, text_encoder2],
                #     prior,
                # )

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        # TODO simplify to save prior only
                        pipeline = AutoPipelineForText2Image.from_pretrained(
                            args.pretrained_decoder_model_name_or_path,
                            prior_prior=accelerator.unwrap_model(prior),
                            prior_text_encoder=accelerator.unwrap_model(text_encoder),
                            prior_tokenizer=tokenizer,
                        )
                        ckpt_name = train_util.get_step_ckpt_name(args, "", global_step)
                        pipeline.prior_pipe.save_pretrained(os.path.join(args.output_dir, ckpt_name))

                        # TODO remove older saved models

            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if args.logging_dir is not None:
                logs = {"loss": current_loss}
                logs["lr"] = float(lr_scheduler.get_last_lr()[0])
                if (
                    args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower()
                ):  # tracking d*lr value
                    logs["lr/d*lr"] = (
                        lr_scheduler.optimizers[0].param_groups[0]["d"] * lr_scheduler.optimizers[0].param_groups[0]["lr"]
                    )
                accelerator.log(logs, step=global_step)

            # TODO moving averageにする
            loss_total += current_loss
            avr_loss = loss_total / (step + 1)
            logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_total / len(train_dataloader)}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                epoch_no = epoch + 1
                saving = epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
                if saving:
                    pipeline = AutoPipelineForText2Image.from_pretrained(
                        args.pretrained_decoder_model_name_or_path,
                        prior_prior=accelerator.unwrap_model(prior),
                        prior_text_encoder=accelerator.unwrap_model(text_encoder),
                        prior_tokenizer=tokenizer,
                    )
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "", epoch)
                    pipeline.prior_pipe.save_pretrained(os.path.join(args.output_dir, ckpt_name))

                    # TODO remove older saved models

        # TODO ここでサンプルを生成する

    is_main_process = accelerator.is_main_process

    accelerator.end_training()

    if args.save_state:  # and is_main_process:
        train_util.save_state_on_train_end(args, accelerator)

    # del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            args.pretrained_decoder_model_name_or_path,
            prior_prior=accelerator.unwrap_model(prior),
            prior_text_encoder=accelerator.unwrap_model(text_encoder),
            prior_tokenizer=tokenizer,
        )
        ckpt_name = train_util.get_last_ckpt_name(args, "")
        pipeline.prior_pipe.save_pretrained(os.path.join(args.output_dir, ckpt_name))
        print("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # train_util.add_sd_models_arguments(parser)
    parser.add_argument(
        "--pretrained_prior_model_name_or_path",
        type=str,
        default="warp-ai/wuerstchen-prior",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_decoder_model_name_or_path",
        type=str,
        default="warp-ai/wuerstchen",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    # train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)

    # TODO add assertion for SD related arguments

    parser.add_argument("--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    train(args)
