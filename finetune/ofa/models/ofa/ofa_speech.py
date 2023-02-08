import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
import torch.distributed as dist

from fairseq.data.data_utils import compute_mask_indices
from fairseq.models.wav2vec.wav2vec2 import (
    Wav2Vec2Config,
    TransformerEncoder as SpeechTransformerEncoder,
    make_conv_pos
)
from fairseq.modules import (
    SamePad,
    TransposeLast,
)
from fairseq.utils import index_put
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel
from fairseq.modules import LayerNorm
from torch import Tensor
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.tasks import FairseqTask
from .unify_transformer import Embedding, TransformerDecoder

@dataclass
class SpeechTextUnifyEncoderConfig(Wav2Vec2Config):
    audio_mask_length: int = 10
    audio_mask_prob: float = 0.65
    no_emb_update_unsup: bool = True


@dataclass
class SpeechTextUnifyConfig(SpeechTextUnifyEncoderConfig):
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings " "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights " "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )

    max_phone_positions: int = field(
        default=2048
    )
    decoder_prompt: bool = field(
        default=False,
        metadata={"help": "use prompt tuning in the decoder"})
    decoder_prompt_type: str = field(
        default= "prefix",
        metadata={"help": "the type of prompt tuning"})
    decoder_prompt_length: int = field(
        default=10,
        metadata={"help": "use prompt tuning in the decoder"})
    decoder_prompt_projection: bool = field(
        default=False,
        metadata={"help": "use prompt projection"})
    decoder_prompt_dim: int = field(
        default=10,
        metadata={"help": "decoder prompt dimension if use decoder prompt projection"})

    quant_noise_pq: int = 0
    token_bucket_size: int = 256
    image_bucket_size: int = 42
    attn_scale_factor: int = 2

    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_fn: str = "gelu"
    dropout: float = 0.1
    adaptive_softmax_dropout: float = 0
    share_all_embeddings: bool = False
    adaptive_input: bool = False
    no_cross_attention: bool = False
    cross_self_attention: bool = False
    decoder_output_dim: int = 768
    decoder_input_dim: int = 768
    no_scale_embedding: bool = False
    layernorm_embedding: bool = False
    tie_adaptive_weights: bool = False
    checkpoint_activations: bool = False
    offload_activations: bool = False
    encoder_layerdrop: float = 0
    quant_noise_pq_block_size: int = 8
    quant_noise_scalar: float = 0
    relu_dropout: float = 0.0

    max_source_positions: int = 1024
    pooler_activation_fn: str = "tanh"
    pooler_dropout: float = 0.0
    pooler_classifier: str = "mlp"
    resnet_drop_path_rate: float = 0.0
    encoder_drop_path_rate: float = 0.0
    decoder_drop_path_rate: float = 0.0
    resnet_type: str = "resnet152"
    freeze_encoder_embedding: bool = False
    freeze_decoder_embedding: bool = False
    add_type_embedding: bool = True
    code_image_size: int = 128
    patch_layernorm_embedding: bool = True
    code_layernorm_embedding: bool = True
    entangle_position_embedding: bool = True
    disable_entangle: bool = False
    sync_bn: bool = False
    scale_attn: bool = False
    scale_fc: bool = False
    scale_heads: bool = False

    audio_mask_length: int = 10
    audio_mask_prob: float = 0.65
    no_emb_update_unsup: bool = True

    phone_dict_size: Optional[int] = field(
        default=None,
        metadata={"help": "decoder prompt dimension if use decoder prompt projection"})
    bitfit: bool = False


@register_model("ofa_speech", dataclass=SpeechTextUnifyConfig)
class OFASpeech(FairseqEncoderDecoderModel):
    def __init__(self, cfg: SpeechTextUnifyConfig, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder

    
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: SpeechTextUnifyConfig, task: FairseqTask):
        """Build a new model instance."""
        src_dict, tgt_dict, phone_dict = task.source_dictionary, task.target_dictionary, task.phone_dictionary
        # phone_dict = None
        # src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        encoder_embed_tokens = build_embedding(tgt_dict, cfg.encoder_embed_dim)
        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens, phone_dict)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens, phone_dict=None):
        return SpeechTextUnifyEncoder(cfg, src_dict, embed_tokens, phone_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(self,
                src_tokens,
                src_lengths,
                prev_output_tokens,
                patch_images: Optional[torch.Tensor] = None,
                patch_images_2: Optional[torch.Tensor] = None,
                patch_masks: Optional[torch.Tensor] = None,
                code_masks: Optional[torch.Tensor] = None,
                sample_patch_num: Optional[int] = None,
                fbank: Optional[torch.Tensor] = None,
                fbank_length: Optional[torch.Tensor] = None,
                fbank_masks: Optional[torch.Tensor] = None,
                audio_code_masks: Optional[torch.Tensor] = None,
                phone_items: Optional[torch.Tensor] = None,
                phone_lengths: Optional[torch.Tensor] = None,
                phone_masks: Optional[torch.Tensor] = None,
                encoder_features_only: Optional[torch.Tensor] = True,
                mask: Optional[torch.Tensor] = False,
                mask_prob: Optional[torch.Tensor] = None,
                layer=None,
                features_only: bool = False,
                classification_head_name: Optional[str] = None,
                token_embeddings: Optional[torch.Tensor] = None,
                return_all_hiddens: bool = False,
                alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None,
                ):
        if classification_head_name is not None:
            features_only = True
        encoder_out = self.encoder(
            # id,
            src_tokens,
            src_lengths=src_lengths,
            patch_images=patch_images,
            patch_masks=patch_masks,
            patch_images_2=patch_images_2,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
            sample_patch_num=sample_patch_num,
            fbank=fbank,
            fbank_length=fbank_length,
            fbank_masks=fbank_masks,
            audio_code_masks=audio_code_masks,
            phone_items=phone_items,
            phone_lengths=phone_lengths,
            phone_masks=phone_masks,
            encoder_features_only=encoder_features_only,
            mask=mask,
            mask_prob=mask_prob)
        x, extra = self.decoder(
            prev_output_tokens,
            code_masks=code_masks,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        if "phone_distribution" in encoder_out:
            extra["phone_distribution"] = encoder_out["phone_distribution"][0]
        if "kl_loss" in encoder_out:
            extra["kl_loss"] = encoder_out["kl_loss"]
        if "encoder_padding_mask" in encoder_out:
            extra["encoder_padding_mask"] = encoder_out["encoder_padding_mask"][0]

        pad = self.encoder.padding_idx
        if classification_head_name is not None:
            prev_lengths = prev_output_tokens.ne(pad).sum(1)
            gather_index = prev_lengths[:, None, None].expand(x.size(0), 1, x.size(2)) - 1
            sentence_representation = x.gather(1, gather_index).squeeze()
            if self.classification_heads[classification_head_name].use_two_images:
                hidden_size = sentence_representation.size(1)
                sentence_representation = sentence_representation.view(-1, hidden_size * 2)
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break
        return x, extra

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_encoder_normalized_probs(self, net_output, log_probs, **kwargs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["phone_distribution"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["phone_distribution"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        decoder_input_loaded_dict_size = state_dict["decoder.embed_tokens.weight"].size(0)
        decoder_output_loaded_dict_size = state_dict["decoder.output_projection.weight"].size(0)

        if decoder_input_loaded_dict_size < len(self.decoder.dictionary):
            num_langids_to_add = len(self.decoder.dictionary) - decoder_input_loaded_dict_size
            embed_dim = state_dict["decoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["decoder.embed_tokens.weight"].dtype,
            )
            new_lang_embed_to_add = new_lang_embed_to_add.to(state_dict["decoder.embed_tokens.weight"])
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [state_dict["decoder.embed_tokens.weight"], new_lang_embed_to_add]
            )

        if decoder_output_loaded_dict_size < len(self.decoder.dictionary):
            num_langids_to_add = len(self.decoder.dictionary) - decoder_output_loaded_dict_size
            embed_dim = state_dict["decoder.output_projection.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["decoder.output_projection.weight"].dtype,
            )
            new_lang_embed_to_add = new_lang_embed_to_add.to(state_dict["decoder.output_projection.weight"])
            state_dict["decoder.output_projection.weight"] = torch.cat(
                [state_dict["decoder.output_projection.weight"], new_lang_embed_to_add]
            )

class SpeechTextUnifyEncoder(FairseqEncoder):
    def __init__(self, cfg: SpeechTextUnifyConfig, dictionary, embed_tokens, phone_dictionary):
        super().__init__(dictionary)
        self.cfg = cfg
        self.embed = cfg.encoder_embed_dim
        
        self.padding_idx = embed_tokens.padding_idx
         
        # fbank encoder
        self.subsample = Conv2dSubsampling4(
            80 * 1,
            cfg.encoder_embed_dim
        )
        self.post_subsample_proj = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim)

        # phone and text encoder
        # self.phone_padding_idx = embed_tokens.padding_idx
        # self.phone_item_embedding = Embedding(141, cfg.encoder_embed_dim, self.phone_padding_idx)
        self.phone_padding_idx = phone_dictionary.pad()
        self.phone_item_embedding = Embedding(len(phone_dictionary), cfg.encoder_embed_dim, self.phone_padding_idx)
        self.phone_dictionary = phone_dictionary

        # mask
        self.mask_prob = cfg.audio_mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.audio_mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)

        self.final_proj = nn.Linear(self.embed, self.embed)

        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg: SpeechTextUnifyEncoderConfig, task=None, embed_tokens=None):
        """Build a new model instance."""
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        return cls(cfg, src_dict, embed_tokens)

    def apply_mask(
            self,
            x,
            padding_mask,
            mask_indices=None,
            mask_channel_indices=None,
            mask_prob=None
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0 or mask_prob is not None:
            if mask_indices is None:
                if mask_prob is None:
                    mask_prob = self.mask_prob
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                        .to(x.device)
                        .unsqueeze(1)
                        .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
            self,
            src_tokens,
            src_lengths,
            patch_images: Optional[torch.Tensor] = None,
            patch_images_2: Optional[torch.Tensor] = None,
            patch_masks: Optional[torch.Tensor] = None,
            code_masks: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            sample_patch_num: Optional[int] = None,
            fbank: Optional[torch.Tensor] = None,
            fbank_length: Optional[torch.Tensor] = None,
            fbank_masks: Optional[torch.Tensor] = None,
            audio_code_masks: Optional[torch.Tensor] = None,
            phone_items: Optional[torch.Tensor] = None,
            phone_lengths: Optional[torch.Tensor] = None,
            phone_masks: Optional[torch.Tensor] = None,
            encoder_features_only: Optional[torch.Tensor] = True,
            mask: Optional[torch.Tensor] = False,
            mask_prob: Optional[torch.Tensor] = None,
            layer=None,
    ):

        features, fbank_feature_length = self.subsample(fbank, fbank_length)

        if self.post_subsample_proj is not None:
            features = self.post_subsample_proj(features)

        padding_mask = (
            torch.BoolTensor(features.shape[:2]).fill_(False)
            # if self.pad_audio else None
        ).to(features.device)
        for i, l in enumerate(fbank_feature_length):
            diff = l - padding_mask.shape[-1]
            if diff < 0:
                padding_mask[i, diff:] = True

        pre_encoder_features = features.clone()
        features = self.dropout_input(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_prob=mask_prob
            )
        else:
            x = features
            mask_indices = None

        padding_mask[~fbank_masks] = True

        phone_x = None
        phone_padding_mask = None
        if phone_items is not None:
            # if phone_items.less(self.cfg.phone_dict_size + 1).all() == False:
            #     torch.set_printoptions(profile="full")
            #     print(phone_items)
            phone_x = self.phone_item_embedding(phone_items)
            phone_padding_mask = phone_items.eq(self.phone_padding_idx)
            phone_padding_mask[~phone_masks] = True
            if mask_indices is not None:
                phone_mask_indices = phone_padding_mask.new_zeros(phone_padding_mask.size()).bool()
                mask_indices = torch.cat([mask_indices, phone_mask_indices], dim=1)

        pre_padding_mask = padding_mask.clone()
        x, layer_results, pos_embed, padding_mask = self.encoder(
            x,
            padding_mask=padding_mask,
            phone_x=phone_x,
            phone_padding_mask=phone_padding_mask,
            layer=layer,
            context_layer=6
        )


        if self.cfg.phone_dict_size is not None:
            emb_weight = self.phone_item_embedding.weight[3:self.cfg.phone_dict_size, :]
        else:
            emb_weight = self.phone_item_embedding.weight[3:self.phone_dictionary.index("<mask>"), :]
        # emb_weight = self.phone_item_embedding.weight[3:124, :]
        if encoder_features_only == False:  # no gradient for embedding here
            emb_weight = emb_weight.detach()

        phone_distribution = F.linear(x, emb_weight, None)

        if encoder_features_only:
            return {
                "x": x,
                "phone_distribution": [phone_distribution.transpose(0, 1)],
                "padding_mask": padding_mask,
                "encoder_out": [x.transpose(0, 1)],  # T x B x C
                "encoder_padding_mask": [padding_mask],  # B x T
                "position_embeddings": [pos_embed],
                "kl_loss": None
            }

        result = {
            "losses": {},
        }

        with torch.no_grad():
            self.encoder.eval()
            y, y_layer_results, _, _ = self.encoder.extract_features(
                pre_encoder_features,
                padding_mask=pre_padding_mask,
                phone_x=phone_x,
                phone_padding_mask=phone_padding_mask,
                min_layer=0,# self.cfg.encoder_layers - self.average_top_k_layers,
                context_layer=6
            )
            y = {
                "x": y,
                "padding_mask": padding_mask,
                "layer_results": y_layer_results,
            }

            if self.cfg.phone_dict_size is not None:
                emb_weight = self.phone_item_embedding.weight[3:self.cfg.phone_dict_size, :]
            else:
                emb_weight = self.phone_item_embedding.weight[3:self.phone_dictionary.index("<mask>"), :]

            y = F.linear(y['x'], emb_weight, None)
            y = y[mask_indices]
            self.encoder.train()

        y_student = phone_distribution[mask_indices]

        def _kl_loss(p, q):
            loss = F.kl_div(utils.log_softmax(p, dim=-1), utils.softmax(q, dim=-1), reduction='sum')
            return loss

        y = y
        kl_loss = _kl_loss(y_student.float(), y.float())

        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["pred_var"] = self.compute_var(y_student.float())

        if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
            logger.error(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
        if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )

        # if self.ema is not None:
        #     result["ema_decay"] = self.ema.get_decay() * 1000
        # return result

        return {
            "phone_distribution": [phone_distribution.transpose(1, 0)],
            "encoder_out": [x.transpose(1, 0)],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
            "position_embeddings": [pos_embed],  # B x T x C
            "kl_loss": kl_loss
        }

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]

        if len(encoder_out["position_embeddings"]) == 0:
            new_position_embeddings = []
        else:
            new_position_embeddings = [(encoder_out["position_embeddings"][0]).index_select(0, new_order)]

        kl_loss = encoder_out["kl_loss"]
        if len(encoder_out["phone_distribution"]) == 0:
            new_phone_distribution = []
        else:
            new_phone_distribution = [(encoder_out["phone_distribution"][0]).index_select(1, new_order)]

        return {
            "phone_distribution": new_phone_distribution,
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,
            "position_embeddings": new_position_embeddings,
            "kl_loss": kl_loss
        }

class TransformerEncoder(SpeechTransformerEncoder):

    def __init__(self, args: Wav2Vec2Config):
        super().__init__(args)

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding=k // 2,
                                groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(l)
                    ]
                )

            self.phone_pos_conv = make_conv_block(
                self.embedding_dim, k, args.conv_pos_groups, num_layers
            )

        else:
            self.phone_pos_conv = make_conv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups,
            )

        self.phone_layer_norm = LayerNorm(self.embedding_dim)

    def forward(self, x, padding_mask=None, phone_x=None, phone_padding_mask=None, layer=None, context_layer=None):
        x, layer_results, x_conv, pre_padding_mask = self.extract_features(x, padding_mask, phone_x, phone_padding_mask,
                                                                           layer, context_layer=context_layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results, x_conv, pre_padding_mask

    def extract_features(
            self,
            x,
            padding_mask=None,
            phone_x=None,
            phone_padding_mask=None,
            tgt_layer=None,
            min_layer=0,
            context_layer=None,
    ):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        if phone_x is not None:
            if phone_padding_mask is not None:
                phone_x = index_put(phone_x, phone_padding_mask, 0)

            phone_x_conv = self.phone_pos_conv(phone_x.transpose(1, 2))
            phone_x_conv = phone_x_conv.transpose(1, 2)
            phone_x = phone_x + phone_x_conv

            if not self.layer_norm_first:
                phone_x = self.layer_norm(phone_x)

        pre_padding_mask = padding_mask.clone()

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):

            if i < context_layer and (~padding_mask).any() == False:
                continue

            if i == context_layer and phone_x is not None and phone_x_conv is not None:
                x = x.transpose(0, 1)
                x = torch.cat([x, phone_x], dim=1)
                padding_mask = torch.cat([padding_mask, phone_padding_mask], dim=1)
                pre_padding_mask = padding_mask.clone()
                x_conv = torch.cat([x_conv, phone_x_conv], dim=1)
                x = x.transpose(0, 1)

            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results, x_conv, pre_padding_mask

class Conv2dSubsampling4(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(2):
            out = ((out.float() - 1) // 2 + 1).floor().long()
        return out

    def forward(
            self,
            x: torch.Tensor,
            x_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        return x, self.get_out_seq_lens_tensor(x_length)

@register_model_architecture("ofa_speech", "ofa_speech_base")
def ofa_speech_base_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", "layer_norm")
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.dropout_input = getattr(args, "dropout_input", 0.0)
    args.dropout_features = getattr(args, "dropout_features", 0.0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 1.0)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)

    args.audio_mask_prob = getattr(args, "audio_mask_prob", 0.7)
    args.audio_mask_length = getattr(args, "audio_mask_length", 10)

    args.pos_conv_depth = getattr(args, "pos_conv_depth", 5)
    args.conv_pos = getattr(args, "conv_pos", 95)

    args.require_same_masks = getattr(args, "require_same_masks", True)
    args.mask_dropout = getattr(args, "mask_dropout", 0)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", 3072
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)

    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_classifier = getattr(args, "pooler_classifier", "mlp")
    args.resnet_drop_path_rate = getattr(args, "resnet_drop_path_rate", 0.0)
    args.encoder_drop_path_rate = getattr(args, "encoder_drop_path_rate", 0.0)
    args.decoder_drop_path_rate = getattr(args, "decoder_drop_path_rate", 0.0)
    args.resnet_type = getattr(args, "resnet_type", "resnet152")
    args.token_bucket_size = getattr(args, "token_bucket_size", 256)
    args.image_bucket_size = getattr(args, "image_bucket_size", 42)
    args.freeze_encoder_embedding = getattr(args, "freeze_encoder_embedding", False)
    args.freeze_decoder_embedding = getattr(args, "freeze_decoder_embedding", False)
    args.add_type_embedding = getattr(args, "add_type_embedding", True)
    args.attn_scale_factor = getattr(args, "attn_scale_factor", 2)
    args.code_image_size = getattr(args, "code_image_size", 128)
    args.patch_layernorm_embedding = getattr(args, "patch_layernorm_embedding", True)
    args.code_layernorm_embedding = getattr(args, "code_layernorm_embedding", True)
    args.entangle_position_embedding = getattr(args, "entangle_position_embedding", True)
    args.disable_entangle = getattr(args, "disable_entangle", False)
    args.sync_bn = getattr(args, "sync_bn", False)
    args.scale_attn = getattr(args, "scale_attn", False)
    args.scale_fc = getattr(args, "scale_fc", False)
    args.scale_heads = getattr(args, "scale_heads", False)
    args.scale_resids = getattr(args, "scale_resids", False)


@register_model_architecture("ofa_speech", "ofa_speech_large")
def ofa_speech_large_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", "layer_norm")
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.dropout_input = getattr(args, "dropout_input", 0.0)
    args.dropout_features = getattr(args, "dropout_features", 0.0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 1.0)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_ffn_embed_dim = getattr(
        args, "encoder_ffn_embed_dim", 4096
    )
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    args.audio_mask_prob = getattr(args, "audio_mask_prob", 0.7)
    args.audio_mask_length = getattr(args, "audio_mask_length", 10)

    args.instance_norm_target_layer = getattr(args, "instance_norm_target_layer", True)
    args.average_top_k_layers = getattr(args, "average_top_k_layers", 8)

    args.pos_conv_depth = getattr(args, "pos_conv_depth", 5)
    args.conv_pos = getattr(args, "conv_pos", 95)

    args.require_same_masks = getattr(args, "require_same_masks", True)
    args.mask_dropout = getattr(args, "mask_dropout", 0)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", 4096
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)

    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_classifier = getattr(args, "pooler_classifier", "mlp")
    args.resnet_drop_path_rate = getattr(args, "resnet_drop_path_rate", 0.0)
    args.encoder_drop_path_rate = getattr(args, "encoder_drop_path_rate", 0.0)
    args.decoder_drop_path_rate = getattr(args, "decoder_drop_path_rate", 0.0)
    args.resnet_type = getattr(args, "resnet_type", "resnet152")
    args.token_bucket_size = getattr(args, "token_bucket_size", 256)
    args.image_bucket_size = getattr(args, "image_bucket_size", 42)
    args.freeze_encoder_embedding = getattr(args, "freeze_encoder_embedding", False)
    args.freeze_decoder_embedding = getattr(args, "freeze_decoder_embedding", False)
    args.add_type_embedding = getattr(args, "add_type_embedding", True)
    args.attn_scale_factor = getattr(args, "attn_scale_factor", 2)
    args.code_image_size = getattr(args, "code_image_size", 128)
    args.patch_layernorm_embedding = getattr(args, "patch_layernorm_embedding", True)
    args.code_layernorm_embedding = getattr(args, "code_layernorm_embedding", True)
    args.entangle_position_embedding = getattr(args, "entangle_position_embedding", True)
    args.disable_entangle = getattr(args, "disable_entangle", False)
    args.sync_bn = getattr(args, "sync_bn", False)
    args.scale_attn = getattr(args, "scale_attn", False)
    args.scale_fc = getattr(args, "scale_fc", False)
    args.scale_heads = getattr(args, "scale_heads", False)
    args.scale_resids = getattr(args, "scale_resids", False)
