# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BeamableMM,
    FairseqDropout,
    GradMultiply,
    LearnedPositionalEmbedding,
    LinearizedConvolution,
)


@register_model("fconv")
class FConvModel(FairseqEncoderDecoderModel):
    """
    A fully convolutional model, i.e. a convolutional encoder and a
    convolutional decoder, as described in `"Convolutional Sequence to Sequence
    Learning" (Gehring et al., 2017) <https://arxiv.org/abs/1705.03122>`_.

    Args:
        encoder (FConvEncoder): the encoder
        decoder (FConvDecoder): the decoder

    The Convolutional model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.fconv_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        def moses_subword(path):
            return {
                "path": path,
                "tokenizer": "moses",
                "bpe": "subword_nmt",
            }

        return {
            "conv.wmt14.en-fr": moses_subword(
                "https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2"
            ),
            "conv.wmt14.en-de": moses_subword(
                "https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2"
            ),
            "conv.wmt17.en-de": moses_subword(
                "https://dl.fbaipublicfiles.com/fairseq/models/wmt17.v2.en-de.fconv-py.tar.bz2"
            ),
        }

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder.num_attention_layers = sum(
            layer is not None for layer in decoder.attention
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR',
                            help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR',
                            help='decoder attention [True, ...]')
        parser.add_argument('--share-input-output-embed', action='store_true',
                            help='share input and output embeddings (requires'
                                 ' --decoder-out-embed-dim and --decoder-embed-dim'
                                 ' to be equal)')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        encoder_embed_dict = None
        if args.encoder_embed_path:
            encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
            utils.print_embed_overlap(encoder_embed_dict, task.source_dictionary)

        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)

        encoder = FConvEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            embed_dict=encoder_embed_dict,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )
        decoder = FConvDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            embed_dict=decoder_embed_dict,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed,
        )
        return FConvModel(encoder, decoder)


class FConvEncoder(FairseqEncoder):
    """
    Convolutional encoder consisting of `len(convolutions)` layers.

    Args:
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_dim (int, optional): embedding dimension
        embed_dict (str, optional): filename from which to load pre-trained
            embeddings
        max_positions (int, optional): maximum supported input sequence length
        convolutions (list, optional): the convolutional layer structure. Each
            list item `i` corresponds to convolutional layer `i`. Layers are
            given as ``(out_channels, kernel_width, [residual])``. Residual
            connections are added between layers when ``residual=1`` (which is
            the default behavior).
        dropout (float, optional): dropout to be applied before each conv layer
    """

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        embed_dict=None,
        max_positions=1024,
        convolutions=((512, 3),) * 20,
        dropout=0.1,
    ):
        super().__init__(dictionary)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.num_attention_layers = None

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(
                embed_dict, self.dictionary, self.embed_tokens
            )

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            self.padding_idx,
        )

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for _, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(
                Linear(residual_dim, out_channels)
                if residual_dim != out_channels
                else None
            )
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(
                ConvTBC(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    dropout=dropout,
                    padding=padding,
                )
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`

        Returns:
            dict:
                - **encoder_out** (tuple): a tuple with two elements, where the
                  first element is the last encoder layer's output and the
                  second element is the same quantity summed with the input
                  embedding (used for attention). The shape of both tensors is
                  `(batch, src_len, embed_dim)`.
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # used to mask padding in input
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()  # -> T x B
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        residuals = [x]
        # temporal convolutions
        for proj, conv, res_layer in zip(
            self.projections, self.convolutions, self.residuals
        ):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

            x = self.dropout_module(x)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=2)

            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()  # -> B x T
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

        # scale gradients (this only affects backward, not forward)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return {
            "encoder_out": (x, y),
            "encoder_padding_mask": encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = (
                encoder_out["encoder_out"][0].index_select(0, new_order),
                encoder_out["encoder_out"][1].index_select(0, new_order),
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])

        # don't attend over padding
        if encoder_padding_mask is not None:
            x = (
                x.float()
                .masked_fill(encoder_padding_mask.unsqueeze(1), float("-inf"))
                .type_as(x)
            )  # FP16 support: cast to float and back

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])

        # scale attention output (respecting potentially different lengths)
        s = encoder_out[1].size(1)
        if encoder_padding_mask is None:
            x = x * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(x).sum(
                dim=1, keepdim=True
            )  # exclude padding
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module("bmm", BeamableMM(beamable_mm_beam_size))


class FConvDecoder(FairseqIncrementalDecoder):
    """Convolutional decoder"""

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        embed_dict=None,
        out_embed_dim=256,
        max_positions=1024,
        convolutions=((512, 3),) * 20,
        attention=True,
        dropout=0.1,
        share_embed=False,
        positional_embeddings=True,
        adaptive_softmax_cutoff=None,
        adaptive_softmax_dropout=0.0,
    ):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([2]))
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.need_attn = True

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError(
                "Attention is expected to be a list of booleans of "
                "length equal to the number of layers."
            )

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(
                embed_dict, self.dictionary, self.embed_tokens
            )

        self.embed_positions = (
            PositionalEmbedding(
                max_positions,
                embed_dim,
                padding_idx,
            )
            if positional_embeddings
            else None
        )

        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(
                Linear(residual_dim, out_channels)
                if residual_dim != out_channels
                else None
            )
            self.convolutions.append(
                LinearizedConv1d(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    padding=(kernel_size - 1),
                    dropout=dropout,
                )
            )
            self.attention.append(
                AttentionLayer(out_channels, embed_dim) if attention[i] else None
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.adaptive_softmax = None
        self.fc2 = self.fc3 = None

        if adaptive_softmax_cutoff is not None:
            assert not share_embed
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings,
                in_channels,
                adaptive_softmax_cutoff,
                dropout=adaptive_softmax_dropout,
            )
        else:
            self.fc2 = Linear(in_channels, out_embed_dim)
            if share_embed:
                assert out_embed_dim == embed_dim, (
                    "Shared embed weights implies same dimensions "
                    " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
                )
                self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        if encoder_out is not None:
            encoder_padding_mask = encoder_out["encoder_padding_mask"]
            encoder_out = encoder_out["encoder_out"]

            # split and transpose encoder outputs
            encoder_a, encoder_b = self._split_encoder_out(
                encoder_out, incremental_state
            )

        if self.embed_positions is not None:
            pos_embed = self.embed_positions(prev_output_tokens, incremental_state)
        else:
            pos_embed = 0

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self._embed_tokens(prev_output_tokens, incremental_state)

        # embed tokens and combine with positional embeddings
        x += pos_embed
        x = self.dropout_module(x)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        for proj, conv, attention, res_layer in zip(
            self.projections, self.convolutions, self.attention, self.residuals
        ):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = self.dropout_module(x)
            x = conv(x, incremental_state)
            x = F.glu(x, dim=2)

            # attention
            if attention is not None:
                x = self._transpose_if_training(x, incremental_state)

                x, attn_scores = attention(
                    x, target_embedding, (encoder_a, encoder_b), encoder_padding_mask
                )

                if not self.training and self.need_attn:
                    attn_scores = attn_scores / num_attn_layers
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)

                x = self._transpose_if_training(x, incremental_state)

            # residual
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = self._transpose_if_training(x, incremental_state)

        # project back to size of vocabulary if not using adaptive softmax
        if self.fc2 is not None and self.fc3 is not None:
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.fc3(x)

        return x, avg_attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        encoder_out = utils.get_incremental_state(
            self, incremental_state, "encoder_out"
        )
        if encoder_out is not None:
            encoder_out = tuple(eo.index_select(0, new_order) for eo in encoder_out)
            utils.set_incremental_state(
                self, incremental_state, "encoder_out", encoder_out
            )

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return (
            self.embed_positions.max_positions
            if self.embed_positions is not None
            else float("inf")
        )

    def upgrade_state_dict(self, state_dict):
        if utils.item(state_dict.get("decoder.version", torch.Tensor([1]))[0]) < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.convolutions):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict["decoder.version"] = torch.Tensor([1])
        return state_dict

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def _split_encoder_out(self, encoder_out, incremental_state):
        """Split and transpose encoder outputs.

        This is cached when doing incremental inference.
        """
        cached_result = utils.get_incremental_state(
            self, incremental_state, "encoder_out"
        )
        if cached_result is not None:
            return cached_result

        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)

        if incremental_state is not None:
            utils.set_incremental_state(self, incremental_state, "encoder_out", result)
        return result

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x


def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception(
                "invalid number of parameters in convolution spec "
                + str(spec)
                + ". expected 2 or 3"
            )
    return tuple(extended)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, dropout=0.0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from fairseq.modules import ConvTBC

    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


@register_model_architecture("fconv", "fconv")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_layers = getattr(args, "encoder_layers", "[(512, 3)] * 20")
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_layers = getattr(args, "decoder_layers", "[(512, 3)] * 20")
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 256)
    args.decoder_attention = getattr(args, "decoder_attention", "True")
    args.share_input_output_embed = getattr(args, "share_input_output_embed", False)


@register_model_architecture("fconv", "fconv_iwslt_de_en")
def fconv_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", "[(256, 3)] * 4")
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_layers = getattr(args, "decoder_layers", "[(256, 3)] * 3")
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 256)
    base_architecture(args)


@register_model_architecture("fconv", "fconv_wmt_en_ro")
def fconv_wmt_en_ro(args):
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    base_architecture(args)


@register_model_architecture("fconv", "fconv_wmt_en_de")
def fconv_wmt_en_de(args):
    convs = "[(512, 3)] * 9"  # first 9 layers have 512 units
    convs += " + [(1024, 3)] * 4"  # next 4 layers have 1024 units
    convs += " + [(2048, 1)] * 2"  # final 2 layers use 1x1 convolutions

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_layers = getattr(args, "encoder_layers", convs)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_layers = getattr(args, "decoder_layers", convs)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    base_architecture(args)


@register_model_architecture("fconv", "fconv_wmt_en_fr")
def fconv_wmt_en_fr(args):
    convs = "[(512, 3)] * 6"  # first 6 layers have 512 units
    convs += " + [(768, 3)] * 4"  # next 4 layers have 768 units
    convs += " + [(1024, 3)] * 3"  # next 3 layers have 1024 units
    convs += " + [(2048, 1)] * 1"  # next 1 layer uses 1x1 convolutions
    convs += " + [(4096, 1)] * 1"  # final 1 layer uses 1x1 convolutions

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_layers = getattr(args, "encoder_layers", convs)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_layers = getattr(args, "decoder_layers", convs)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    base_architecture(args)
