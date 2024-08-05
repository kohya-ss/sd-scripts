import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from library.strategy_base import TokenizeStrategy, TextEncodingStrategy, TextEncoderOutputsCachingStrategy


from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


class SdxlTokenizeStrategy(TokenizeStrategy):
    def __init__(self, max_length: Optional[int], tokenizer_cache_dir: Optional[str] = None) -> None:
        self.tokenizer1 = self._load_tokenizer(CLIPTokenizer, TOKENIZER1_PATH, tokenizer_cache_dir=tokenizer_cache_dir)
        self.tokenizer2 = self._load_tokenizer(CLIPTokenizer, TOKENIZER2_PATH, tokenizer_cache_dir=tokenizer_cache_dir)
        self.tokenizer2.pad_token_id = 0  # use 0 as pad token for tokenizer2

        if max_length is None:
            self.max_length = self.tokenizer1.model_max_length
        else:
            self.max_length = max_length + 2

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        return (
            torch.stack([self._get_input_ids(self.tokenizer1, t, self.max_length) for t in text], dim=0),
            torch.stack([self._get_input_ids(self.tokenizer2, t, self.max_length) for t in text], dim=0),
        )


class SdxlTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        pass

    def _pool_workaround(
        self, text_encoder: CLIPTextModelWithProjection, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
    ):
        r"""
        workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
        instead of the hidden states for the EOS token
        If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

        Original code from CLIP's pooling function:

        \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
        \# take features from the eot embedding (eot_token is the highest number in each sequence)
        \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
        """

        # input_ids: b*n,77
        # find index for EOS token

        # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
        # eos_token_index = torch.where(input_ids == eos_token_id)[1]
        # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

        # Create a mask where the EOS tokens are
        eos_token_mask = (input_ids == eos_token_id).int()

        # Use argmax to find the last index of the EOS token for each element in the batch
        eos_token_index = torch.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine
        eos_token_index = eos_token_index.to(device=last_hidden_state.device)

        # get hidden states for EOS token
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index
        ]

        # apply projection: projection may be of different dtype than last_hidden_state
        pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
        pooled_output = pooled_output.to(last_hidden_state.dtype)

        return pooled_output

    def _get_hidden_states_sdxl(
        self,
        input_ids1: torch.Tensor,
        input_ids2: torch.Tensor,
        tokenizer1: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        text_encoder1: Union[CLIPTextModel, torch.nn.Module],
        text_encoder2: Union[CLIPTextModelWithProjection, torch.nn.Module],
        unwrapped_text_encoder2: Optional[CLIPTextModelWithProjection] = None,
    ):
        # input_ids: b,n,77 -> b*n, 77
        b_size = input_ids1.size()[0]
        max_token_length = input_ids1.size()[1] * input_ids1.size()[2]
        input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
        input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77
        input_ids1 = input_ids1.to(text_encoder1.device)
        input_ids2 = input_ids2.to(text_encoder2.device)

        # text_encoder1
        enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
        hidden_states1 = enc_out["hidden_states"][11]

        # text_encoder2
        enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
        hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

        # pool2 = enc_out["text_embeds"]
        unwrapped_text_encoder2 = unwrapped_text_encoder2 or text_encoder2
        pool2 = self._pool_workaround(unwrapped_text_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id)

        # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
        n_size = 1 if max_token_length is None else max_token_length // 75
        hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
        hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

        if max_token_length is not None:
            # bs*3, 77, 768 or 1024
            # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer1.model_max_length):
                states_list.append(hidden_states1[:, i : i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
            states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
            hidden_states1 = torch.cat(states_list, dim=1)

            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
            states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer2.model_max_length):
                chunk = hidden_states2[:, i : i + tokenizer2.model_max_length - 2]  # <BOS> の後から 最後の前まで
                # this causes an error:
                # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                # if i > 1:
                #     for j in range(len(chunk)):  # batch_size
                #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
            states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
            hidden_states2 = torch.cat(states_list, dim=1)

            # pool はnの最初のものを使う
            pool2 = pool2[::n_size]

        return hidden_states1, hidden_states2, pool2

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of models, [text_encoder1, text_encoder2, unwrapped text_encoder2 (optional)]
            tokens: List of tokens, for text_encoder1 and text_encoder2
        """
        if len(models) == 2:
            text_encoder1, text_encoder2 = models
            unwrapped_text_encoder2 = None
        else:
            text_encoder1, text_encoder2, unwrapped_text_encoder2 = models
        tokens1, tokens2 = tokens
        sdxl_tokenize_strategy = tokenize_strategy  # type: SdxlTokenizeStrategy
        tokenizer1, tokenizer2 = sdxl_tokenize_strategy.tokenizer1, sdxl_tokenize_strategy.tokenizer2

        hidden_states1, hidden_states2, pool2 = self._get_hidden_states_sdxl(
            tokens1, tokens2, tokenizer1, tokenizer2, text_encoder1, text_encoder2, unwrapped_text_encoder2
        )
        return [hidden_states1, hidden_states2, pool2]


class SdxlTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    SDXL_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_te_outputs.npz"

    def __init__(
        self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, is_partial: bool = False
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + SdxlTextEncoderOutputsCachingStrategy.SDXL_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "hidden_state1" not in npz or "hidden_state2" not in npz or "pool2" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        hidden_state1 = data["hidden_state1"]
        hidden_state2 = data["hidden_state2"]
        pool2 = data["pool2"]
        return [hidden_state1, hidden_state2, pool2]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        sdxl_text_encoding_strategy = text_encoding_strategy  # type: SdxlTextEncodingStrategy
        captions = [info.caption for info in infos]

        tokens1, tokens2 = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            hidden_state1, hidden_state2, pool2 = sdxl_text_encoding_strategy.encode_tokens(
                tokenize_strategy, models, [tokens1, tokens2]
            )
        if hidden_state1.dtype == torch.bfloat16:
            hidden_state1 = hidden_state1.float()
        if hidden_state2.dtype == torch.bfloat16:
            hidden_state2 = hidden_state2.float()
        if pool2.dtype == torch.bfloat16:
            pool2 = pool2.float()

        hidden_state1 = hidden_state1.cpu().numpy()
        hidden_state2 = hidden_state2.cpu().numpy()
        pool2 = pool2.cpu().numpy()

        for i, info in enumerate(infos):
            hidden_state1_i = hidden_state1[i]
            hidden_state2_i = hidden_state2[i]
            pool2_i = pool2[i]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    hidden_state1=hidden_state1_i,
                    hidden_state2=hidden_state2_i,
                    pool2=pool2_i,
                )
            else:
                info.text_encoder_outputs = [hidden_state1_i, hidden_state2_i, pool2_i]
