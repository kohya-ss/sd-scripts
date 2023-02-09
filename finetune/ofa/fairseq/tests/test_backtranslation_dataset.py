# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import tests.utils as test_utils
import torch
from fairseq.data import (
    BacktranslationDataset,
    LanguagePairDataset,
    TransformEosDataset,
)
from fairseq.sequence_generator import SequenceGenerator


class TestBacktranslationDataset(unittest.TestCase):
    def setUp(self):
        (
            self.tgt_dict,
            self.w1,
            self.w2,
            self.src_tokens,
            self.src_lengths,
            self.model,
        ) = test_utils.sequence_generator_setup()

        dummy_src_samples = self.src_tokens

        self.tgt_dataset = test_utils.TestDataset(data=dummy_src_samples)
        self.cuda = torch.cuda.is_available()

    def _backtranslation_dataset_helper(
        self,
        remove_eos_from_input_src,
        remove_eos_from_output_src,
    ):
        tgt_dataset = LanguagePairDataset(
            src=self.tgt_dataset,
            src_sizes=self.tgt_dataset.sizes,
            src_dict=self.tgt_dict,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=None,
        )

        generator = SequenceGenerator(
            [self.model],
            tgt_dict=self.tgt_dict,
            max_len_a=0,
            max_len_b=200,
            beam_size=2,
            unk_penalty=0,
        )

        backtranslation_dataset = BacktranslationDataset(
            tgt_dataset=TransformEosDataset(
                dataset=tgt_dataset,
                eos=self.tgt_dict.eos(),
                # remove eos from the input src
                remove_eos_from_src=remove_eos_from_input_src,
            ),
            src_dict=self.tgt_dict,
            backtranslation_fn=(
                lambda sample: generator.generate([self.model], sample)
            ),
            output_collater=TransformEosDataset(
                dataset=tgt_dataset,
                eos=self.tgt_dict.eos(),
                # if we remove eos from the input src, then we need to add it
                # back to the output tgt
                append_eos_to_tgt=remove_eos_from_input_src,
                remove_eos_from_src=remove_eos_from_output_src,
            ).collater,
            cuda=self.cuda,
        )
        dataloader = torch.utils.data.DataLoader(
            backtranslation_dataset,
            batch_size=2,
            collate_fn=backtranslation_dataset.collater,
        )
        backtranslation_batch_result = next(iter(dataloader))

        eos, pad, w1, w2 = self.tgt_dict.eos(), self.tgt_dict.pad(), self.w1, self.w2

        # Note that we sort by src_lengths and add left padding, so actually
        # ids will look like: [1, 0]
        expected_src = torch.LongTensor([[w1, w2, w1, eos], [pad, pad, w1, eos]])
        if remove_eos_from_output_src:
            expected_src = expected_src[:, :-1]
        expected_tgt = torch.LongTensor([[w1, w2, eos], [w1, w2, eos]])
        generated_src = backtranslation_batch_result["net_input"]["src_tokens"]
        tgt_tokens = backtranslation_batch_result["target"]

        self.assertTensorEqual(expected_src, generated_src)
        self.assertTensorEqual(expected_tgt, tgt_tokens)

    def test_backtranslation_dataset_no_eos_in_output_src(self):
        self._backtranslation_dataset_helper(
            remove_eos_from_input_src=False,
            remove_eos_from_output_src=True,
        )

    def test_backtranslation_dataset_with_eos_in_output_src(self):
        self._backtranslation_dataset_helper(
            remove_eos_from_input_src=False,
            remove_eos_from_output_src=False,
        )

    def test_backtranslation_dataset_no_eos_in_input_src(self):
        self._backtranslation_dataset_helper(
            remove_eos_from_input_src=True,
            remove_eos_from_output_src=False,
        )

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


if __name__ == "__main__":
    unittest.main()
