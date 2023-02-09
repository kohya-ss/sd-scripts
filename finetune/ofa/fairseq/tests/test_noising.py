# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, List

import tests.utils as test_utils
import torch
from fairseq import utils
from fairseq.data import (
    Dictionary,
    LanguagePairDataset,
    TransformEosDataset,
    data_utils,
    noising,
)


class TestDataNoising(unittest.TestCase):
    def _get_test_data_with_bpe_cont_marker(self, append_eos=True):
        """
        Args:
            append_eos: if True, each input sentence in the source tokens tensor
                will have an EOS appended to the end.

        Returns:
            vocabs: BPE vocab with continuation markers as suffixes to denote
                non-end of word tokens. This is the standard BPE format used in
                fairseq's preprocessing.
            x: input tensor containing numberized source tokens, with EOS at the
                end if append_eos is true
            src_lengths: and source lengths.
        """
        vocab = Dictionary()
        vocab.add_symbol("he@@")
        vocab.add_symbol("llo")
        vocab.add_symbol("how")
        vocab.add_symbol("are")
        vocab.add_symbol("y@@")
        vocab.add_symbol("ou")
        vocab.add_symbol("n@@")
        vocab.add_symbol("ew")
        vocab.add_symbol("or@@")
        vocab.add_symbol("k")

        src_tokens = [
            ["he@@", "llo", "n@@", "ew", "y@@", "or@@", "k"],
            ["how", "are", "y@@", "ou"],
        ]
        x, src_lengths = x, src_lengths = self._convert_src_tokens_to_tensor(
            vocab=vocab, src_tokens=src_tokens, append_eos=append_eos
        )
        return vocab, x, src_lengths

    def _get_test_data_with_bpe_end_marker(self, append_eos=True):
        """
        Args:
            append_eos: if True, each input sentence in the source tokens tensor
                will have an EOS appended to the end.

        Returns:
            vocabs: BPE vocab with end-of-word markers as suffixes to denote
                tokens at the end of a word. This is an alternative to fairseq's
                standard preprocessing framework and is not generally supported
                within fairseq.
            x: input tensor containing numberized source tokens, with EOS at the
                end if append_eos is true
            src_lengths: and source lengths.
        """
        vocab = Dictionary()
        vocab.add_symbol("he")
        vocab.add_symbol("llo_EOW")
        vocab.add_symbol("how_EOW")
        vocab.add_symbol("are_EOW")
        vocab.add_symbol("y")
        vocab.add_symbol("ou_EOW")
        vocab.add_symbol("n")
        vocab.add_symbol("ew_EOW")
        vocab.add_symbol("or")
        vocab.add_symbol("k_EOW")

        src_tokens = [
            ["he", "llo_EOW", "n", "ew_EOW", "y", "or", "k_EOW"],
            ["how_EOW", "are_EOW", "y", "ou_EOW"],
        ]
        x, src_lengths = x, src_lengths = self._convert_src_tokens_to_tensor(
            vocab=vocab, src_tokens=src_tokens, append_eos=append_eos
        )
        return vocab, x, src_lengths

    def _get_test_data_with_word_vocab(self, append_eos=True):
        """
        Args:
            append_eos: if True, each input sentence in the source tokens tensor
                will have an EOS appended to the end.

        Returns:
            vocabs: word vocab
            x: input tensor containing numberized source tokens, with EOS at the
                end if append_eos is true
            src_lengths: and source lengths.
        """
        vocab = Dictionary()

        vocab.add_symbol("hello")
        vocab.add_symbol("how")
        vocab.add_symbol("are")
        vocab.add_symbol("you")
        vocab.add_symbol("new")
        vocab.add_symbol("york")
        src_tokens = [
            ["hello", "new", "york", "you"],
            ["how", "are", "you", "new", "york"],
        ]
        x, src_lengths = self._convert_src_tokens_to_tensor(
            vocab=vocab, src_tokens=src_tokens, append_eos=append_eos
        )
        return vocab, x, src_lengths

    def _convert_src_tokens_to_tensor(
        self, vocab: Dictionary, src_tokens: List[List[str]], append_eos: bool
    ):
        src_len = [len(x) for x in src_tokens]
        # If we have to append EOS, we include EOS in counting src length
        if append_eos:
            src_len = [length + 1 for length in src_len]

        x = torch.LongTensor(len(src_tokens), max(src_len)).fill_(vocab.pad())
        for i in range(len(src_tokens)):
            for j in range(len(src_tokens[i])):
                x[i][j] = vocab.index(src_tokens[i][j])
            if append_eos:
                x[i][j + 1] = vocab.eos()

        x = x.transpose(1, 0)
        return x, torch.LongTensor(src_len)

    def assert_eos_at_end(self, x, x_len, eos):
        """Asserts last token of every sentence in x is EOS """
        for i in range(len(x_len)):
            self.assertEqual(
                x[x_len[i] - 1][i],
                eos,
                (
                    "Expected eos (token id {eos}) at the end of sentence {i} "
                    "but got {other} instead"
                ).format(i=i, eos=eos, other=x[i][-1]),
            )

    def assert_word_dropout_correct(self, x, x_noised, x_len, l_noised):
        # Expect only the first word (2 bpe tokens) of the first example
        # was dropped out
        self.assertEqual(x_len[0] - 2, l_noised[0])
        for i in range(l_noised[0]):
            self.assertEqual(x_noised[i][0], x[i + 2][0])

    def test_word_dropout_with_eos(self):
        vocab, x, x_len = self._get_test_data_with_bpe_cont_marker(append_eos=True)

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2)
            self.assert_word_dropout_correct(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def assert_word_blanking_correct(self, x, x_noised, x_len, l_noised, unk):
        # Expect only the first word (2 bpe tokens) of the first example
        # was blanked out
        self.assertEqual(x_len[0], l_noised[0])
        for i in range(l_noised[0]):
            if i < 2:
                self.assertEqual(x_noised[i][0], unk)
            else:
                self.assertEqual(x_noised[i][0], x[i][0])

    def test_word_blank_with_eos(self):
        vocab, x, x_len = self._get_test_data_with_bpe_cont_marker(append_eos=True)

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2, vocab.unk())
            self.assert_word_blanking_correct(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised, unk=vocab.unk()
            )
            self.assert_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def generate_unchanged_shuffle_map(self, length):
        return {i: i for i in range(length)}

    def assert_word_shuffle_matches_expected(
        self,
        x,
        x_len,
        max_shuffle_distance: int,
        vocab: Dictionary,
        expected_shufle_maps: List[Dict[int, int]],
        expect_eos_at_end: bool,
        bpe_end_marker=None,
    ):
        """
        This verifies that with a given x, x_len, max_shuffle_distance, and
        vocab, we get the expected shuffle result.

        Args:
            x: Tensor of shape (T x B) = (sequence_length, batch_size)
            x_len: Tensor of length B = batch_size
            max_shuffle_distance: arg to pass to noising
            expected_shuffle_maps: List[mapping] where mapping is a
                Dict[old_index, new_index], mapping x's elements from their
                old positions in x to their new positions in x.
            expect_eos_at_end: if True, check the output to make sure there is
                an EOS at the end.
            bpe_end_marker: str denoting the BPE end token. If this is not None, we
                set the BPE cont token to None in the noising classes.
        """
        bpe_cont_marker = None
        if bpe_end_marker is None:
            bpe_cont_marker = "@@"

        with data_utils.numpy_seed(1234):
            word_shuffle = noising.WordShuffle(
                vocab, bpe_cont_marker=bpe_cont_marker, bpe_end_marker=bpe_end_marker
            )
            x_noised, l_noised = word_shuffle.noising(
                x, x_len, max_shuffle_distance=max_shuffle_distance
            )

        # For every example, we have a different expected shuffle map. We check
        # that each example is shuffled as expected according to each
        # corresponding shuffle map.
        for i in range(len(expected_shufle_maps)):
            shuffle_map = expected_shufle_maps[i]
            for k, v in shuffle_map.items():
                self.assertEqual(x[k][i], x_noised[v][i])

        # Shuffling should not affect the length of each example
        for pre_shuffle_length, post_shuffle_length in zip(x_len, l_noised):
            self.assertEqual(pre_shuffle_length, post_shuffle_length)
        if expect_eos_at_end:
            self.assert_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def test_word_shuffle_with_eos(self):
        vocab, x, x_len = self._get_test_data_with_bpe_cont_marker(append_eos=True)

        # Assert word shuffle with max shuffle distance 0 causes input to be
        # unchanged
        self.assert_word_shuffle_matches_expected(
            x=x,
            x_len=x_len,
            max_shuffle_distance=0,
            vocab=vocab,
            expected_shufle_maps=[
                self.generate_unchanged_shuffle_map(example_len)
                for example_len in x_len
            ],
            expect_eos_at_end=True,
        )

        # Assert word shuffle with max shuffle distance 3 matches our expected
        # shuffle order
        self.assert_word_shuffle_matches_expected(
            x=x,
            x_len=x_len,
            vocab=vocab,
            max_shuffle_distance=3,
            expected_shufle_maps=[
                self.generate_unchanged_shuffle_map(x_len[0]),
                {0: 0, 1: 3, 2: 1, 3: 2},
            ],
            expect_eos_at_end=True,
        )

    def test_word_shuffle_with_eos_nonbpe(self):
        """The purpose of this is to test shuffling logic with word vocabs"""
        vocab, x, x_len = self._get_test_data_with_word_vocab(append_eos=True)

        # Assert word shuffle with max shuffle distance 0 causes input to be
        # unchanged
        self.assert_word_shuffle_matches_expected(
            x=x,
            x_len=x_len,
            max_shuffle_distance=0,
            vocab=vocab,
            expected_shufle_maps=[
                self.generate_unchanged_shuffle_map(example_len)
                for example_len in x_len
            ],
            expect_eos_at_end=True,
        )

        # Assert word shuffle with max shuffle distance 3 matches our expected
        # shuffle order
        self.assert_word_shuffle_matches_expected(
            x=x,
            x_len=x_len,
            vocab=vocab,
            max_shuffle_distance=3,
            expected_shufle_maps=[
                {0: 0, 1: 1, 2: 3, 3: 2},
                {0: 0, 1: 2, 2: 1, 3: 3, 4: 4},
            ],
            expect_eos_at_end=True,
        )

    def test_word_shuffle_without_eos(self):
        """Same result as word shuffle with eos except no EOS at end"""
        vocab, x, x_len = self._get_test_data_with_bpe_cont_marker(append_eos=False)

        # Assert word shuffle with max shuffle distance 0 causes input to be
        # unchanged
        self.assert_word_shuffle_matches_expected(
            x=x,
            x_len=x_len,
            max_shuffle_distance=0,
            vocab=vocab,
            expected_shufle_maps=[
                self.generate_unchanged_shuffle_map(example_len)
                for example_len in x_len
            ],
            expect_eos_at_end=False,
        )

        # Assert word shuffle with max shuffle distance 3 matches our expected
        # shuffle order
        self.assert_word_shuffle_matches_expected(
            x=x,
            x_len=x_len,
            vocab=vocab,
            max_shuffle_distance=3,
            expected_shufle_maps=[
                self.generate_unchanged_shuffle_map(x_len[0]),
                {0: 0, 1: 3, 2: 1, 3: 2},
            ],
            expect_eos_at_end=False,
        )

    def test_word_shuffle_without_eos_with_bpe_end_marker(self):
        """Same result as word shuffle without eos except using BPE end token"""
        vocab, x, x_len = self._get_test_data_with_bpe_end_marker(append_eos=False)

        # Assert word shuffle with max shuffle distance 0 causes input to be
        # unchanged
        self.assert_word_shuffle_matches_expected(
            x=x,
            x_len=x_len,
            max_shuffle_distance=0,
            vocab=vocab,
            expected_shufle_maps=[
                self.generate_unchanged_shuffle_map(example_len)
                for example_len in x_len
            ],
            expect_eos_at_end=False,
            bpe_end_marker="_EOW",
        )

        # Assert word shuffle with max shuffle distance 3 matches our expected
        # shuffle order
        self.assert_word_shuffle_matches_expected(
            x=x,
            x_len=x_len,
            vocab=vocab,
            max_shuffle_distance=3,
            expected_shufle_maps=[
                self.generate_unchanged_shuffle_map(x_len[0]),
                {0: 0, 1: 3, 2: 1, 3: 2},
            ],
            expect_eos_at_end=False,
            bpe_end_marker="_EOW",
        )

    def assert_no_eos_at_end(self, x, x_len, eos):
        """Asserts that the last token of each sentence in x is not EOS """
        for i in range(len(x_len)):
            self.assertNotEqual(
                x[x_len[i] - 1][i],
                eos,
                "Expected no eos (token id {eos}) at the end of sentence {i}.".format(
                    eos=eos, i=i
                ),
            )

    def test_word_dropout_without_eos(self):
        """Same result as word dropout with eos except no EOS at end"""
        vocab, x, x_len = self._get_test_data_with_bpe_cont_marker(append_eos=False)

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2)
            self.assert_word_dropout_correct(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_no_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def test_word_blank_without_eos(self):
        """Same result as word blank with eos except no EOS at end"""
        vocab, x, x_len = self._get_test_data_with_bpe_cont_marker(append_eos=False)

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2, vocab.unk())
            self.assert_word_blanking_correct(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised, unk=vocab.unk()
            )
            self.assert_no_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def _get_noising_dataset_batch(
        self,
        src_tokens_no_pad,
        src_dict,
        append_eos_to_tgt=False,
    ):
        """
        Constructs a NoisingDataset and the corresponding
        ``LanguagePairDataset(NoisingDataset(src), src)``. If
        *append_eos_to_tgt* is True, wrap the source dataset in
        :class:`TransformEosDataset` to append EOS to the clean source when
        using it as the target.
        """
        src_dataset = test_utils.TestDataset(data=src_tokens_no_pad)

        noising_dataset = noising.NoisingDataset(
            src_dataset=src_dataset,
            src_dict=src_dict,
            seed=1234,
            max_word_shuffle_distance=3,
            word_dropout_prob=0.2,
            word_blanking_prob=0.2,
            noising_class=noising.UnsupervisedMTNoising,
        )
        tgt = src_dataset
        language_pair_dataset = LanguagePairDataset(
            src=noising_dataset, tgt=tgt, src_sizes=None, src_dict=src_dict
        )
        language_pair_dataset = TransformEosDataset(
            language_pair_dataset,
            src_dict.eos(),
            append_eos_to_tgt=append_eos_to_tgt,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=language_pair_dataset,
            batch_size=2,
            collate_fn=language_pair_dataset.collater,
        )
        denoising_batch_result = next(iter(dataloader))
        return denoising_batch_result

    def test_noising_dataset_with_eos(self):
        src_dict, src_tokens, _ = self._get_test_data_with_bpe_cont_marker(
            append_eos=True
        )

        # Format data for src_dataset
        src_tokens = torch.t(src_tokens)
        src_tokens_no_pad = []
        for src_sentence in src_tokens:
            src_tokens_no_pad.append(
                utils.strip_pad(tensor=src_sentence, pad=src_dict.pad())
            )
        denoising_batch_result = self._get_noising_dataset_batch(
            src_tokens_no_pad=src_tokens_no_pad, src_dict=src_dict
        )

        eos, pad = src_dict.eos(), src_dict.pad()

        # Generated noisy source as source
        expected_src = torch.LongTensor(
            [[4, 5, 10, 11, 8, 12, 13, eos], [pad, pad, pad, 6, 8, 9, 7, eos]]
        )
        # Original clean source as target (right-padded)
        expected_tgt = torch.LongTensor(
            [[4, 5, 10, 11, 8, 12, 13, eos], [6, 7, 8, 9, eos, pad, pad, pad]]
        )
        generated_src = denoising_batch_result["net_input"]["src_tokens"]
        tgt_tokens = denoising_batch_result["target"]

        self.assertTensorEqual(expected_src, generated_src)
        self.assertTensorEqual(expected_tgt, tgt_tokens)

    def test_noising_dataset_without_eos(self):
        """
        Similar to test noising dataset with eos except that we have to set
        *append_eos_to_tgt* to ``True``.
        """

        src_dict, src_tokens, _ = self._get_test_data_with_bpe_cont_marker(
            append_eos=False
        )

        # Format data for src_dataset
        src_tokens = torch.t(src_tokens)
        src_tokens_no_pad = []
        for src_sentence in src_tokens:
            src_tokens_no_pad.append(
                utils.strip_pad(tensor=src_sentence, pad=src_dict.pad())
            )
        denoising_batch_result = self._get_noising_dataset_batch(
            src_tokens_no_pad=src_tokens_no_pad,
            src_dict=src_dict,
            append_eos_to_tgt=True,
        )

        eos, pad = src_dict.eos(), src_dict.pad()

        # Generated noisy source as source
        expected_src = torch.LongTensor(
            [[4, 5, 10, 11, 8, 12, 13], [pad, pad, pad, 6, 8, 9, 7]]
        )
        # Original clean source as target (right-padded)
        expected_tgt = torch.LongTensor(
            [[4, 5, 10, 11, 8, 12, 13, eos], [6, 7, 8, 9, eos, pad, pad, pad]]
        )

        generated_src = denoising_batch_result["net_input"]["src_tokens"]
        tgt_tokens = denoising_batch_result["target"]

        self.assertTensorEqual(expected_src, generated_src)
        self.assertTensorEqual(expected_tgt, tgt_tokens)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


if __name__ == "__main__":
    unittest.main()
