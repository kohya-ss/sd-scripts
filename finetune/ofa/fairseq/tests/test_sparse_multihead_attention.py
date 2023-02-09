# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from fairseq.modules.sparse_multihead_attention import SparseMultiheadAttention


class TestSparseMultiheadAttention(unittest.TestCase):
    def test_sparse_multihead_attention(self):
        attn_weights = torch.randn(1, 8, 8)
        bidirectional_sparse_mask = torch.tensor(
            [
                [0, 0, 0, 0, 0, float("-inf"), float("-inf"), 0],
                [0, 0, 0, 0, 0, float("-inf"), float("-inf"), 0],
                [0, 0, 0, 0, 0, float("-inf"), float("-inf"), 0],
                [0, 0, 0, 0, 0, float("-inf"), float("-inf"), 0],
                [float("-inf"), float("-inf"), float("-inf"), 0, 0, 0, 0, 0],
                [float("-inf"), float("-inf"), float("-inf"), 0, 0, 0, 0, 0],
                [float("-inf"), float("-inf"), float("-inf"), 0, 0, 0, 0, 0],
                [float("-inf"), float("-inf"), float("-inf"), 0, 0, 0, 0, 0],
            ]
        )

        bidirectional_attention = SparseMultiheadAttention(
            16, 1, stride=4, expressivity=1, is_bidirectional=True
        )
        bidirectional_attention_sparse_mask = (
            bidirectional_attention.buffered_sparse_mask(attn_weights, 8, 8)
        )
        torch.all(
            torch.eq(bidirectional_attention_sparse_mask, bidirectional_sparse_mask)
        )

        sparse_mask = torch.tensor(
            [
                [
                    0,
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                ],
                [
                    0,
                    0,
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                ],
                [
                    0,
                    0,
                    0,
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                ],
                [0, 0, 0, 0, 0, float("-inf"), float("-inf"), float("-inf")],
                [
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    0,
                    0,
                    0,
                    float("-inf"),
                    float("-inf"),
                ],
                [
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    0,
                    0,
                    0,
                    0,
                    float("-inf"),
                ],
                [float("-inf"), float("-inf"), float("-inf"), 0, 0, 0, 0, 0],
            ]
        )

        attention = SparseMultiheadAttention(
            16, 1, stride=4, expressivity=1, is_bidirectional=False
        )
        attention_sparse_mask = attention.buffered_sparse_mask(attn_weights, 8, 8)

        torch.all(torch.eq(attention_sparse_mask, sparse_mask))


if __name__ == "__main__":
    unittest.main()
