# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import logging
import os
import os.path as op
import sys

from dump_hubert_feature import HubertFeatureReader
from feature_utils import get_shard_range, dump_feature
from fairseq.data.audio.audio_utils import get_waveform
from fairseq.data.audio.speech_to_text_dataset import (
    read_from_uncompressed_zip,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature_s2t")


class HubertFeatureReaderS2T(HubertFeatureReader):
    def read_audio(self, path, ref_len=None):
        path, *extra = path.split(":")
        assert len(extra) == 2
        assert path.endswith(".zip")

        data = read_from_uncompressed_zip(path, int(extra[0]), int(extra[1]))
        f = io.BytesIO(data)
        wav, sr = get_waveform(f)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav


def get_path_iterator(root, tsv, nshard, rank):
    with open(tsv) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        subpaths = [op.join(root, e["audio"]) for e in reader]
        start, end = get_shard_range(len(subpaths), nshard, rank)
        subpaths = subpaths[start:end]
        def iterate():
            for subpath in subpaths:
                yield op.join(root, subpath), None
    return iterate, len(subpaths)


def main(
    root, tsv_path, ckpt_path, layer, nshard, rank, feat_dir, split, max_chunk
):
    reader = HubertFeatureReaderS2T(ckpt_path, layer, max_chunk)
    generator, num = get_path_iterator(root, tsv_path, nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("tsv_path")
    parser.add_argument("ckpt_path")
    parser.add_argument("layer", type=int)
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("split")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
