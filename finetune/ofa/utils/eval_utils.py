# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import string
import math
import json
from itertools import chain
import os

# import torch
import torch.distributed as dist
from fairseq import utils

# from ofa.data import data_utils
# from ofa.tasks.nlg_tasks.gigaword import fix_tokenization
# import editdistance


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x

def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def eval_caption(task, generator, models, sample, **kwargs):
    transtab = str.maketrans({key: None for key in string.punctuation})
    hypos = task.inference_step(generator, models, sample, **kwargs)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        for hypo in hypos[i]:
            detok_hypo_str = decode_fn(hypo["tokens"], task.tgt_dict, task.bpe, generator)
            results.append({"image_id": str(sample_id), "caption": detok_hypo_str.translate(transtab).strip()})
    return results, None


def eval_caption_cn(task, generator, models, sample, **kwargs):
    hypos = task.inference_step(generator, models, sample, **kwargs)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        detok_hypo_str = decode_fn(
            hypos[i][0]["tokens"], task.tgt_dict, task.bpe, generator
        )
        results.append(
            {
                "image_id": str(sample_id),
                "caption": detok_hypo_str.strip(),
            }
        )
    return results, None

def eval_step(task, generator, models, sample, **kwargs):
    if task.cfg._name == 'caption':
        return eval_caption(task, generator, models, sample, **kwargs)
    elif task.cfg._name == "caption_cn":
        return eval_caption_cn(task, generator, models, sample, **kwargs)
    else:
        raise NotImplementedError


def merge_results(task, cfg, logger, score_cnt, score_sum, results):
    if task.cfg._name == 'image_gen':
        if cfg.distributed_training.distributed_world_size > 1:
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info("score_sum: {}, score_cnt: {}, score: {}".format(
                score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
            ))
    else:
        gather_results = None
        if cfg.distributed_training.distributed_world_size > 1:
            gather_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_results, results)
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info("score_sum: {}, score_cnt: {}, score: {}".format(
                score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
            ))

        if cfg.distributed_training.distributed_world_size == 1 or dist.get_rank() == 0:
            os.makedirs(cfg.common_eval.results_path, exist_ok=True)
            output_path = os.path.join(cfg.common_eval.results_path, "{}_predict.json".format(cfg.dataset.gen_subset))
            gather_results = list(chain(*gather_results)) if gather_results is not None else results
            with open(output_path, 'w') as fw:
                json.dump(gather_results, fw)
