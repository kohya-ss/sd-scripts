# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import string
import math

import torch

from ofa.data import data_utils


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


def eval_refcoco(task, generator, models, sample, **kwargs):
    def _calculate_ap_score(hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    gen_out = task.inference_step(generator, models, sample)
    hyps = []
    for i in range(len(gen_out)):
        hyps.append(gen_out[i][0]["tokens"][:-1] - len(task.src_dict) + task.cfg.num_bins)
    hyps = torch.stack(hyps, dim=0)
    hyps = hyps / (task.cfg.num_bins - 1) * task.cfg.max_image_size
    hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
    hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

    results = [
        {"uniq_id": sample_id,
         "box": [hyps[i][0].item(), hyps[i][1].item(), hyps[i][2].item(), hyps[i][3].item()]}
        for i, sample_id in enumerate(sample["id"].tolist())
    ]
    scores = _calculate_ap_score(hyps, sample['region_coords'].float())
    return results, scores


def eval_snli_ve(task, generator, models, sample, **kwargs):
    encoder_out = models[0].encoder(
        sample["net_input"]["src_tokens"],
        src_lengths=sample["net_input"]["src_lengths"],
        patch_images=sample["net_input"]["patch_images"],
        patch_masks=sample["net_input"]["patch_masks"]
    )
    device = sample["net_input"]["src_tokens"].device
    eos_item = torch.tensor([task.src_dict.eos()])
    pad = task.src_dict.pad()
    valid_result = []
    for valid_answers, valid_constraint_masks in zip(task.valid_answers_list, task.valid_constraint_masks_list):
        valid_size = len(valid_answers)
        valid_tgt_items = [
            torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
            for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
        ]
        valid_prev_items = [
            torch.cat([torch.tensor(decoder_prompt), valid_answer])
            for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
        ]
        valid_constraint_mask_items = [
            torch.cat(
                [torch.zeros(len(decoder_prompt) - 1, valid_constraint_mask.size(1)).bool(), valid_constraint_mask],
                dim=0
            )
            for decoder_prompt in sample["decoder_prompts"] for valid_constraint_mask in valid_constraint_masks
        ]
        valid_tgt = data_utils.collate_tokens(valid_tgt_items, pad_idx=pad).to(device)
        valid_prev_output = data_utils.collate_tokens(valid_prev_items, pad_idx=pad).to(device)
        valid_constraint_masks = data_utils.collate_tokens(valid_constraint_mask_items, pad_idx=pad).to(device)

        new_encoder_out = {}
        new_encoder_out["encoder_out"] = [
            encoder_out["encoder_out"][0].repeat_interleave(valid_size, dim=1)
        ]
        new_encoder_out["encoder_padding_mask"] = [
            encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_size, dim=0)
        ]
        new_encoder_out["position_embeddings"] = [
            encoder_out["position_embeddings"][0].repeat_interleave(valid_size, dim=0)
        ]

        decoder_out = models[0].decoder(valid_prev_output, encoder_out=new_encoder_out)
        decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
        lprobs = models[0].get_normalized_probs(decoder_out, log_probs=True)
        scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(valid_tgt.eq(task.tgt_dict.pad()), 0)
        scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
        scores = scores.sum(1)
        scores = scores.view(-1, valid_size)
        valid_result.append(scores)
    valid_result = torch.cat(valid_result, dim=-1)
    predicts = valid_result.argmax(1).tolist()
    hyps = [task.index2ans[predict_index] for predict_index in predicts]
    results = [{"uniq_id": id, "answer": hyp} for id, hyp in zip(sample["id"].tolist(), hyps)]
    scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
    return results, scores


def eval_vqa_gen(task, generator, models, sample, **kwargs):
    hypos = task.inference_step(generator, models, sample)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        detok_hypo_str = decode_fn(hypos[i][0]["tokens"], task.tgt_dict, task.bpe, generator)
        results.append({"question_id": sample_id, "answer": detok_hypo_str.strip()})
    scores = [ref_dict.get(result['answer'], 0) for ref_dict, result in zip(sample['ref_dict'], results)]
    return results, scores


def zero_shot_step(task, generator, models, sample, **kwargs):
    generator.zero_shot = True
    generator.constraint_trie = None
    if task.cfg._name == 'vqa_gen':
        return eval_vqa_gen(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'refcoco':
        return eval_refcoco(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'snli_ve':
        return eval_snli_ve(task, generator, models, sample, **kwargs)
    else:
        raise NotImplementedError
