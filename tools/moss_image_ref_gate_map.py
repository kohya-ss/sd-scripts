# Print the learned reference gate table (block x timestep band) stored in a MOSS-Image editing LoRA file
# (moss_image_train_network_edit.py --ref_mode kv --ref_gate).
#
#   python tools/moss_image_ref_gate_map.py path/to/lora.safetensors [--budget K]
#
# --budget K also prints the blocks which would stay open per band after the binarization (top-K per band).

import argparse

import torch
from safetensors import safe_open

from library import moss_image_ref_gate


def main():
    parser = argparse.ArgumentParser(description="Print the reference gate table of a MOSS-Image editing LoRA")
    parser.add_argument("lora", type=str, help="LoRA file with ref_gate.* keys")
    parser.add_argument("--budget", type=int, default=None, help="show the top-K blocks per band (binarization preview)")
    args = parser.parse_args()

    with safe_open(args.lora, "pt") as f:
        if "ref_gate.logits" not in f.keys():
            raise SystemExit(f"no reference gate table (ref_gate.logits) in {args.lora}")
        logits = f.get_tensor("ref_gate.logits").float()
        edges = f.get_tensor("ref_gate.band_edges").float().tolist()
        metadata = f.metadata() or {}

    gate = moss_image_ref_gate.ReferenceGate(logits.shape[0], edges, trainable=False)
    gate.logits.copy_(logits)
    for key in ["ss_ref_gate_budget", "ss_ref_gate_binarize", "ss_ref_gate_bands", "ss_ref_timestep", "ss_ref_kv_lora"]:
        if key in metadata:
            print(f"{key}: {metadata[key]}")
    print(gate.table_string())

    if args.budget is not None:
        top = logits.topk(min(args.budget, logits.shape[0]), dim=0).indices
        for b, (lo, hi) in enumerate(gate.band_ranges()):
            print(f"band {int(hi)}-{int(lo)}: top-{args.budget} blocks {sorted(top[:, b].tolist())}")


if __name__ == "__main__":
    main()
