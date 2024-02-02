import argparse
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import numpy as np
from library import train_util

def sigma_rel_to_gamma(sigma_rel):
    t = sigma_rel ** -2
    gamma = np.roots([1, 7, 16 - t, 12 - t]).real.max()
    return gamma

def p_dot_p(t_a, gamma_a, t_b, gamma_b):
    t_ratio = t_a / t_b
    t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = np.maximum(t_a, t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio**t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den

def solve_weights(t_i, gamma_i, t_r, gamma_r):
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
    B = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
    X = np.linalg.solve(A, B)
    return X

# TODO add sample generation for different gammas


def reconstruct_weights_from_snapshots(args):
    # TODO add checks for target_step 

    args.device = "cpu"
    assert (args.target_sigma_rel or args.target_gamma) and not (args.target_sigma_rel and args.target_gamma), "Either target_sigma_rel or target_gamma is required"
    if args.target_sigma_rel:
        args.target_gamma = sigma_rel_to_gamma(args.target_sigma_rel)

    args.snapshot_dir = args.snapshot_dir.rstrip('\\').rstrip('/')
    snaps = os.listdir(args.snapshot_dir)
    print(f"{len(snaps)} snapshots found")
    gammas = [float(os.path.splitext(s)[0].split("_")[-1]) for s in snaps] 
    # # load gammas from snapshots 
    #gammas = []
    #for s in snaps:
    #    with safe_open(os.path.join(args.snapshot_dir, s), framework="pt", device=args.device) as f:
    #        gammas.append(float(f.get_tensor('ema_gamma')))
    ts = [int(os.path.splitext(s)[0].split("_")[-2]) for s in snaps]

    #if not args.target_step:
    #    args.target_step = ts[-1] + 1

    x = solve_weights(ts, gammas, ts[-1] + 1, args.target_gamma)    # x = solve_weights(ts, gammas, args.target_step, args.target_gamma)
    #print(x)
    x = torch.from_numpy(x)  # .to(device=args.device)

    if args.saving_precision == "fp16":
        save_dtype = torch.float16
    elif args.saving_precision == "bf16":
        save_dtype = torch.bfloat16
    else:
        save_dtype = torch.float

    supplementary_key_ratios = {}
    merged_sd = None
    first_model_keys = set()
    dtype = torch.float
    for i, s in enumerate(tqdm(snaps)):
        # load snapshots and add weights 

        if merged_sd is None:
            # load first model
            #print(f"Loading {s}, x = {x[i].item()}...")
            merged_sd = {}
            with safe_open(os.path.join(args.snapshot_dir, s), framework="pt", device=args.device) as f:
                for key in f.keys():
                    value = f.get_tensor(key)

                    first_model_keys.add(key)

                    value = x[i] * value.to(dtype)  # first model's value * ratio
                    merged_sd[key] = value

            print(f" Model has {len(merged_sd)} keys " )
            continue

        # load other models
        #print(f"Loading {s}, x = {x[i].item()}...")

        with safe_open(os.path.join(args.snapshot_dir, s), framework="pt", device=args.device) as f:
            model_keys = f.keys()
            for key in model_keys:
                if key not in merged_sd:
                    print(f"Skip: {key}")
                    continue

                value = f.get_tensor(key)
                merged_sd[key] = merged_sd[key] + x[i] * value.to(dtype)

            # enumerate keys not in this model
            model_keys = set(model_keys)
            for key in merged_sd.keys():
                if key in model_keys:
                    continue
                print(f"Key {key} not in model {s}, use first model's value")
                if key in supplementary_key_ratios:
                    supplementary_key_ratios[key] += x[i]
                else:
                    supplementary_key_ratios[key] = x[i]

    # save
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.snapshot_dir)
    output_file = os.path.join(args.output_dir, os.path.basename(args.snapshot_dir).replace("_snapshots","") + "_gamma_{:.2f}.safetensors".format(args.target_gamma))

    print(f"Saving to {output_file}...")

    # convert to save_dtype
    for k in merged_sd.keys():
        merged_sd[k] = merged_sd[k].to(save_dtype)
    
    metadata = {}
    with safe_open(os.path.join(args.snapshot_dir, snaps[-1]), framework="pt", device=args.device) as f:
        metadata = f.metadata()
    metadata["reconstructed_ema_gamma"] = str(args.target_gamma)
    model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(merged_sd, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash
        
    save_file(merged_sd, output_file, metadata)

    print("Done!")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct EMA weights from snapshots ")
    parser.add_argument("snapshot_dir", type=str, help="Folder with snapshots ")
    parser.add_argument("--target_gamma", type=float, help="Averaging factor. Recommended values: 5 - 50  ")  #  Lower gamma gives more weight to early steps 
    #parser.add_argument("--base_model", type=str, help="If EMA is unet-only, text encoder and vae will be copied from this model.")
    parser.add_argument("--target_sigma_rel", type=float, default = None, help="Averaging length. Alternative way of specifying gamma. Allowed values: 0 < sigma_rel < 0.28 ")
    #parser.add_argument("--target_step", type=int, default = None, help="Last step to average at ")
    parser.add_argument("--output_dir", type=str, default = None, help="Output folder ")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use, default is cpu")
    #parser.add_argument(
    #    "--precision", type=str, default="float", choices=["float", "fp16", "bf16"], help="Calculation precision, default is float"
    #)
    parser.add_argument(
        "--saving_precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="Saving precision, default is float",
    )

    args = parser.parse_args()
    reconstruct_weights_from_snapshots(args)
