import argparse
from safetensors.torch import load_file, save_file

vae_conversion_map_attn = [
    # (old format, new format)
    (".q.", ".to_q."),
    (".k.", ".to_k."),
    (".v.", ".to_v."),
    (".proj_out.", ".to_out.0."),
]

def convert_vae_attn_state_dict(state_dict):
    new_state_dict = {}
    mapping = {k: k for k in state_dict.keys()}
    for k, v in mapping.items():
        # Only deal with attention layers of mid blocks
        if ".mid.attn_1." in k:
            print(f"Reverting key {k} to old format")
            for old, new in vae_conversion_map_attn:
                v = v.replace(new, old)
    new_state_dict = {v: state_dict[k] for k, v in mapping.items()}
    return new_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the safetensors file to revert.")
    parser.add_argument("--save_path", default=None, type=str, required=True, help="Path to the output checkpoint.")

    args = parser.parse_args()

    assert args.checkpoint_path is not None, "Must provide a checkpoint path!"
    assert args.save_path is not None, "Must provide a saving path!"
    
    # Load models from safetensors
    state_dict = load_file(args.checkpoint_path)

    # Convert the VAE model
    new_state_dict = convert_vae_attn_state_dict(state_dict)

    save_file(new_state_dict, args.save_path)
    