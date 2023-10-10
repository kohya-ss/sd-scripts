import argparse
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def split(args):
    # load embedding
    if args.embedding.endswith(".safetensors"):
        embedding = load_file(args.embedding)
        with safe_open(args.embedding, framework="pt") as f:
            metadata = f.metadata()
    else:
        embedding = torch.load(args.embedding)
        metadata = None

    # check format
    if "emb_params" in embedding:
        # SD1/2
        keys = ["emb_params"]
    elif "clip_l" in embedding:
        # SDXL
        keys = ["clip_l", "clip_g"]
    else:
        print("Unknown embedding format")
        exit()
    num_vectors = embedding[keys[0]].shape[0]

    # prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare splits
    if args.vectors_per_split is not None:
        num_splits = (num_vectors + args.vectors_per_split - 1) // args.vectors_per_split
        vectors_for_split = [args.vectors_per_split] * num_splits
        if sum(vectors_for_split) > num_vectors:
            vectors_for_split[-1] -= sum(vectors_for_split) - num_vectors
            assert sum(vectors_for_split) == num_vectors
    elif args.vectors is not None:
        vectors_for_split = args.vectors
        num_splits = len(vectors_for_split)
    else:
        print("Must specify either --vectors_per_split or --vectors / --vectors_per_split または --vectors のどちらかを指定する必要があります")
        exit()

    assert (
        sum(vectors_for_split) == num_vectors
    ), "Sum of vectors must be equal to the number of vectors in the embedding / 分割したベクトルの合計はembeddingのベクトル数と等しくなければなりません"

    # split
    basename = os.path.splitext(os.path.basename(args.embedding))[0]
    done_vectors = 0
    for i, num_vectors in enumerate(vectors_for_split):
        print(f"Splitting {num_vectors} vectors...")

        split_embedding = {}
        for key in keys:
            split_embedding[key] = embedding[key][done_vectors : done_vectors + num_vectors]

        output_file = os.path.join(args.output_dir, f"{basename}_{i}.safetensors")
        save_file(split_embedding, output_file, metadata)
        print(f"Saved to {output_file}")

        done_vectors += num_vectors

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge models")
    parser.add_argument("--embedding", type=str, help="Embedding to split")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--vectors_per_split",
        type=int,
        default=None,
        help="Number of vectors per split. If num_vectors is 8 and vectors_per_split is 3, then 3, 3, 2 vectors will be split",
    )
    parser.add_argument("--vectors", type=int, default=None, nargs="*", help="number of vectors for each split. e.g. 3 3 2")
    args = parser.parse_args()
    split(args)
