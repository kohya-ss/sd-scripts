# Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention

1. Download openwebtext dataset and preprocess it following instructions in [data/openwebtext/readme.md](data/openwebtext/readme.md).
2. Install dependencies:
```bash
# Create a new conda environment with Python 3.12
conda create --name llm-pretrain python==3.12

# Activate the newly created environment
conda activate llm-pretrain

# Install PyTorch, torchvision, and torchaudio with CUDA 12.4 support
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other required packages
pip install tqdm einops
```
4. Run the following command to reproduce the stable and unstable training runs:
```bash
# The batch size has been reduced to enable training on four 24GB GPUs.

# Stable training
STABLE="1" torchrun --standalone --nproc_per_node=4 train.py

# Unstable training
STABLE="0" torchrun --standalone --nproc_per_node=4 train.py
```


> **Training tips:** Start with a **large value of $\beta$** in line 88 of [attention.py](attention.py).
>
> * If you observe **NaN losses**, **decrease $\beta$** to a smaller value.
> * If the **loss explodes**, **increase $\beta$** slightly.
>
> For a detailed explanation of this tips, refer to **Section 4** of the paper.
