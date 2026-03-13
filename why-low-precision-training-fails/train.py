"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "3600"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
# random.seed(0)
from datetime import timedelta
import torch
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
# torch.random.manual_seed(0)

import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
# np.random.seed(0)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F

from model import GPTConfig
from model import GPT

# torch.autograd.set_detect_anomaly(True)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
log_max_logits = False
log_mean_logits = False
log_adam_m_v = False

initial_seed = 42
out_dir = f'GPT2-S_1e-3_0.0_2000_BF16_with_FA_{"stable" if os.environ.get("STABLE", "0") == "1" else "unstable"}' # output directory
eval_interval = 1000
log_interval = 1000
eval_iters = 1000
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False 
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 16
batch_size = 20
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-3 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
# backend = 'nccl' # 'nccl', 'gloo', etc.
backend = "nccl"
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# dtype = 'float32'
compile = True # use PyTorch 2.0 to compile the model to be faster
start_idx = 38000
ckpt_path = "GPT2-S_1e-3_0.0_2000_BF16_with_FA_0917-v2/opt_state/model_iter_70000.pt"
opt_path = "GPT2-S_1e-3_0.0_2000_BF16_with_FA_0917-v2/opt_state/optimizer_iter_70000.pt"
indices_path = 'GPT2-S_1e-3_0.1_0_BF16_with_FA'
reproduce_instability = False # if True, will use the provided checkpoint and indices to reproduce the instability
use_flash = True # use flash attention if available,
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    # init_process_group(backend="nccl" if torch.distributed.is_nccl_available() else "gloo", timeout=timedelta(seconds=6000))

    print(f"Running in DDP mode with backend {backend}")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    print(f'master_process: {master_process}, ddp_rank: {ddp_rank}\n')
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    ddp_rank = 0
    ddp_local_rank = 0
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/opt_state', exist_ok=True)
# torch.manual_seed(initial_seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = False # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = False # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_indices():
    current_rank_indices_path = ''
    if ddp:
        current_rank_indices_path = os.path.join(indices_path, f'generated_indices_rank_{ddp_rank}.txt')
    with open(current_rank_indices_path, 'r') as f:
        logged_indices_for_this_rank = [list(map(int, line.strip().split(','))) for line in f]#
    return logged_indices_for_this_rank

## Debugging setup (optional)
# import debugpy
# debugpy.listen(5678 + ddp_rank)
# print(f"Waiting for debugger on rank {ddp_rank}")
# debugpy.wait_for_client()

# poor man's data loader
data_dir = os.path.join('data', dataset)

# --- Conditional Index Loading or Preparation for Saving ---
logged_indices = None
indices_file_to_save_handle = None
generated_indices_file_path = ""
train_data_total_tokens = 0 # Will be set if generating indices

if reproduce_instability:
    if not indices_path or not os.path.exists(indices_path):
        raise FileNotFoundError(
            f"reproduce_instability is True, but indices_path ('{indices_path}') is not provided or not found."
        )
    # load_indices function is defined below or already exists
    logged_indices = load_indices() # load_indices should handle DDP distribution internally
    if master_process:
        # The count here should reflect total effective batches across all ranks if possible,
        # or clearly state it's per-rank from load_indices's own print.
        # load_indices already prints per-rank info.
        print(f"Running in REPRODUCE mode. Indices loaded from '{indices_path}'.")
else: # Not reproducing, so we will generate indices
    try:
        train_data_map_for_len = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        train_data_total_tokens = len(train_data_map_for_len)
        del train_data_map_for_len
        if train_data_total_tokens <= block_size :
             raise ValueError(f"block_size ({block_size}) is too large for the training data size ({train_data_total_tokens}).")
    except FileNotFoundError:
        raise FileNotFoundError(f"Training data file not found at {os.path.join(data_dir, 'train.bin')}. Needed for generating indices.")

    generated_indices_file_path = os.path.join(out_dir, f'generated_indices_rank_{ddp_rank}.txt')
    indices_file_to_save_handle = open(generated_indices_file_path, 'w')
    print(f'Rank {ddp_rank} will generate indices and save to {generated_indices_file_path}')
    
loss_file_path = os.path.join(out_dir, 'loss.csv')
if master_process:
    loss_file_handle = open(loss_file_path, 'w')
    loss_file_handle.write("step,train_loss,val_loss\n")
    print(f'Master process will log losses to {loss_file_path}')

def load_data_for_micro_batch(indices_for_micro_batch):
    # Accesses global: data_dir, block_size, device_type, device, ptdtype
    # Ensure data_map is created fresh or handled carefully if memmap is an issue
    data_map = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    
    # Ensure indices_for_micro_batch is a list or 1D tensor of Python ints or longs
    if isinstance(indices_for_micro_batch, torch.Tensor):
        ix_tensor = indices_for_micro_batch.to(dtype=torch.int64)
    else:
        ix_tensor = torch.tensor(indices_for_micro_batch, dtype=torch.int64)

    x_list = []
    y_list = []
    for i_val_tensor in ix_tensor:
        i_val = i_val_tensor.item() # Get Python int from 0-dim tensor
        x_list.append(torch.from_numpy((data_map[i_val : i_val + block_size]).astype(np.int64)))
        y_list.append(torch.from_numpy((data_map[i_val + 1 : i_val + 1 + block_size]).astype(np.int64)))
    
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'val':
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        # batch_size here is the micro-batch size for evaluation
        ix = torch.randint(len(data) - block_size, (16,))
    elif split == 'train': # This path is now for estimate_loss, should use random sampling
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (16,)) # micro-batch size
    else:
        raise ValueError(f"Unknown split: {split}")

    # Common logic for constructing x and y from ix
    x_list = []
    y_list = []
    for i_val_tensor in ix: # ix is a tensor of random indices
        i_val = i_val_tensor.item()
        x_list.append(torch.from_numpy((data[i_val : i_val + block_size]).astype(np.int64)))
        y_list.append(torch.from_numpy((data[i_val + 1 : i_val + 1 + block_size]).astype(np.int64)))
    
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, use_flash=use_flash) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {ckpt_path}")
    # resume training from a checkpoint.
    state_dict = torch.load(ckpt_path, map_location=device)
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    iter_num = 0
    best_val_loss = np.inf
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"Unknown init_from value: {init_from}")


model_args['use_flash'] = use_flash # use flash attention if available, regardless of init_from

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    print(f'Resume optimizer from {opt_path}...')
    opt_checkpoint = torch.load(opt_path, map_location=device)
    optimizer.load_state_dict(opt_checkpoint)
state_dict = None # free up memory after loading, if we loaded one
opt_checkpoint = None

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
# REMOVE: X, Y = get_batch('train') # fetch the very first batch (no longer needed here)
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
norm = 0
diff_sum_list = []

# Extract r[G, t]
if log_max_logits:
    attn_layer_text = [0, len(raw_model.transformer.h)-1]
    attn_logits_collected_layers = [raw_model.transformer.h[0], raw_model.transformer.h[-1]]
    # only log first layer
    for layer in attn_logits_collected_layers:
        layer.attn.log_max_logits = True
        layer.attn.register_buffer("bias", torch.tril(torch.ones(gptconf.block_size, gptconf.block_size, device=device))
                                            .view(1, 1, gptconf.block_size, gptconf.block_size))
if log_adam_m_v:
    collected_layers = [raw_model.transformer.wte, raw_model.transformer.h[0], raw_model.transformer.h[1], raw_model.transformer.h[2], raw_model.transformer.h[3]]
    r_values = {}

torch.cuda.empty_cache()

while True:
    # torch.cuda.empty_cache()
    # pos_scale = 4.5 - (4.5 - 2) * (iter_num / max_iters) # Linearly decay pos_scale from 4.5 to 2 over training

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if iter_num == 0 and eval_only:
        break
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    if reproduce_instability:
        if iter_num >= len(logged_indices): 
            if master_process:
                print(f"Rank {ddp_local_rank if ddp else 0} ran out of precomputed effective batch indices from '{indices_path}' at iter_num {iter_num}. "
                      f"Available effective batches for this rank: {len(logged_indices)}. Stopping.")
            break
        current_effective_batch_indices_list = logged_indices[iter_num]
    else: # Generate indices for this effective batch (reproduce_instability = False)
        num_indices_for_effective_batch = batch_size * gradient_accumulation_steps
        # Each rank generates its own sequence based on its unique seed.
        # Master process (rank 0) saves its generated sequence.
        current_effective_batch_indices_list = torch.randint(
            0, train_data_total_tokens - block_size, (num_indices_for_effective_batch,)
        ).tolist()

        if indices_file_to_save_handle:
            indices_str = ",".join(map(str, current_effective_batch_indices_list))
            indices_file_to_save_handle.write(indices_str + "\n")
            # if iter_num > 0 and iter_num % 1000 == 0: 
            indices_file_to_save_handle.flush()
    for micro_step in range(gradient_accumulation_steps):
        start_idx_in_effective_batch = micro_step * batch_size
        end_idx_in_effective_batch = start_idx_in_effective_batch + batch_size

        # Ensure slicing is within bounds of current_effective_batch_indices_list
        # This should be guaranteed if the previous length check passed.
        indices_for_this_micro_batch = current_effective_batch_indices_list[start_idx_in_effective_batch:end_idx_in_effective_batch]

        X, Y = load_data_for_micro_batch(indices_for_this_micro_batch)
        # print(f"DDP {ddp_local_rank if ddp else 0}, iter {iter_num}, micro_step {micro_step}, X: {X}, Y: {Y}")
        # exit()

        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            # logits, loss = model(X, Y, save_ckpt=(iter_num % log_interval == 0))
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    print(f"iter {iter_num}, loss {loss.item()}")
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if (iter_num % log_interval == 0) and master_process:
        torch.save(raw_model.state_dict(), f"{out_dir}/opt_state/model_iter_{iter_num}.pt")

    scaler.step(optimizer)
    scaler.update()

    if (iter_num % log_interval == 0) and master_process:
        # save optimizer state to file
        torch.save(optimizer.state_dict(), f"{out_dir}/opt_state/optimizer_iter_{iter_num}.pt")
    optimizer.zero_grad(set_to_none=True)
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    #
    if (iter_num % log_interval == 0) and master_process:
        # print(f'iter {iter_num}')
        print('starting logging ')
        lossf = loss.item() * gradient_accumulation_steps
        log_text = f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, grad norm {norm:.2f}, lr {lr:.4f}"
        if log_max_logits and use_flash:
            log_text += f", attention logits layer {' '.join(str(x) for x in attn_layer_text)}"
            log_text += f', max'
            for attn_layer in attn_logits_collected_layers:
                log_text += f', {attn_layer.attn.max_attention_logit.item():.2f}' 
            log_text += f', min'
            for attn_layer in attn_logits_collected_layers:
                log_text += f', {attn_layer.attn.min_attention_logits.item():.2f}' 
            log_text += f', mean'
            for attn_layer in attn_logits_collected_layers:
                log_text += f', {attn_layer.attn.mean_attention_logits.item():.2f}' 
            log_text += f', entropy'
            for attn_layer in attn_logits_collected_layers:
                log_text += f', {attn_layer.attn.entropy.item():.2f}' 

        if log_mean_logits:
            log_text += f', mean output logits {logits.mean().item():.2f}'
            log_text += f', min output logits {logits.min().item():.2f}'
            log_text += f', max output logits {logits.max().item():.2f}'
                    
        print(log_text)

    if (iter_num % eval_interval == 0) and master_process:
        losses = estimate_loss()
        # torch.cuda.empty_cache()
        # print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if loss_file_handle:
            loss_file_handle.write(f"{iter_num},{losses['train']},{losses['val']}\n")
            loss_file_handle.flush()
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "grad_norm": norm
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
        print('finished validation\n')
    iter_num += 1
    local_iter_num += 1
    if iter_num >= max_iters: 
        break
    if not reproduce_instability and indices_file_to_save_handle is None:
        # If not reproducing instability and indices file handle is missing, stop training
        if master_process:
            print("No indices file handle for saving generated indices. Stopping training.")
        break

if not reproduce_instability and indices_file_to_save_handle:
    indices_file_to_save_handle.close()
    print(f"Indices saved to {generated_indices_file_path} for rank {ddp_rank}")
if loss_file_handle and master_process:
    loss_file_handle.close()
    print(f"Losses saved to {loss_file_path}")
if ddp:
    destroy_process_group()
