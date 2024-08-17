import math
from typing import Iterable, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import Replicate


class Adam_mini(torch.optim.Optimizer):
    def __init__(
            self,
            named_parameters: Iterable[Tuple[str, nn.Parameter]],
            lr: Union[float, torch.Tensor] = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            *,
            model_sharding: bool = True,
            dim: int = 2048,
            n_heads: int = 32,
            n_kv_heads: Optional[int] = None,
    ):
        '''
        named_parameters: model.named_parameters()

        lr: learning rate

        betas: same betas as Adam

        eps: same eps as Adam

        weight_decay: weight_decay coefficient

        model_sharding: set to True if you are using model parallelism with more than 1 GPU, including FSDP and zero_1,2,3 in Deepspeed. Set to False if otherwise.

        dim: dimension for hidden feature.  Could be unspecified if you are training non-transformer models.

        n_heads: number of attention heads. Could be unspecified if you are training non-transformer models.

        n_kv_heads: number of head for Key and Value. Or equivalently, number of query groups in Group query Attention. Also known as "n_query_groups".  If not specified, it will be equal to n_head. Could be unspecified if you are training non-transformer models.
        '''
        self.dim = dim
        self.n_heads = n_heads
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, f"{n_heads} {n_kv_heads}"
            self.n_kv_heads = n_kv_heads
        else:
            self.n_kv_heads = n_heads

        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.model_sharding = model_sharding
        if self.model_sharding:
            print("=====>>> Adam-mini is using model_sharding")

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not self.dim == int(self.dim):
            raise ValueError("Invalid dim value: {}".format(self.dim))
        if not self.n_heads == int(self.n_heads):
            raise ValueError("Invalid n_heads value: {}".format(self.n_heads))
        if not self.n_kv_heads == int(self.n_kv_heads):
            raise ValueError("Invalid n_kv_heads value: {}".format(self.n_kv_heads))

        optim_groups = []
        count_embd = count_output = count_wq = count_wk = 0
        for param_name, param in named_parameters:
            if not param.requires_grad:
                continue
            print('Adam-mini found the param block with name:', param_name)
            state = {}
            state["name"] = param_name
            state["params"] = param
            if "norm" in param_name or "ln_f" in param_name:
                state["weight_decay"] = 0.0
            else:
                state["weight_decay"] = weight_decay
            if "embed" in param_name or "wte" in param_name or "embd" in param_name:
                count_embd += 1
            if "lm_head.weight" in param_name or "output.weight" in param_name:
                count_output += 1
            if "q_proj.weight" in param_name or "wq.weight" in param_name or "attn_qkv.lora_down" in param_name or "attn_proj.lora_down" in param_name:
                count_wq += 1
                assert (self.dim * self.dim) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
                state["head_numel"] = self.dim * self.dim // self.n_heads
            if "k_proj.weight" in param_name or "wk.weight" in param_name or "attn_qkv.lora_up" in param_name or "attn_proj.lora_up" in param_name or "mlp" in param_name:
                count_wk += 1
                assert (self.dim * self.dim) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
                state["head_numel"] = self.dim * self.dim // self.n_heads
            optim_groups.append(state)

        print(
            f'Adam-mini found {count_embd} embedding layers, {count_output} output layers, {count_wq} Querys, {count_wk} Keys.')

        if count_embd == 0:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No embedding layer found. If you are training Transformers, please check the name of your embedding layer and manually add them to 'self.embd_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.embd_names.add('the name of your embedding layer'). ")
        if count_output == 0:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No output layer found.  If you are training Transformers (without weight-tying), please check the name of your output layer and manually add them to 'self.embd_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.embd_names.add('the name of your output layer').  Please ignore this warning if you are using weight-tying.")
        if count_wq == 0:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Query found.  If you are training Transformers, please check the name of your Query in attention blocks and manually add them to 'self.wqk_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.wqk_names.add('the name of your Query'). ")

        if count_wk == 0:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Key found.  If you are training Transformers, please check the name of your Key in attention blocks and manually add them to 'self.wqk_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.wqk_names.add('the name of your Key').")

        if count_output + count_embd + count_wq + count_wk == 0:
            print(
                "=====>>>  Warning by Adam-mini: you are using default PyTorch partition for Adam-mini. It can cause training instability on large-scale Transformers.")

        # embd_blocks, including embd and output layers. Use normal adamW updates for these blocks
        self.embd_names = {"embed", "embd", "wte", "lm_head.weight", "output.weight"}
        # Query and Keys, will  assign lrs by heads
        self.wqk_names = {"k_proj.weight", "q_proj.weight", "wq.weight", "wk.weight"}

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps)
        super().__init__(optim_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lr = group["lr"]
            name = group["name"]
            eps = group["eps"]

            for p in group["params"]:

                state = self.state[p]
                if any(embd_name in name for embd_name in self.embd_names):  # this is for embedding and output layer
                    if p.grad is None:
                        continue
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p, dtype=torch.float32)
                        state["step"] = 0
                        state["v"] = torch.zeros_like(p, dtype=torch.float32)

                    grad = p.grad.to(torch.float32)
                    state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = lr / bias_correction_1
                    p.addcdiv_(state["m"], h, value=-stepsize)
                elif any(wqk_name in name for wqk_name in self.wqk_names):  # this is for query and key
                    if p.grad is None:
                        continue
                    head_numel = group["head_numel"]
                    if len(state) == 0:
                        m = torch.zeros_like(p, dtype=torch.float32)
                        state["m"] = m.view(-1, head_numel)
                        state["head"] = state["m"].size(0)
                        state["step"] = 0
                        # NOTE: We must use `zeros_like` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # state["vmean"] = torch.zeros(state["head"])
                        state["vmean"] = torch.zeros_like(state["m"][0:state["head"], 0:1])

                    grad = p.grad.to(torch.float32)
                    head = state["head"]
                    grad = grad.view(head, head_numel)
                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(head, 1)
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)
                else:  # other blocks
                    if len(state) == 0:
                        block_numel = torch.tensor(p.numel(), dtype=torch.float32, device=p.device)
                        reduced = False
                        if (self.world_size > 1) and (self.model_sharding is True):
                            tensor_list = [torch.zeros_like(block_numel) for _ in range(self.world_size)]

                            dist.all_gather(tensor_list, block_numel)
                            s = 0
                            block_numel = 0
                            for d in tensor_list:
                                if (d > 0):
                                    s = s + 1
                                block_numel = block_numel + d
                            if (s >= 2):
                                reduced = True

                        state["m"] = torch.zeros_like(p, dtype=torch.float32)
                        state["step"] = 0
                        state["reduced"] = reduced
                        # NOTE: We must use `new_zeros` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # state["vmean"] = torch.zeros(1, device=p.device)
                        # state["vmean"] = p.new_zeros(1)
                        state["vmean"] = torch.zeros_like(torch.sum(p * p))
                        state["block_numel"] = block_numel.item()
                    if p.grad is None:
                        tmp_lr = torch.zeros_like(torch.sum(p * p))
                    else:
                        grad = p.grad.to(torch.float32)
                        tmp_lr = torch.sum(grad * grad)

                    if (state["reduced"]):
                        if "device_mesh" in dir(tmp_lr):
                            # when tmp_lr is a  DTensor in TorchTitan
                            lr_local = tmp_lr.to_local()
                            dist.all_reduce(lr_local, op=dist.ReduceOp.SUM)
                            tmp_lr.redistribute(placements=[Replicate()])
                        else:
                            # when tmp_lr is a  standard tensor
                            # print(f"...... dist all reduce.......")
                            dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

                    if (p.grad is None):
                        continue
                    tmp_lr = tmp_lr / state["block_numel"]

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["step"] += 1
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = (1 / bias_correction_1) / h
                    update = state["m"] * (stepsize.to(state["m"].device))
                    update.mul_(lr)
                    p.add_(-update)
        return loss