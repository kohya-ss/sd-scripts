import torch
import math
from typing import List
import bitsandbytes.functional as F
from torch.nn.functional import normalize

class Automagic_CameAMP(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-6,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=3e-6,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.8, 0.99, 0.999),
        beta1_decay=0.9995,
        eta=2,
        weight_decay=5e-4,
        warmup_steps=500,
        full_finetune=False,
    ):
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        self.full_finetune = full_finetune
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            beta1_decay=beta1_decay,
            eta=eta,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            full_finetune=full_finetune,
        )
        super().__init__(params, defaults)
        self.base_lrs: List[float] = [lr for group in self.param_groups]
        self.weight_decay = weight_decay
        self._step = 0
        self.warmup_steps = warmup_steps

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-10)

    def _get_group_lr(self, group):
        group_lrs = []
        for p in group["params"]:
            state = self.state[p]
            if 'avg_lr' in state:
                group_lrs.append(state['avg_lr'])
        return float(torch.mean(torch.tensor(group_lrs))) if group_lrs else self.lr

    # Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
    def orthograd_(self, p, grad):
        if p.norm(2) <= 1e-30:
            return grad

        G_shape = grad.shape
        w = p.view(-1)
        g = grad.view(-1)
        g_norm = g.norm(2)

        proj = torch.dot(w, g) / torch.dot(w, w).add(1e-30)
        g_orth = g.sub_(w, alpha=proj)
        g_orth_scaled = g_orth.mul_(g_norm / g_orth.norm(2).add(1e-30))

        return g_orth_scaled.view(G_shape)

    def _init_state(self, p, group=None):
        device = p.device
        shape = p.shape
        state = self.state[p]
        state.setdefault("lr_max", 1e-6)
        state.setdefault("step", 0)
        state.setdefault('lr_mask', torch.ones(shape, device=device, dtype=torch.float16) * self.lr)
        state.setdefault('avg_lr', float(self.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))
        state.setdefault("exp_avg", torch.zeros_like(p))
        state.setdefault("exp_avg_res", torch.zeros_like(p))
        state.setdefault("s", torch.zeros_like(p))
        if group['full_finetune'] == False:
            # ==== ALLoRA ====
            #ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
            #https://arxiv.org/abs/2410.09692
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group['eta']**2))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        smoothing = 0.9 
        for group in self.param_groups:
            grads_this_group = []
            for p in group["params"]:
                if p.grad is not None:
                    grads_this_group.append(p.grad.view(-1))
            if len(grads_this_group) == 0:
                continue
            all_group_grads = torch.cat(grads_this_group)
            abs_all_group_grads = torch.abs(all_group_grads)
            sum_abs_all_group_grads = torch.sum(abs_all_group_grads)
            if self._step < self.warmup_steps / 2 and self.weight_decay > 0:
                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False)
            
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                # === grad 初始化 ===
                # ==== AGR自適應梯度正則 ====
                #Adaptive Gradient Regularization: A Faster and Generalizable Optimization Technique for Deep Neural Networks
                #https://arxiv.org/pdf/2407.16944
                abs_grad = torch.abs(p.grad)
                alpha = abs_grad / sum_abs_all_group_grads
                grad = p.grad * (1 - alpha)

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1
                self._step = state["step"]
                if state["step"] == group.get("warmup_steps", 0):
                    if 's' in state:
                        del state['s']
                    if 'last_polarity' in state:
                        del state['last_polarity']

                beta1, beta2, beta3 = group["betas"]
                eps1, eps2 = group["eps"]
                clip = state['step'] ** 0.25
                grad.clamp_(-clip, clip)
                
                exp_avg , exp_avg_res = state['exp_avg'], state["exp_avg_res"]

                if state["step"] < group["warmup_steps"] / 2:
                    # ==== Torque-Aware Momentum ====
                    #https://arxiv.org/abs/2412.18790
                    #https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/tam.py
                    s = state['s']
                    beta1, beta2, beta3 = 0.9, 0.999, 0.9999
                    decay_rate = 0.9
                    corr = normalize(exp_avg, p=2.0, dim=0).mul_(normalize(grad, p=2.0, dim=0))
                    s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)
                    d = ((1.0 + s) / 2.0).add_(eps1).mul_(grad)
                    exp_avg.mul_(beta1).add_(d)
                    exp_avg_bar = exp_avg
                else:
                    #Towards Faster Training of Diffusion Models: An Inspiration of A Consistency Phenomenon
                    #https://arxiv.org/abs/2404.07946
                    #Towards Faster Training of Diffusion Models: An Inspiration of A Consistency Phenomenon
                    #https://arxiv.org/abs/2404.07946
                    beta1_t = max(beta1 * group['beta1_decay'] ** state["step"], 0.4)
                    exp_avg.mul_(beta1_t).add_(grad, alpha=1 - beta1_t)
                    exp_avg_bar = exp_avg * beta1 + grad * (1 - beta1)

                # ==== AdaBelief ====
                #AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients

                #https://arxiv.org/abs/2010.07468
                #https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/adabelief.py
                res = (grad - exp_avg_bar).pow(2) + eps2
                exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)
                update = exp_avg.clone().mul_(exp_avg_res.rsqrt())

                # === Grams ===
                #Grams: Gradient Descent with Adaptive Momentum Scaling
                #https://arxiv.org/abs/2412.17107
                #https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/grams.py
                grams_update = update.abs() * grad.sign()
                alpha = 1.0 * group['beta1_decay'] ** state["step"]
                update = alpha * grams_update + (1 - alpha) * update

                if state["step"] < group["warmup_steps"] / 2:
                    # === 正交梯度 ===
                    #Grokking at the Edge of Numerical Stability

                    #https://arxiv.org/abs/2501.04697
                    #https://github.com/LoganBooker/prodigy-plus-schedule-free/tree/dev
                    update = self.orthograd_(p, update)

                # ==== Automagic lrmask ====
                # https://github.com/ostris/ai-toolkit/blob/main/toolkit/optimizers/automagic.py

                if state["step"] < group["warmup_steps"]:
                    last_polarity = state['last_polarity']
                    current_polarity = (grad > 0)
                    sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
                    state['last_polarity'] = current_polarity
                    lr_mask = state['lr_mask']
                    new_lr = torch.where(
                        sign_agree > 0,
                        lr_mask + self.lr_bump,
                        lr_mask - self.lr_bump
                    )
                    if group["lr"] > state["lr_max"]:
                        new_lr = new_lr + (group["lr"] - state["lr_max"])
                        state["lr_max"] = group["lr"]
                    new_lr = torch.clamp(new_lr, min=self.min_lr, max=self.max_lr)
                    state['lr_mask'] = new_lr
                    state['avg_lr'] = torch.mean(new_lr).item()
                else:
                    new_lr = state['lr_mask']
                    if group["lr"] > state["lr_max"]:
                        state["lr_max"] = group["lr"]
                    if group["lr"] < state["lr_max"]:
                        new_lr = new_lr * max((group["lr"] / state["lr_max"]), 0.1)

                if "row_scaling" in state:
                    new_lr = new_lr * state["row_scaling"]

                #Mirror, Mirror of the Flow: How Does Regularization Shape Implicit Bias?
                #https://arxiv.org/abs/2504.12883
                if state["step"] < group["warmup_steps"] / 2 and group["weight_decay"] > 0:
                    #Adaptive Weight Decay for Deep Neural Networks
                    #https://arxiv.org/abs/1907.08931
                    param_abs_grad = abs_grad.mean()
                    norm_grad = (param_abs_grad - mean_norm) / std_norm
                    ada_alpha = 4
                    theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                    p.data.mul_(1 - new_lr * group["weight_decay"] * theta)
                update = update.mul(new_lr)
                p.add_(-update)

        return loss

    def state_dict(self):
        state = super().state_dict()
        state['magic_version'] = 1
        return state

    def load_state_dict(self, state_dict):
        if 'magic_version' not in state_dict or state_dict['magic_version'] != 1:
            print('[WARNING] 您載入了非預期state dict，某些動態mask參數可能未正確同步！')
        super().load_state_dict(state_dict)

class Automagic_CameAMP8bit(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-6,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=3e-6,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=2.5,
        warmup_steps=500,
        full_finetune=False,
        verbose=False,
    ):
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        self.verbose = verbose
        self.full_finetune = full_finetune

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            full_finetune=full_finetune,
        )
        super().__init__(params, defaults)
        self.base_lrs: List[float] = [lr for group in self.param_groups]

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-10)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / (exp_avg_sq_row.mean(dim=-1, keepdim=True) + 1e-12)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def _get_group_lr(self, group):
        group_lrs = []
        for p in group["params"]:
            state = self.state[p]
            if 'avg_lr' in state:
                group_lrs.append(state['avg_lr'])
        return float(torch.mean(torch.tensor(group_lrs))) if group_lrs else self.lr

    def _ratio(self, new_p, p, pre):
        curr_norm, prev_norm = torch.norm(new_p - pre), torch.norm(p - pre)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    def _init_state(self, p, group=None):
        device = p.device
        shape = p.shape
        state = self.state[p]
        state.setdefault("step", 0)

        # 1. 初始化 lr_mask 量化
        lr_mask_init = torch.ones(shape, device=device, dtype=torch.float32) * self.lr
        q_lr_mask, q_lr_mask_scale = F.quantize_blockwise(lr_mask_init, blocksize=2048)
        state.setdefault('lr_mask_q', q_lr_mask)
        state.setdefault('lr_mask_q_scale', q_lr_mask_scale)
        state.setdefault('avg_lr', float(self.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))

        # 2. 初始化 exp_avg 量化
        exp_avg_fp32 = torch.zeros_like(p)
        q_exp_avg, q_exp_avg_scale = F.quantize_blockwise(exp_avg_fp32, blocksize=2048)
        state.setdefault("exp_avg_q", q_exp_avg)
        state.setdefault("exp_avg_q_scale", q_exp_avg_scale)

        if len(shape) >= 2:
            state["exp_avg_sq_row"]  = torch.zeros(shape[:-1], device=device)
            state["exp_avg_sq_col"]  = torch.zeros(shape[:-2]+shape[-1:], device=device)
            state["exp_avg_res_row"] = torch.zeros(shape[:-1], device=device)
            state["exp_avg_res_col"] = torch.zeros(shape[:-2]+shape[-1:], device=device)
        else:
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["exp_avg_res"] = torch.zeros_like(p)
        if group is not None and group.get('full_finetune', False):
            state.setdefault("pre", p.clone())
        else:
            state.setdefault("pre", None)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                state = self.state[p]
                factored = len(p.shape) >= 2

                # === state 初始化 ===
                if len(state) == 0:
                    self._init_state(p, group)

                if 'step' not in state: state['step'] = 0
                state["step"] += 1

                beta1, beta2, beta3 = group["betas"]
                eps1, eps2 = group["eps"]

                # --- Adafactor/RMS核心部分 ---
                update = grad.pow(2) + eps1
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1 - beta2)
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1 - beta2)
                    update = grad.clone().mul_(exp_avg_sq.rsqrt())
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                # --- exp_avg (8bit) ---
                exp_avg = F.dequantize_blockwise(state["exp_avg_q"], state["exp_avg_q_scale"], blocksize=2048)
                exp_avg.mul_(beta1).add_(update, alpha=1-beta1)
                q_exp_avg, q_exp_avg_scale = F.quantize_blockwise(exp_avg, blocksize=2048)
                state["exp_avg_q"] = q_exp_avg
                state["exp_avg_q_scale"] = q_exp_avg_scale

                # ==== CAME ====
                res = (update - exp_avg).pow(2) + eps2
                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]
                    exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0-beta3)
                    exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0-beta3)
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    update = res_approx.mul(exp_avg)
                else:
                    update = exp_avg.clone()

                # === lrmask動態調節並回寫 ===
                if state["step"] <= group["warmup_steps"]:
                    last_polarity = state['last_polarity']
                    current_polarity = (grad > 0)
                    sign_agree = torch.where(last_polarity == current_polarity, 1, -1)
                    state['last_polarity'] = current_polarity
                    lr_mask = F.dequantize_blockwise(state['lr_mask_q'], state['lr_mask_q_scale'], blocksize=2048)
                    new_lr = torch.where(
                        sign_agree > 0,
                        lr_mask + self.lr_bump,
                        lr_mask - self.lr_bump
                    )
                    new_lr = torch.clamp(new_lr, min=self.min_lr, max=self.max_lr)
                else:
                    # 解量化後直接用
                    new_lr = F.dequantize_blockwise(state['lr_mask_q'], state['lr_mask_q_scale'], blocksize=2048)

                q_lr_mask, q_lr_mask_scale = F.quantize_blockwise(new_lr, blocksize=2048)
                state['lr_mask_q'] = q_lr_mask
                state['lr_mask_q_scale'] = q_lr_mask_scale
                state['avg_lr'] = torch.mean(new_lr).item()

                if group["lr"] < 1e-6:
                    new_lr = new_lr * (group["lr"] / 1e-6)

                update = update.mul(new_lr)

                # SPD
                do_spd = False
                if state["step"] <= group["warmup_steps"] and group["weight_decay"] > 0:
                    pre = state["pre"] if state["pre"] is not None else torch.zeros_like(p)
                    condition = -torch.sum(grad * (p - pre))
                    if condition < 0.0:
                        do_spd = True
                        new_p = p - update
                        ratio = self._ratio(new_p, p, pre)
                        new_p = new_p - group["weight_decay"] * ratio * (new_p - pre)
                        p.copy_(new_p)
                if not do_spd:
                    p.add_(-update)

        if self.verbose:
            print([group["lr"] for group in self.param_groups])
        return loss

    def state_dict(self):
        orig_sd = super().state_dict()
        new_state = {}
        for k, v in orig_sd['state'].items():
            # don't store unquantized lr_mask/exp_avg!
            save_state = {kk: vv for kk, vv in v.items()
                          if kk not in ('lr_mask', 'exp_avg_q', 'exp_avg_q_scale', 'lr_mask_q', 'lr_mask_q_scale')}
            # 保存 exp_avg 的量化張量+scale
            if 'exp_avg_q' in v and 'exp_avg_q_scale' in v:
                save_state['exp_avg_q'] = v['exp_avg_q']
                save_state['exp_avg_q_scale'] = v['exp_avg_q_scale']
            # 保存 lr_mask 的量化張量+scale
            if 'lr_mask_q' in v and 'lr_mask_q_scale' in v:
                save_state['lr_mask_q'] = v['lr_mask_q']
                save_state['lr_mask_q_scale'] = v['lr_mask_q_scale']
            new_state[k] = save_state
        orig_sd['state'] = new_state
        orig_sd['magic8_version'] = 1
        return orig_sd

    def load_state_dict(self, state_dict):
        if 'magic8_version' not in state_dict or state_dict['magic8_version'] != 1:
            print('[WARNING] 您載入了非預期state dict，部分8bit參數可能不同步！')
        basic_sd = {'state': {}, 'param_groups': state_dict['param_groups']}
        for k, v in state_dict['state'].items():
            basic_sd['state'][k] = {kk: vv for kk, vv in v.items()
                                    if kk not in ('exp_avg_q', 'exp_avg_q_scale', 'lr_mask_q', 'lr_mask_q_scale')}
        super().load_state_dict(basic_sd)
        # restore exp_avg_q, exp_avg_q_scale, lr_mask_q, lr_mask_q_scale
        param_map = [p for g in self.param_groups for p in g['params']]
        for idx, p in enumerate(param_map):
            idx_str = str(idx)
            if idx_str not in state_dict['state']:
                continue
            src = state_dict['state'][idx_str]
            st = self.state[p]
            if 'exp_avg_q' in src and 'exp_avg_q_scale' in src:
                st['exp_avg_q'] = src['exp_avg_q']
                st['exp_avg_q_scale'] = src['exp_avg_q_scale']
            if 'lr_mask_q' in src and 'lr_mask_q_scale' in src:
                st['lr_mask_q'] = src['lr_mask_q']
                st['lr_mask_q_scale'] = src['lr_mask_q_scale']