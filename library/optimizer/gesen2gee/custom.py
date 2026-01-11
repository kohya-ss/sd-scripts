from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def RexWithRestarts(
    optimizer: Optimizer,
    lr_start: float = 1,
    lr_end: float = 0.001,
    first_cycle: int = 500,
    num_warmup_steps: int = 0,
    last_epoch: int = -1,
) -> LambdaLR:

    def lr_lambda(current_step: int):
        if current_step <= num_warmup_steps:
            return lr_start * (current_step / num_warmup_steps)

        cycle_step = current_step % first_cycle
        cycle_progress = cycle_step / first_cycle
        
        if cycle_progress == 0:
            return lr_end

        lr = lr_end + (lr_start - lr_end) * ((1 - cycle_progress) / (1 - cycle_progress / 2))
        return lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def Rex(
    optimizer: Optimizer,
    lr_start: float = 1,
    lr_end: float = 0.001,
    total_steps: int = 500,
    num_warmup_steps: int = 0,
    last_epoch: int = -1,
) -> LambdaLR:
    
    def lr_lambda(current_step: int):
        if current_step <= num_warmup_steps:
            return lr_start * (current_step / num_warmup_steps)
        
        if current_step <= total_steps:
            cycle_progress = current_step / total_steps
            if cycle_progress == 0:
                return lr_start  # 起始點
            lr = lr_end + (lr_start - lr_end) * ((1 - cycle_progress) / (1 - cycle_progress / 2))
            return lr
        return lr_end

    return LambdaLR(optimizer, lr_lambda, last_epoch)