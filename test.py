from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.optim as optim

# 初始化 Accelerator
accelerator = Accelerator(mixed_precision="bf16")

# 創建一個簡單的模型和數據
model = nn.Linear(10, 1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 準備數據
data = torch.randn(5, 10)
target = torch.randn(5, 1)

weight_type = torch.bfloat16
model = model.to(accelerator.device, dtype=weight_type)
data = data.to(accelerator.device, dtype=weight_type)
target = target.to(accelerator.device, dtype=weight_type)


# 使用 accelerator.prepare 函數
model, optimizer, data, target = accelerator.prepare(model, optimizer, data, target)

# 前向傳播和反向傳播
with accelerator.autocast():
    output = model(data)
    loss = loss_fn(output, target)
print(loss.dtype, output.dtype)
optimizer.zero_grad()
accelerator.backward(loss)
optimizer.step()