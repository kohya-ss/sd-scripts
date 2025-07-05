import torch
from library.avg_ckpt_util import average_state_dicts

def test_uniform_average():
    sd1 = {'lora_a': torch.tensor(1.0), 'lora_b': torch.tensor(3.0)}
    sd2 = {'lora_a': torch.tensor(3.0), 'lora_b': torch.tensor(5.0)}
    avg = average_state_dicts([sd1, sd2], mode='uniform')
    assert torch.allclose(avg['lora_a'], torch.tensor(2.0))
    assert torch.allclose(avg['lora_b'], torch.tensor(4.0))
