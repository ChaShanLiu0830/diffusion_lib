from typing import List, Union, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class BaseSampler:
    def __init__(self, model: nn.Module, method:Any, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model = model
        self.method = method
        self.device = device
    
    @torch.inference_mode()
    def sample_batch(self, num_samples: int, dataloader: DataLoader, steps: int = 1) -> torch.Tensor:
        all_samples = []
        for i in range(num_samples):
            samples = []
            for batch in dataloader:
                batch = self.method.batch_operation(batch)
                x_0 = self.method.get_target(batch)
                x_t = self.method.get_noise(x_0.shape)
                condition = self.method.get_condition(batch)
                
                for i in reversed(range(0, self.method.scheduler.num_timesteps, steps)):
                    t = torch.full((x_t.size(0),), i, device=self.device, dtype=torch.long)
                    eps_pred = self.model(x_t, t, **condition)
                    x_t = self.method.p_sample(x_t, t, eps_pred)
                
                samples.append(x_t.detach().cpu())
            all_samples.append(torch.cat(samples, dim=0))
        return torch.stack(all_samples, dim=-1)
        