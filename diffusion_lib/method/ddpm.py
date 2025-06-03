from typing import Any, Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from typing import Union
from ..utils.func import repeat

class DDPMScheduler:
    def __init__(self,
                 num_timesteps: int,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 device: torch.device = torch.device("cpu")):
        """
        DDPM noise scheduler following the original paper.
        
        Args:
            num_timesteps: Number of diffusion timesteps.
            beta_start: Starting value for beta schedule.
            beta_end: Ending value for beta schedule.
            device: Device to store tensors on.
        """
        self.device = device
        self.num_timesteps = num_timesteps
        
        # Create linear beta schedule
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
        self.betas = torch.tensor(betas, device=device)
        
        # Compute alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute useful quantities
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)


# ----------------- Diffusion model ---------------------------
class DDPM(nn.Module):
    def __init__(self, scheduler: DDPMScheduler, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        DDPM diffusion method implementation.
        
        Args:
            scheduler: DDPMScheduler instance containing noise schedules.
        """
        super().__init__()
        self.scheduler = scheduler
        self.device = device
    # ----- forward q(x_t | x_0) -----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_bar_t  = repeat(self.scheduler.sqrt_alphas_cumprod[t], x0)
        one_minus_t  = repeat(self.scheduler.sqrt_one_minus_cumprod[t], x0)
        eps          = torch.randn_like(x0)
        x_t          = alpha_bar_t * x0 + one_minus_t * eps
        return x_t, eps

    # ----- reverse p(x_{t-1} | x_t) (stochastic) -----
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor,
                 eps_pred: torch.Tensor) -> torch.Tensor:
        betas_t    = repeat(self.scheduler.betas[t],  x_t)
        alphas_t   = repeat(self.scheduler.alphas[t], x_t)
        sqrt_one_t = repeat(self.scheduler.sqrt_one_minus_cumprod[t], x_t)

        coef_x0  = 1. / torch.sqrt(alphas_t)
        coef_eps = betas_t / sqrt_one_t
        x0_pred  = coef_x0 * (x_t - coef_eps * eps_pred)

        noise    = torch.randn_like(x_t)
        x_prev   = x0_pred + torch.sqrt(betas_t) * noise
        return x_prev

    # ----- DDIM deterministic / stochastic step -----
    def ddim_step(self, x_t: torch.Tensor, t: torch.Tensor,
                  eps_pred: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        alpha_bar_t    = repeat(self.scheduler.alphas_cumprod[t],   x_t)
        alpha_bar_prev = repeat(self.scheduler.alphas_cumprod[t-1], x_t)
        alpha_t = repeat(self.scheduler.alphas[t], x_t)
        
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                  torch.sqrt(1 - alpha_bar_t / alpha_t)
        # print(alpha_bar_prev.shape, x_t.shape, eps_pred.shape, alpha_bar_t.shape, x0_pred.shape, sigma_t.shape)     
        
        noise   = torch.randn_like(x_t)
        x_prev  = (torch.sqrt(alpha_bar_prev) * x0_pred
                   + torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_pred
                   + sigma_t * noise)
        return x_prev
    
    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target noise.
        
        Args:
            model_output: Predicted noise from the model.
            target: Target noise.
            
        Returns:
            loss: MSE loss.
        """
        return F.mse_loss(model_output, target)
    
    def batch_operation(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process batch data (placeholder for any batch-level operations).
        
        Args:
            batch: Input batch.
            
        Returns:
            batch: Processed batch (unchanged in base implementation).
        """
        return batch.to(self.device)
    
    def get_target(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        return batch.to(self.device)
    
    def get_noise(self, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape).to(self.device)
    
    def get_condition(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        return {}
    
    