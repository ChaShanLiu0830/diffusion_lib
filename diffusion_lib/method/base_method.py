from typing import Any, Dict, Optional, Tuple, Union
import torch

class BaseMethod:
    """Base class for diffusion methods providing core operations."""
    
    def __init__(self, scheduler_config: Dict[str, Any]):
        """
        BaseMethod provides core diffusion operations.

        Args:
            scheduler_config: Parameters for beta schedule, timesteps, etc.
        """
        self.scheduler_config = scheduler_config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validates the scheduler configuration."""
        required_keys = ['num_timesteps']
        for key in required_keys:
            if key not in self.scheduler_config:
                raise ValueError(f"Missing required scheduler config key: {key}")

    def q_sample(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process from clean data x0 to noisy xt.

        Args:
            x0: Original clean data.

        Returns:
            xt: Noisy version of x0.
            noise: Noise added to x0.
        """
        raise NotImplementedError("Implement forward diffusion here")

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, 
                       condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reverse denoising process from xt to a cleaner state.

        Args:
            xt: Noisy data at timestep t.
            t: Current timestep.
            condition: Optional condition input.

        Returns:
            x_t_minus_1: Denoised data at timestep t-1.
        """
        raise NotImplementedError("Implement reverse diffusion here")

    def batch_operation(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> None:
        """
        Operation performed at the start of each batch (e.g., optimal transport).

        Args:
            batch: Current batch data.
        """
        pass

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes loss between model output and target.

        Args:
            model_output: Output from the model.
            target: Target values.

        Returns:
            loss: Computed loss value.
        """
        raise NotImplementedError("Implement loss computation here") 