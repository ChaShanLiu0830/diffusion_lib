from typing import List, Union, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..manager.file_manager import FileManager


class BaseSampler:
    """
    Base sampler class for generating samples using trained diffusion models.
    
    This class follows SOLID principles by separating sampling logic from
    model loading and file management concerns.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 method: Any, 
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 config: Optional[Dict[str, Any]] = None,
                 load_best_model: bool = True):
        """
        Initialize the BaseSampler.
        
        Args:
            model: PyTorch diffusion model for sampling
            method: Sampling method containing scheduler and noise functions
            device: Device to run sampling on
            config: Configuration dictionary for file management
            load_best_model: Whether to automatically load the best saved model
        """
        self.model = model.to(device)
        self.method = method
        self.device = device
        self.file_manager = FileManager(config=config if config is not None else {})
        
        # Load best model if requested and available
        if load_best_model:
            self._load_best_model()
    
    def _load_best_model(self) -> None:
        """
        Load the best saved model if available.
        
        Raises:
            RuntimeWarning: If no best model is found but continues with current model
        """
        try:
            checkpoint_info = self.file_manager.load_best_model(
                model=self.model,
                device=self.device
            )
            print(f"Loaded best model from epoch {checkpoint_info['epoch']} "
                  f"with validation loss {checkpoint_info['best_val_loss']:.4f}")
        except FileNotFoundError:
            print("Warning: No best model found. Using current model state for sampling.")
    
    def load_specific_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a specific checkpoint for sampling.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dict containing checkpoint information
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf'))
        }
        
        print(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")
        return checkpoint_info
    
    @torch.inference_mode()
    def sample_batch(self, 
                     num_samples: int, 
                     dataloader: DataLoader, 
                     steps: int = 1,
                     return_intermediate: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate samples using the loaded model.
        
        Args:
            num_samples: Number of sample sets to generate
            dataloader: DataLoader providing conditioning information
            steps: Step size for sampling (default: 1 for full sampling)
            return_intermediate: Whether to return intermediate sampling steps
            
        Returns:
            Generated samples as torch.Tensor, or dict with samples and intermediates
            if return_intermediate is True
        """
        self.model.eval()
        
        all_samples = []
        all_intermediates = [] if return_intermediate else None
        
        print(f"Generating {num_samples} sample sets...")
        
        for sample_idx in tqdm(range(num_samples), desc="Sampling"):
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
        