from typing import Any, Dict
import torch
import torch.nn.functional as F


class BaseEvaluator:
    """Base evaluator class for computing sample quality metrics."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize base evaluator.
        
        Args:
            device: Device to run evaluations on.
        """
        self.device = device

    def compute_metrics(self, samples: torch.Tensor, references: torch.Tensor) -> Dict[str, float]:
        """
        Computes evaluation metrics comparing samples and references.

        Args:
            samples: Generated samples.
            references: Ground-truth data.

        Returns:
            metrics: Calculated metrics (MSE, MAE, etc.).
        """
        metrics = {}
        
        # Ensure tensors are on the same device
        samples = samples.to(self.device)
        references = references.to(self.device)
        
        # Basic reconstruction metrics
        metrics['mse'] = self._compute_mse(samples, references)
        metrics['mae'] = self._compute_mae(samples, references)
        metrics['psnr'] = self._compute_psnr(samples, references)
        
        # Add more sophisticated metrics in subclasses
        return metrics
    
    def _compute_mse(self, samples: torch.Tensor, references: torch.Tensor) -> float:
        """
        Compute Mean Squared Error.
        
        Args:
            samples: Generated samples.
            references: Reference data.
            
        Returns:
            mse: Mean squared error value.
        """
        return F.mse_loss(samples, references).item()
    
    def _compute_mae(self, samples: torch.Tensor, references: torch.Tensor) -> float:
        """
        Compute Mean Absolute Error.
        
        Args:
            samples: Generated samples.
            references: Reference data.
            
        Returns:
            mae: Mean absolute error value.
        """
        return F.l1_loss(samples, references).item()
    
    def _compute_psnr(self, samples: torch.Tensor, references: torch.Tensor, 
                     max_val: float = 1.0) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Args:
            samples: Generated samples.
            references: Reference data.
            max_val: Maximum possible value in the data.
            
        Returns:
            psnr: PSNR value in dB.
        """
        mse = F.mse_loss(samples, references)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()
    
    def compute_diversity_metrics(self, samples: torch.Tensor) -> Dict[str, float]:
        """
        Compute diversity metrics for generated samples.
        
        Args:
            samples: Generated samples.
            
        Returns:
            diversity_metrics: Dictionary of diversity metrics.
        """
        metrics = {}
        
        # Flatten samples for pairwise comparisons
        batch_size = samples.shape[0]
        flattened = samples.view(batch_size, -1)
        
        # Compute pairwise distances
        pairwise_distances = torch.cdist(flattened, flattened, p=2)
        
        # Remove diagonal (self-distances)
        mask = ~torch.eye(batch_size, dtype=bool, device=samples.device)
        distances = pairwise_distances[mask]
        
        metrics['mean_pairwise_distance'] = distances.mean().item()
        metrics['std_pairwise_distance'] = distances.std().item()
        metrics['min_pairwise_distance'] = distances.min().item()
        
        return metrics
    
    def compute_quality_metrics(self, samples: torch.Tensor) -> Dict[str, float]:
        """
        Compute quality metrics for generated samples.
        
        Args:
            samples: Generated samples.
            
        Returns:
            quality_metrics: Dictionary of quality metrics.
        """
        metrics = {}
        
        # Basic statistics
        metrics['mean'] = samples.mean().item()
        metrics['std'] = samples.std().item()
        metrics['min'] = samples.min().item()
        metrics['max'] = samples.max().item()
        
        # Value range metrics
        metrics['dynamic_range'] = (samples.max() - samples.min()).item()
        
        return metrics 