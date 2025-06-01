from typing import Any, Optional, Union, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..utils.func import repeat
import torch.nn.functional as F


class BaseTrainer:
    def __init__(self, model: nn.Module, method: Any,  device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), optimizer: torch.optim.Optimizer = None, **kwargs):
        self.model = model
        self.method = method
        self.device = device
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
    def train(self, dataloader: DataLoader, epochs: int = 100):
        losses = []
        for epoch in range(epochs):
            tot_loss = 0.0
            for batch in dataloader:
                x_0 = batch.to(self.device)
                t  = torch.randint(0, self.method.scheduler.num_timesteps, (x_0.size(0),),
                                   device=self.device, dtype=torch.long)
                x_t, eps = self.method.q_sample(x_0, t)
                eps_pred = self.model(x_t, t)
                loss     = F.mse_loss(eps_pred, eps)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                tot_loss += loss.item()
            losses.append(tot_loss / len(dataloader))
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d} | loss {losses[-1]:.4f}")
        return losses
    
        
        
        
# class BaseTrainer:
#     """Base trainer class encapsulating the main training logic for diffusion models."""
    
#     def __init__(self, model: nn.Module, dataset: DataLoader, method: Any, 
#                  optimizer: torch.optim.Optimizer, logger: Optional[Any] = None, 
#                  evaluator: Optional[Any] = None, device: str = "cuda"):
#         """
#         BaseTrainer encapsulates the main training logic for diffusion models.

#         Args:
#             model: Neural network to train (e.g., UNet).
#             dataset: DataLoader yielding batches (Tensor or Dict).
#             method: Diffusion method with forward/backward processes.
#             optimizer: Optimizer for training the model.
#             logger: Optional logger to track training progress.
#             evaluator: Optional evaluator for sample quality.
#             device: Device to run training on.
#         """
#         self.model = model
#         self.dataset = dataset
#         self.method = method
#         self.optimizer = optimizer
#         self.logger = logger
#         self.evaluator = evaluator
#         self.device = device
        
#         # Move model to device
#         self.model.to(self.device)
        
#         # Set model in method if it has set_model method
#         if hasattr(self.method, 'set_model'):
#             self.method.set_model(self.model)
        
#         self.global_step = 0

#     def train(self, num_epochs: int) -> None:
#         """
#         Runs the main training loop.

#         Args:
#             num_epochs: Number of epochs to train.
#         """
#         self.model.train()
        
#         for epoch in range(num_epochs):
#             epoch_loss = 0.0
#             num_batches = 0
            
#             for batch_idx, batch in enumerate(self.dataset):
#                 # Perform batch operation if needed
#                 self.method.batch_operation(batch)
                
#                 # Training step
#                 loss = self._train_step(batch)
#                 epoch_loss += loss
#                 num_batches += 1
                
#                 # Log step-level metrics
#                 if self.logger and batch_idx % 100 == 0:
#                     self.logger.log({"loss": loss, "epoch": epoch, "batch": batch_idx}, 
#                                    step=self.global_step)
                
#                 self.global_step += 1
            
#             # Log epoch-level metrics
#             avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
#             if self.logger:
#                 self.logger.log({"epoch_loss": avg_epoch_loss, "epoch": epoch}, 
#                                step=epoch)
            
#             print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}")
            
#             # Evaluation if evaluator is provided
#             if self.evaluator and (epoch + 1) % 100 == 0:
#                 self._evaluate(epoch)

#     def _train_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> float:
#         """
#         Performs one optimization step and updates the model.

#         Args:
#             batch: A single batch of data.

#         Returns:
#             loss: Calculated loss for the current batch.
#         """
        
#         x0 = self.method.get_target(batch)
#         # print(x0)
#         # Zero gradients
#         self.optimizer.zero_grad()
        
#         # Sample random timesteps
#         t = torch.randint(0, self.method.scheduler.num_timesteps, (x0.shape[0],), 
#                          device=self.device, dtype=torch.long)
        
#         # Forward diffusion process - get noised data and target noise
#         x_t, noise = self.method.q_sample(x0, t)
        
#         predicted_noise = self.model(x_t, t)
        
#         # Compute loss between predicted and actual noise
#         loss = self.method.compute_loss(predicted_noise, noise)
        
#         # Backward pass
#         loss.backward()
        
#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
#         # Optimizer step
#         self.optimizer.step()
        
#         return loss.item()
    
#     def _evaluate(self, epoch: int) -> None:
#         """
#         Performs evaluation using the evaluator.
        
#         Args:
#             epoch: Current epoch number.
#         """
#         if self.evaluator is None:
#             return
            
#         self.model.eval()
#         with torch.no_grad():
#             # Generate samples for evaluation
#             sample_batch = next(iter(self.dataset))
#             x0 = self.method.get_target(sample_batch)
            
#             # Generate samples using simplified reverse process
#             data_shape = x0.shape[1:]
#             num_eval_samples = x0.shape[0]
            
#             # Start from noise
#             x_t = torch.randn(num_eval_samples, *data_shape, device=self.device)
            
#             # Quick sampling with fewer steps for evaluation
#             num_timesteps = self.method.scheduler.num_timesteps
#             eval_steps = list(reversed(range(1, num_timesteps, 10)))  # Use fewer steps for faster evaluation
            
#             for t in eval_steps:
#                 t_tensor = torch.full((num_eval_samples,), t, device=self.device, dtype=torch.long)
#                 predicted_noise = self.model(x_t, t_tensor)
#                 x_t = self.method.p_sample(x_t, t_tensor, predicted_noise)
            
#             # Evaluate generated samples against reference
#             metrics = self.evaluator.compute_metrics(x_t, x0)
            
#             if self.logger:
#                 self.logger.log(metrics, step=epoch)
#                 if hasattr(self.logger, 'log_image'):
#                     self.logger.log_image(x_t, "generated_samples", step=epoch)
        
#         self.model.train()
    
#     def save_checkpoint(self, path: str, epoch: int) -> None:
#         """
#         Save model checkpoint.
        
#         Args:
#             path: Path to save checkpoint.
#             epoch: Current epoch.
#         """
#         checkpoint = {
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'epoch': epoch,
#             'global_step': self.global_step,
#         }
#         torch.save(checkpoint, path)
    
#     def load_checkpoint(self, path: str) -> int:
#         """
#         Load model checkpoint.
        
#         Args:
#             path: Path to checkpoint file.
            
#         Returns:
#             epoch: Epoch number from checkpoint.
#         """
#         checkpoint = torch.load(path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.global_step = checkpoint.get('global_step', 0)
#         return checkpoint['epoch'] 
    
    
    