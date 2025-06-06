from typing import Any, Optional, Union, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..utils.func import repeat
import torch.nn.functional as F
from ..logger.base_logger import BaseLogger
from tqdm import tqdm
from ..manager.file_manager import FileManager


class BaseTrainer:
    """
    Base trainer class for handling model training with proper state management.
    
    This class follows SOLID principles by separating concerns between training logic,
    file management, and logging.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        method: Any,  
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        optimizer: torch.optim.Optimizer = None, 
        logger: BaseLogger = None, 
        config: dict = None, 
        **kwargs
    ):
        """
        Initialize the BaseTrainer.
        
        Args:
            model: PyTorch model to train
            method: Training method containing scheduler and loss computation
            device: Device to run training on
            optimizer: Optimizer for training (default: Adam with lr=1e-4)
            logger: Logger for training metrics
            config: Configuration dictionary for file management and training
            **kwargs: Additional arguments
        """
        self.model = model.to(device)
        self.method = method
        self.device = device
        self.optim = optimizer if optimizer is not None else torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.logger = logger
        self.file_manager = FileManager(config=config if config is not None else {})
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
    def train(self, 
              train_dataloader: DataLoader, 
              valid_dataloader: DataLoader = None, 
              epochs: int = 100, 
              save_every: int = 10,
              resume_training: bool = True) -> Dict[str, Any]:
        """
        Train the model with proper checkpointing and validation tracking.
        
        Args:
            train_dataloader: DataLoader for training data
            valid_dataloader: DataLoader for validation data
            epochs: Total number of epochs to train
            save_every: Save checkpoint every N epochs
            resume_training: Whether to resume from existing checkpoint
            
        Returns:
            Dict containing training history and final metrics
        """
        # Resume training if requested and checkpoint exists
        if resume_training:
            self._resume_training()
        
        # Get restored loss histories from file manager
        restored_histories = self.file_manager.get_loss_histories()
        training_history = {
            'train_losses': restored_histories['training_loss_history'],
            'val_losses': restored_histories['val_loss_history'],
            'epochs': list(range(len(restored_histories['training_loss_history'])))
        }
        
        for epoch in tqdm(range(self.start_epoch, epochs), desc="Training"):
            info = {'epoch': epoch}
            
            # Training phase
            self.model.train()
            train_loss = self._run_epoch(train_dataloader, grad=True)
            info['train_loss'] = train_loss
            training_history['train_losses'].append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            if valid_dataloader is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self._run_epoch(valid_dataloader, grad=False)
                    info['val_loss'] = val_loss
                    training_history['val_losses'].append(val_loss)
            
            training_history['epochs'].append(epoch)
            
            # Logging
            if self.logger is not None:
                self.logger.log(info)
                self.logger.log(f"Epoch {epoch:4d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
            
            # Save model based on validation loss improvement or regular intervals
            should_save_regular = epoch % save_every == 0 or epoch == epochs - 1
            model_saved = self.file_manager.save_model(
                model=self.model,
                epoch=epoch,
                metrics=info,
                optimizer=self.optim
            )
            
            # Save at regular intervals if not saved due to improvement
            if should_save_regular and not model_saved and not self.file_manager.save_best_only:
                self.file_manager.save_model(
                    model=self.model,
                    epoch=epoch,
                    metrics=info,
                    optimizer=self.optim
                )
            
            # Update best validation loss for tracking
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.logger is not None:
                    self.logger.log(f"New best validation loss: {val_loss:.4f}")
        
        # Final training summary
        final_metrics = {
            'final_train_loss': training_history['train_losses'][-1] if training_history['train_losses'] else 0.0,
            'final_val_loss': training_history['val_losses'][-1] if training_history['val_losses'] else 0.0,
            'best_val_loss': self.best_val_loss,
            'total_epochs_trained': len(training_history['epochs'])
        }
        
        if self.logger is not None:
            self.logger.log("Training completed!")
            self.logger.log(final_metrics)
        
        return {
            'history': training_history,
            'final_metrics': final_metrics
        }
    
    def _resume_training(self) -> None:
        """
        Resume training from the latest checkpoint.
        """
        resume_info = self.file_manager.get_resume_info()
        
        if resume_info['has_checkpoint']:
            checkpoint_info = self.file_manager.load_checkpoint(
                model=self.model,
                device=self.device,
                optimizer=self.optim
            )
            
            self.start_epoch = checkpoint_info['epoch'] + 1
            self.best_val_loss = checkpoint_info['best_val_loss']
            
            if self.logger is not None:
                self.logger.log(f"Resumed training from epoch {checkpoint_info['epoch']}")
                self.logger.log(f"Best validation loss so far: {self.best_val_loss:.4f}")
        else:
            if self.logger is not None:
                self.logger.log("No checkpoint found, starting training from scratch")
    
    def _run_epoch(self, dataloader: DataLoader, grad: bool = True) -> float:
        """
        Run a single epoch of training or validation.
        
        Args:
            dataloader: DataLoader to iterate over
            grad: Whether to compute gradients and update weights
            
        Returns:
            Average loss for the epoch
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch = self.method.batch_operation(batch)
            x_0 = self.method.get_target(batch)
            condition = self.method.get_condition(batch)
            
            t = torch.randint(
                0, self.method.scheduler.num_timesteps, 
                (x_0.size(0),),
                device=self.device, 
                dtype=torch.long
            )
            
            x_t, eps = self.method.q_sample(x_0, t, **condition)
            eps_pred = self.model(x_t, t)
            loss = self.method.compute_loss(eps_pred, eps)

            if grad:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            total_loss += loss.item()
            # num_batches += 1
        
        return total_loss / len(dataloader)
            
    