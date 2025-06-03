import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path


class FileManager:
    """
    Manages file operations for model saving, loading, and training state persistence.
    
    This class follows the Single Responsibility Principle by focusing solely on
    file management operations for machine learning workflows.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FileManager with configuration.
        
        Args:
            config: Configuration dictionary containing paths and settings.
                   Expected keys:
                   - 'save_dir': Directory to save models and checkpoints
                   - 'model_name': Base name for model files
                   - 'save_best_only': Whether to only save best models (default: True)
        """
        self.config = config
        self.save_dir = Path(config.get('save_dir', './checkpoints'))
        self.model_name = config.get('model_name', 'model')
        self.save_best_only = config.get('save_best_only', True)
        
        # Create directories if they don't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Checkpoint tracking
        self.checkpoint_file = self.save_dir / 'training_state.json'
        self._load_training_state()
    
    def save_model(self, model: nn.Module, epoch: int, metrics: Dict[str, Any], 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None) -> bool:
        """
        Save model checkpoint with training metadata.
        
        Args:
            model: PyTorch model to save
            epoch: Current training epoch
            metrics: Dictionary containing training metrics
            optimizer: Optional optimizer state to save
            scheduler: Optional scheduler state to save
            
        Returns:
            bool: True if model was saved, False otherwise
        """
        val_loss = metrics.get('val_loss', float('inf'))
        should_save = False
        
        # Determine if we should save this model
        if not self.save_best_only:
            should_save = True
        elif val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            should_save = True
        
        if should_save:
            model_path = self.save_dir / f"{self.model_name}_epoch_{epoch}.pth"
            
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'best_val_loss': self.best_val_loss,
                'config': self.config
            }
            
            # Add optimizer and scheduler states if provided
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # Save the checkpoint
            torch.save(checkpoint, model_path)
            
            # Update best model path if this is the best model
            if val_loss <= self.best_val_loss:
                self.best_model_path = str(model_path)
                self._save_best_model_info()
            
            # Save training state
            self._save_training_state(epoch, metrics)
            
            return True
        
        return False
    
    def load_best_model(self, model: nn.Module, device: torch.device) -> Dict[str, Any]:
        """
        Load the best saved model.
        
        Args:
            model: PyTorch model to load weights into
            device: Device to load the model on
            
        Returns:
            Dict containing loaded checkpoint metadata
            
        Raises:
            FileNotFoundError: If no best model is found
        """
        if self.best_model_path is None or not os.path.exists(self.best_model_path):
            raise FileNotFoundError("No best model found to load")
        
        checkpoint = torch.load(self.best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf'))
        }
    
    def load_checkpoint(self, model: nn.Module, device: torch.device,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """
        Load the latest checkpoint for resuming training.
        
        Args:
            model: PyTorch model to load weights into
            device: Device to load the model on
            optimizer: Optional optimizer to restore state
            scheduler: Optional scheduler to restore state
            
        Returns:
            Dict containing checkpoint information for resuming training
        """
        if self.best_model_path is None or not os.path.exists(self.best_model_path):
            return {'epoch': 0, 'metrics': {}, 'best_val_loss': float('inf')}
        
        checkpoint = torch.load(self.best_model_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Update internal state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'best_val_loss': self.best_val_loss
        }
    
    def get_resume_info(self) -> Dict[str, Any]:
        """
        Get information needed to resume training.
        
        Returns:
            Dict containing resume information
        """
        return {
            'best_val_loss': self.best_val_loss,
            'best_model_path': self.best_model_path,
            'has_checkpoint': self.best_model_path is not None and os.path.exists(self.best_model_path)
        }
    
    def _save_training_state(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Save current training state to JSON file.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
        """
        state = {
            'epoch': epoch,
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'best_model_path': self.best_model_path
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_training_state(self) -> None:
        """Load previous training state if it exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
                self.best_val_loss = state.get('best_val_loss', float('inf'))
                self.best_model_path = state.get('best_model_path')
    
    def _save_best_model_info(self) -> None:
        """Save information about the best model."""
        best_info = {
            'best_model_path': self.best_model_path,
            'best_val_loss': self.best_val_loss
        }
        
        best_info_file = self.save_dir / 'best_model_info.json'
        with open(best_info_file, 'w') as f:
            json.dump(best_info, f, indent=2)