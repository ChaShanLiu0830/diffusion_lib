from typing import Any, Dict
import os
from datetime import datetime
# import logging

class BaseLogger:
    """Base logger class for tracking training metrics."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "diffusion_experiment"):
        """
        Initialize base logger.
        
        Args:
            log_dir: Directory to save logs.
            experiment_name: Name of the experiment.
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")

    def log(self, info: Dict[str, Any], step: int) -> None:
        """
        Logs training metrics.

        Args:
            info: Metrics to log (e.g., loss, accuracy).
            step: Training step or epoch.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] Step {step}: "
        
        # Format metrics
        metrics_str = ", ".join([f"{key}: {value}" for key, value in info.items()])
        log_entry += metrics_str
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
        
        # Print to console
        print(log_entry)

    def log_image(self, images: Any, tag: str, step: int) -> None:
        """
        Logs generated images.

        Args:
            images: Image tensor or array to log.
            tag: Label or category for images.
            step: Step or epoch of training.
        """
        # Base implementation just logs that images were generated
        # Subclasses can implement actual image saving/visualization
        self.log({"message": f"Generated {tag} images"}, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameters.
        """
        with open(self.log_file, 'a') as f:
            f.write("\nHyperparameters:\n")
            for key, value in hparams.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 50 + "\n")
    
    def close(self) -> None:
        """Close the logger."""
        with open(self.log_file, 'a') as f:
            f.write(f"\nExperiment ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") 