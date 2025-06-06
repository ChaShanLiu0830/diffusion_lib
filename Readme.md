# Diffusion Library

A comprehensive, modular PyTorch-based framework for training, sampling, and evaluating diffusion models. This library provides a flexible architecture designed for ease of experimentation and extensibility in diffusion model research.

## 🚀 Features

- **Modular Design**: Clean separation of concerns with independent components
- **DDPM Implementation**: Complete Denoising Diffusion Probabilistic Models implementation
- **Advanced Training**: Comprehensive training with checkpointing, resume capability, and loss history tracking
- **Flexible Sampling**: Multiple sampling strategies including DDIM deterministic sampling
- **Evaluation Metrics**: Built-in quality and diversity metrics for generated samples
- **Configuration Management**: YAML-based configuration with validation
- **File Management**: Automatic model saving, loading, and training state persistence
- **Logging System**: Comprehensive logging with timestamp tracking and hyperparameter logging

## 📁 Library Structure

```
diffusion_lib/
├── trainer/
│   └── base_trainer.py          # Training loops with checkpointing and loss tracking
├── sampler/
│   └── base_sampler.py          # Sample generation with various strategies
├── method/
│   ├── base_method.py           # Abstract base for diffusion methods
│   └── ddpm.py                  # DDPM implementation with scheduler
├── logger/
│   └── base_logger.py           # Logging system with file and console output
├── manager/
│   ├── config_manager.py        # YAML configuration management
│   └── file_manager.py          # Model and checkpoint management with loss history
├── evaluator/
│   └── base_evaluator.py        # Evaluation metrics (MSE, MAE, PSNR, diversity)
├── utils/
│   └── func.py                  # Utility functions for tensor operations
└── __init__.py
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12.0+
- CUDA (optional, for GPU acceleration)

### Install from Source

```bash
git clone https://github.com/yourusername/diffusion_lib.git
cd diffusion_lib
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Training Example

```python
import torch
from diffusion_lib.trainer import BaseTrainer
from diffusion_lib.method.ddpm import DDPM, DDPMScheduler
from diffusion_lib.logger import BaseLogger
from diffusion_lib.manager import ConfigManager

# Configuration
config = {
    'save_dir': './checkpoints',
    'model_name': 'my_diffusion_model',
    'save_best_only': True
}

# Initialize components
scheduler = DDPMScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=2e-2)
method = DDPM(scheduler=scheduler)
logger = BaseLogger(log_dir="./logs", experiment_name="diffusion_experiment")

# Initialize trainer
trainer = BaseTrainer(
    model=your_model,
    method=method,
    device=torch.device("cuda"),
    logger=logger,
    config=config
)

# Train with automatic checkpointing and loss history tracking
results = trainer.train(
    train_dataloader=train_loader,
    valid_dataloader=val_loader,
    epochs=100,
    save_every=10,
    resume_training=True  # Automatically resumes from last checkpoint
)

# Access complete loss history
loss_histories = trainer.get_loss_histories()
print(f"Training loss history: {loss_histories['training_loss_history']}")
print(f"Validation loss history: {loss_histories['val_loss_history']}")
```

### Sampling from Trained Models

```python
from diffusion_lib.sampler import BaseSampler

# Initialize sampler (automatically loads best model)
sampler = BaseSampler(
    model=your_model,
    method=method,
    device=torch.device("cuda"),
    config=config,
    load_best_model=True
)

# Generate samples
samples = sampler.sample_batch(
    num_samples=16,
    dataloader=test_loader,
    steps=1,  # Full sampling
    return_intermediate=False
)
```

### Configuration Management

```python
from diffusion_lib.manager import ConfigManager

# Load configuration from YAML
config_manager = ConfigManager('configs/experiment.yaml')

# Access nested configurations
model_config = config_manager.get_model_config()
training_config = config_manager.get_training_config()
diffusion_config = config_manager.get_diffusion_config()

# Validate required keys
config_manager.validate_required_keys(['model.type', 'training.epochs'])
```

### Evaluation

```python
from diffusion_lib.evaluator import BaseEvaluator

evaluator = BaseEvaluator(device="cuda")

# Compute reconstruction metrics
metrics = evaluator.compute_metrics(generated_samples, ground_truth)
print(f"MSE: {metrics['mse']:.4f}")
print(f"PSNR: {metrics['psnr']:.2f} dB")

# Compute diversity metrics
diversity = evaluator.compute_diversity_metrics(generated_samples)
print(f"Mean pairwise distance: {diversity['mean_pairwise_distance']:.4f}")
```

## 📋 Configuration

Create YAML configuration files for your experiments:

```yaml
# config.yaml
model:
  type: "unet"
  channels: 3
  image_size: 64

training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-4
  save_every: 10

diffusion:
  method: "ddpm"
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02

data:
  dataset: "cifar10"
  data_dir: "./data"
  num_workers: 4

logging:
  log_dir: "./logs"
  experiment_name: "my_experiment"

file_management:
  save_dir: "./checkpoints"
  model_name: "diffusion_model"
  save_best_only: true
```

## 🧪 Advanced Features

### Loss History Tracking

The library automatically tracks and preserves complete training and validation loss histories across training sessions:

```python
# After training or resuming
trainer = BaseTrainer(model, method, config=config)
results = trainer.train(train_loader, val_loader, resume_training=True)

# Access complete loss history for plotting
loss_histories = trainer.get_loss_histories()
train_losses = loss_histories['training_loss_history']
val_losses = loss_histories['val_loss_history']

# Loss history is also available in training results
full_history = results['history']
```

### DDIM Sampling

The DDPM implementation supports both stochastic and deterministic (DDIM) sampling:

```python
# DDIM deterministic sampling
method = DDPM(scheduler)
x_prev = method.ddim_step(x_t, t, eps_pred, eta=0.0)  # eta=0 for deterministic

# Stochastic DDPM sampling
x_prev = method.p_sample(x_t, t, eps_pred)
```

### Custom Evaluation Metrics

Extend the base evaluator for domain-specific metrics:

```python
class CustomEvaluator(BaseEvaluator):
    def compute_custom_metrics(self, samples, references):
        # Implement your custom metrics
        return {"custom_metric": value}
```

## 🔧 Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

The library follows PEP 8 style guidelines. Format code using:

```bash
black diffusion_lib/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Add tests for new functionality
5. Submit a pull request

## 📊 Components Overview

### Trainer (`diffusion_lib.trainer.BaseTrainer`)
- **Training Loop**: Handles epoch-based training with validation
- **Checkpointing**: Automatic model saving with configurable frequency
- **Resume Training**: Seamless training resumption with complete state restoration
- **Loss Tracking**: Comprehensive loss history preservation
- **Optimizer Management**: Integration with PyTorch optimizers and schedulers

### Method (`diffusion_lib.method.DDPM`)
- **Forward Process**: `q_sample()` for adding noise to clean data
- **Reverse Process**: `p_sample()` for denoising steps
- **DDIM Support**: `ddim_step()` for deterministic sampling
- **Loss Computation**: MSE loss between predicted and actual noise
- **Flexible Scheduling**: Linear beta schedule with configurable parameters

### Sampler (`diffusion_lib.sampler.BaseSampler`)
- **Batch Sampling**: Generate multiple sample sets efficiently
- **Model Loading**: Automatic best model loading or specific checkpoint loading
- **Inference Mode**: GPU-optimized inference with `torch.inference_mode()`
- **Flexible Steps**: Configurable sampling step sizes

### Logger (`diffusion_lib.logger.BaseLogger`)
- **File Logging**: Persistent logging to timestamped files
- **Console Output**: Real-time training progress in terminal
- **Structured Logging**: Support for dictionaries and string messages
- **Hyperparameter Tracking**: Log experiment configurations

### File Manager (`diffusion_lib.manager.FileManager`)
- **Model Persistence**: Save and load model checkpoints
- **State Management**: Complete training state preservation
- **Loss History**: Automatic tracking and restoration of loss histories
- **Best Model Tracking**: Automatic identification and storage of best performing models

### Config Manager (`diffusion_lib.manager.ConfigManager`)
- **YAML Support**: Load and save YAML configuration files
- **Nested Access**: Dot notation for accessing nested configuration values
- **Validation**: Ensure required configuration keys are present
- **Dynamic Updates**: Runtime configuration modifications

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Citation

If you use this library in your research, please cite:

```bibtex
@software{diffusion_lib,
  title={Diffusion Library: A Modular PyTorch Framework for Diffusion Models},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/diffusion_lib}
}
```

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/yourusername/diffusion_lib/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/diffusion_lib/discussions)
- **Email**: your.email@example.com