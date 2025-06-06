# Diffusion Library Architecture

## Overview

The Diffusion Library follows a modular, object-oriented design that separates concerns across distinct components. This architecture enables easy extensibility, testing, and maintenance while providing flexibility for different experimental setups.

## Design Principles

### 1. Single Responsibility Principle (SRP)
Each component has a single, well-defined responsibility:
- **Trainer**: Manages training loops and optimization
- **Sampler**: Handles sample generation from trained models
- **Method**: Implements diffusion algorithms (forward/reverse processes)
- **Logger**: Tracks and records training metrics
- **File Manager**: Manages model persistence and state
- **Config Manager**: Handles configuration loading and validation

### 2. Dependency Injection
Components receive their dependencies through constructor injection, making them testable and flexible:

```python
trainer = BaseTrainer(
    model=model,
    method=method,
    device=device,
    logger=logger,
    config=config
)
```

### 3. Interface Segregation
Base classes define minimal interfaces that can be extended:
- `BaseMethod`: Core diffusion operations
- `BaseLogger`: Logging interface
- `BaseEvaluator`: Metrics computation interface

## Component Architecture

### Core Components Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Config    │───▶│   Trainer    │───▶│   Logger    │
│  Manager    │    │              │    │             │
└─────────────┘    └──────┬───────┘    └─────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │ File Manager │
                  │              │
                  └──────────────┘
                          │
                          ▼
                  ┌──────────────┐    ┌─────────────┐
                  │    Model     │◄───│   Method    │
                  │              │    │   (DDPM)    │
                  └──────────────┘    └─────────────┘
                          │
                          ▼
                  ┌──────────────┐    ┌─────────────┐
                  │   Sampler    │───▶│  Evaluator  │
                  │              │    │             │
                  └──────────────┘    └─────────────┘
```

### Data Flow

1. **Configuration Phase**
   - `ConfigManager` loads YAML configuration
   - Validates required parameters
   - Provides configuration to components

2. **Training Phase**
   - `Trainer` orchestrates training loops
   - `Method` implements forward/reverse diffusion
   - `FileManager` handles checkpointing and state persistence
   - `Logger` tracks metrics and progress

3. **Sampling Phase**
   - `Sampler` loads trained models via `FileManager`
   - Uses `Method` for reverse sampling process
   - `Evaluator` computes quality metrics

## Component Details

### Trainer (`diffusion_lib.trainer.BaseTrainer`)

**Responsibilities:**
- Training loop management
- Validation tracking
- Optimizer coordination
- Checkpoint triggering

**Key Methods:**
- `train()`: Main training loop with epoch management
- `_resume_training()`: Restore training state from checkpoints
- `get_loss_histories()`: Access complete loss history

**Dependencies:**
- Model (PyTorch nn.Module)
- Method (diffusion algorithm)
- FileManager (persistence)
- Logger (metrics tracking)

### Method (`diffusion_lib.method.DDPM`)

**Responsibilities:**
- Forward diffusion process (noise addition)
- Reverse diffusion process (denoising)
- Loss computation
- Noise scheduling

**Key Methods:**
- `q_sample()`: Forward process x₀ → x_t
- `p_sample()`: Reverse process x_t → x_{t-1}
- `ddim_step()`: Deterministic sampling
- `compute_loss()`: Training loss calculation

**Design Pattern:** Strategy pattern for different diffusion algorithms

### File Manager (`diffusion_lib.manager.FileManager`)

**Responsibilities:**
- Model checkpoint saving/loading
- Training state persistence
- Loss history tracking
- Best model identification

**Key Features:**
- Automatic best model selection based on validation loss
- Complete loss history preservation across sessions
- Configurable saving strategies (best only vs. regular intervals)

**State Management:**
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': OrderedDict,
    'metrics': Dict[str, Any],
    'best_val_loss': float,
    'training_loss_history': List[float],
    'val_loss_history': List[float],
    'config': Dict[str, Any]
}
```

### Sampler (`diffusion_lib.sampler.BaseSampler`)

**Responsibilities:**
- Sample generation from trained models
- Model loading and management
- Batch processing for efficient sampling

**Sampling Process:**
1. Load trained model (automatically finds best checkpoint)
2. Initialize noise tensor
3. Iteratively denoise using reverse process
4. Return generated samples

### Logger (`diffusion_lib.logger.BaseLogger`)

**Responsibilities:**
- Training metrics logging
- File-based persistence
- Console output formatting
- Experiment tracking

**Features:**
- Timestamped logging entries
- Support for both structured (dict) and string messages
- Hyperparameter logging
- Experiment session management

### Config Manager (`diffusion_lib.manager.ConfigManager`)

**Responsibilities:**
- YAML configuration loading
- Nested parameter access with dot notation
- Configuration validation
- Runtime configuration updates

**Usage Pattern:**
```python
config = ConfigManager('config.yaml')
model_params = config.get('model.params')
config.validate_required_keys(['training.epochs', 'diffusion.num_timesteps'])
```

## Extension Points

### Custom Methods
Implement new diffusion algorithms by extending `BaseMethod`:

```python
class CustomDiffusion(BaseMethod):
    def q_sample(self, x0, t):
        # Custom forward process
        pass
    
    def p_sample(self, xt, t, eps_pred):
        # Custom reverse process
        pass
```

### Custom Loggers
Extend logging capabilities:

```python
class WandBLogger(BaseLogger):
    def log(self, info, step):
        super().log(info, step)
        wandb.log(info, step=step)
```

### Custom Evaluators
Add domain-specific metrics:

```python
class ImageEvaluator(BaseEvaluator):
    def compute_fid(self, samples, references):
        # Implement FID calculation
        pass
```

## Error Handling

The library implements defensive programming with:
- Input validation at component boundaries
- Graceful degradation when optional components are missing
- Clear error messages with actionable guidance
- Recovery mechanisms for training interruptions

## Performance Considerations

### Memory Management
- Automatic GPU memory management
- Batch processing for large datasets
- Gradient accumulation support (planned)

### Checkpointing Strategy
- Configurable checkpoint frequency
- Automatic cleanup of old checkpoints (planned)
- Incremental state saving to minimize I/O

### Inference Optimization
- `torch.inference_mode()` for sampling
- Model compilation support (planned)
- Mixed precision training support (planned)

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock dependencies for isolation
- Edge case validation

### Integration Tests
- End-to-end training workflows
- Configuration validation
- Cross-component compatibility

### Performance Tests
- Memory usage profiling
- Training speed benchmarks
- Sampling efficiency tests

This architecture enables the library to be both powerful for research and production use while maintaining clarity and extensibility for future development. 