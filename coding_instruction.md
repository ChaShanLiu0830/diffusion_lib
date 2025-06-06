# Diffusion Library Development Instructions

Follow these detailed instructions to implement the diffusion library based on the provided skeleton:

## 1. Trainer Implementation (`diffusion_lib/trainer/base_trainer.py`)
- Implement `_train_step(batch)`:
  - Forward the batch through the model.
  - Compute the loss using the diffusion method.
  - Perform a backward pass and optimize the model parameters.
  - Output: Return scalar loss.

## 2. Sampler Implementation (`diffusion_lib/sampler/base_sampler.py`)
- Implement `sample()`:
  - Iterate through the dataloader.
  - For each batch, generate `n_samples` using the model and diffusion method.
  - Output Shape: `[batch_size, n_samples, *data_shape]`.

## 3. Method Implementation (`diffusion_lib/method/base_method.py`)
- Implement `forward_process(x0)`:
  - Add noise according to the configured schedule.
  - Output: Noisy version `xt` and the noise added.

- Implement `reverse_process(xt, t, condition)`:
  - Perform denoising based on the trained model and timestep.
  - Output: Partially denoised data (`x_t_minus_1`).

- Implement optional `batch_operation(batch)` for preprocessing or optimal transport.

## 4. Logger Implementation (`diffusion_lib/logger/base_logger.py`)
- Implement `log(info, step)`:
  - Log training metrics to desired destination (file, terminal, visualization tool).

- Implement `log_image(images, tag, step)`:
  - Optionally log generated images to visualize model performance.

## 5. Configuration Manager (`diffusion_lib/manager/config_manager.py`)
- Already implemented. Ensure correct YAML structure for model, method, dataset, epochs, etc.

## 6. Evaluator Implementation (`diffusion_lib/evaluator/base_evaluator.py`)
- Implement `compute_metrics(samples, references)`:
  - Calculate desired metrics (e.g., FID, IS, MSE).
  - Output: Dictionary of computed metrics.

## General Coding Guidelines:
- Maintain modularity for easy experimentation.
- Follow clear naming conventions and include comprehensive docstrings.
- Implement clear error handling and validation checks where applicable.

---

## diffusion_lib/trainer/base_trainer.py

[The existing detailed skeleton implementation remains here]

## diffusion_lib/sampler/base_sampler.py

[The existing detailed skeleton implementation remains here]

## diffusion_lib/method/base_method.py

[The existing detailed skeleton implementation remains here]

## diffusion_lib/logger/base_logger.py

[The existing detailed skeleton implementation remains here]

## diffusion_lib/manager/config_manager.py

[The existing detailed skeleton implementation remains here]

## diffusion_lib/evaluator/base_evaluator.py

[The existing detailed skeleton implementation remains here]
