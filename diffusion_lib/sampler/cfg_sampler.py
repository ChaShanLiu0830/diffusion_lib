from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader

from ..sampler.base_sampler import BaseSampler


class CFGSampler(BaseSampler):
    """Sampler that applies classifier-free guidance at inference time."""

    def __init__(
        self,
        model: torch.nn.Module,
        method: Any,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        config: Optional[Dict[str, Any]] = None,
        load_best_model: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            method=method,
            device=device,
            config=config,
            load_best_model=load_best_model,
        )

    @torch.inference_mode()
    def sample_batch(
        self,
        num_samples: int,
        dataloader: DataLoader,
        steps: int = 1,
        return_intermediate: bool = False,
        cfg_scale: float = 1.0,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self.model.eval()

        collected_samples = []
        collected_intermediates = [] if return_intermediate else None

        for _ in range(num_samples):
            sample_chunks = []
            for batch in dataloader:
                batch = self.method.batch_operation(batch)
                x_0 = self.method.get_target(batch)
                x_t = self.method.get_noise(x_0.shape)
                condition = self.method.get_condition(batch)

                step_intermediates = [] if return_intermediate else None

                for timestep in reversed(range(0, self.method.scheduler.num_timesteps, steps)):
                    timesteps = torch.full((x_t.size(0),), timestep, device=self.device, dtype=torch.long)

                    cond_kwargs = dict(condition)
                    eps_cond = self.model(x_t, timesteps, **cond_kwargs)

                    uncond_kwargs = dict(condition)
                    uncond_kwargs["is_condition"] = False
                    eps_uncond = self.model(x_t, timesteps, **uncond_kwargs)

                    guided_eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                    x_t = self.method.p_sample(x_t, timesteps, guided_eps)

                    if step_intermediates is not None:
                        step_intermediates.append(x_t.detach().cpu())

                sample_chunks.append(x_t.detach().cpu())

                if collected_intermediates is not None and step_intermediates is not None:
                    collected_intermediates.append(torch.stack(step_intermediates, dim=-1))

            collected_samples.append(torch.cat(sample_chunks, dim=0))

        samples = torch.stack(collected_samples, dim=-1)

        if collected_intermediates is None:
            return samples

        return {"samples": samples, "intermediates": torch.stack(collected_intermediates, dim=0)}

