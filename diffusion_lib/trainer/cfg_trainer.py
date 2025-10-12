from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..logger.base_logger import BaseLogger
from ..trainer.base_trainer import BaseTrainer


class CFGTrainer(BaseTrainer):
    """Trainer that optimizes classifier-free guidance objectives."""

    def __init__(
        self,
        model: nn.Module,
        method: Any,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        optimizer: Optional[torch.optim.Optimizer] = None,
        logger: Optional[BaseLogger] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            method=method,
            device=device,
            optimizer=optimizer,
            logger=logger,
            config=config,
            **kwargs,
        )

    def _run_epoch(self, dataloader: DataLoader, grad: bool = True) -> float:
        total_loss = 0.0

        for batch in dataloader:
            batch = self.method.batch_operation(batch)
            x_0 = self.method.get_target(batch)
            condition = self.method.get_condition(batch)

            timesteps = torch.randint(
                0,
                self.method.scheduler.num_timesteps,
                (x_0.size(0),),
                device=self.device,
                dtype=torch.long,
            )

            x_t, eps = self.method.q_sample(x_0, timesteps)

            cond_kwargs = dict(condition)
            eps_pred_cond = self.model(x_t, timesteps, **cond_kwargs)

            uncond_kwargs = dict(condition)
            uncond_kwargs["is_condition"] = False
            eps_pred_uncond = self.model(x_t, timesteps, **uncond_kwargs)

            loss_cond = self.method.compute_loss(eps_pred_cond, eps)
            loss_uncond = self.method.compute_loss(eps_pred_uncond, eps)
            loss = 0.5 * (loss_cond + loss_uncond)

            if grad:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
