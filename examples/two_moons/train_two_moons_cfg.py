#!/usr/bin/env python3
"""Train and sample a DDPM with classifier-free guidance on two moons."""

import argparse
import math
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from diffusion_lib.method.ddpm import DDPM, DDPMScheduler
from diffusion_lib.trainer.cfg_trainer import CFGTrainer
from diffusion_lib.sampler.cfg_sampler import CFGSampler


class TwoMoonsConditionalDataset(Dataset):
    """Two moons dataset extended with binary labels for conditioning."""

    def __init__(self, n_samples: int = 4000, noise: float = 0.08, seed: int = 0) -> None:
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2 to form two moons.")

        generator = torch.Generator().manual_seed(seed)
        n_outer = n_samples // 2
        n_inner = n_samples - n_outer

        theta_outer = torch.rand(n_outer, generator=generator) * math.pi
        theta_inner = torch.rand(n_inner, generator=generator) * math.pi

        outer = torch.stack((torch.cos(theta_outer), torch.sin(theta_outer)), dim=1)
        inner = torch.stack((1 - torch.cos(theta_inner), 1 - torch.sin(theta_inner) - 0.5), dim=1)

        data = torch.cat((outer, inner), dim=0)
        labels_outer = torch.zeros(n_outer, dtype=torch.long)
        labels_inner = torch.ones(n_inner, dtype=torch.long)
        labels = torch.cat((labels_outer, labels_inner), dim=0)

        if noise > 0:
            data = data + noise * torch.randn(data.shape, generator=generator)

        self.data = data.float()
        self.labels = labels

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


class ZeroConditionalDataset(Dataset):
    """Dataset of zeros paired with a fixed class label for sampling."""

    def __init__(self, num_samples: int, feature_dim: int, label: int) -> None:
        self.points = torch.zeros(num_samples, feature_dim, dtype=torch.float32)
        self.labels = torch.full((num_samples,), label, dtype=torch.long)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.points[idx], self.labels[idx]


class ConditionalDDPM(DDPM):
    """DDPM wrapper that produces conditioning information from dataset labels."""

    def __init__(self, scheduler: DDPMScheduler, device: torch.device, num_classes: int = 2) -> None:
        super().__init__(scheduler=scheduler, device=device)
        self.num_classes = num_classes

    def batch_operation(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        points, labels = batch
        return {
            "points": points.to(self.device),
            "labels": labels.to(self.device),
        }

    def get_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["points"]

    def get_condition(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        labels = batch["labels"]
        class_embed = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        return {"class_embed": class_embed}


class ConditionalTimeConditionedMLP(nn.Module):
    """MLP that conditions on both timestep embeddings and class embeddings."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, time_embed_dim: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.condition_dim = num_classes
        combined_dim = input_dim + time_embed_dim + self.condition_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        class_embed: Optional[torch.Tensor] = None,
        is_condition: bool = True,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        if class_embed is None:
            class_embed = torch.zeros(x.size(0), self.condition_dim, device=x.device)

        if not is_condition:
            class_embed = torch.zeros_like(class_embed)

        h = torch.cat((x, t_emb, class_embed), dim=-1)
        return self.net(h)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding used in diffusion models."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.embed_dim // 2
        timesteps = timesteps.float()
        exponent = torch.arange(half_dim, device=device) / half_dim
        frequencies = torch.exp(-math.log(10000.0) * exponent)
        args = timesteps[:, None] * frequencies[None, :]
        embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        if self.embed_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two moons diffusion example with CFG")
    parser.add_argument("--n-samples", type=int, default=4096, help="Total synthetic samples to generate")
    parser.add_argument("--noise", type=float, default=0.08, help="Std of Gaussian noise added to moons")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--timesteps", type=int, default=200, help="Number of diffusion timesteps")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size of the MLP")
    parser.add_argument("--time-embed-dim", type=int, default=64, help="Dimension of time embeddings")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="Adam learning rate")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint saving frequency")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data used for validation")
    parser.add_argument("--num-generated", type=int, default=2048, help="Samples drawn after training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoints if available")
    parser.add_argument("--cfg-dropout-prob", type=float, default=0.1, help="Classifier-free guidance dropout probability")
    parser.add_argument("--guidance-scale", type=float, default=3.0, help="Guidance scale during sampling")
    parser.add_argument("--sample-class", type=int, default=0, choices=[0, 1], help="Class label to condition sampling on")
    return parser.parse_args()


def build_dataloaders(dataset: Dataset, args: argparse.Namespace) -> Tuple[DataLoader, Optional[DataLoader]]:
    val_fraction = max(0.0, min(0.5, args.val_split))
    val_size = int(len(dataset) * val_fraction)
    if val_fraction > 0 and val_size == 0:
        val_size = 1
    if val_size >= len(dataset):
        val_size = len(dataset) - 1

    if val_size > 0:
        generator = torch.Generator().manual_seed(args.seed)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    else:
        train_dataset = dataset
        val_loader = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader, val_loader


def train_model(dataset: Dataset, args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, ConditionalDDPM, Dict[str, object], str]:
    train_loader, val_loader = build_dataloaders(dataset, args)

    scheduler = DDPMScheduler(num_timesteps=args.timesteps, device=device)
    method = ConditionalDDPM(scheduler=scheduler, device=device, num_classes=2)
    model = ConditionalTimeConditionedMLP(hidden_dim=args.hidden_dim, time_embed_dim=args.time_embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    output_dir = os.path.join(os.path.dirname(__file__), "artifacts_cfg")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_config: Dict[str, object] = {
        "save_dir": checkpoints_dir,
        "model_name": "two_moons_cfg",
        "save_best_only": False,
    }

    trainer = CFGTrainer(
        model=model,
        method=method,
        device=device,
        optimizer=optimizer,
        config=checkpoint_config.copy(),
    )

    history = trainer.train(
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        epochs=args.epochs,
        save_every=args.save_every,
        resume_training=args.resume,
    )

    print("Final metrics:", history["final_metrics"])

    return model, method, checkpoint_config, output_dir


def sample_with_cfg_sampler(
    model: nn.Module,
    method: ConditionalDDPM,
    device: torch.device,
    config: Dict[str, object],
    num_points: int,
    batch_size: int,
    feature_dim: int,
    guidance_scale: float,
    sample_class: int,
) -> torch.Tensor:
    sampler = CFGSampler(
        model=model,
        method=method,
        device=device,
        config=config.copy(),
        load_best_model=True,
    )

    sampling_loader = DataLoader(
        ZeroConditionalDataset(num_points, feature_dim, label=sample_class),
        batch_size=batch_size,
        shuffle=False,
    )

    samples = sampler.sample_batch(num_samples=1, dataloader=sampling_loader, cfg_scale=guidance_scale)
    return samples[..., 0]


def save_scatter(points: torch.Tensor, path: str, title: str) -> None:
    arr = points.detach().cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.scatter(arr[:, 0], arr[:, 1], s=6, alpha=0.6)
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TwoMoonsConditionalDataset(n_samples=args.n_samples, noise=args.noise, seed=args.seed)
    model, method, checkpoint_config, output_dir = train_model(dataset, args, device)

    samples = sample_with_cfg_sampler(
        model=model,
        method=method,
        device=device,
        config=checkpoint_config,
        num_points=args.num_generated,
        batch_size=args.batch_size,
        feature_dim=dataset.data.shape[1],
        guidance_scale=args.guidance_scale,
        sample_class=args.sample_class,
    )

    data_plot = os.path.join(output_dir, "two_moons_data_cfg.png")
    sample_plot = os.path.join(output_dir, "two_moons_samples_cfg.png")
    tensor_path = os.path.join(output_dir, "two_moons_samples_cfg.pt")

    save_scatter(dataset.data, data_plot, "Two moons dataset (CFG)")
    save_scatter(samples, sample_plot, "DDPM two moons samples (CFG)")
    torch.save(samples, tensor_path)

    print(f"Artifacts saved in {output_dir}")
    print(f"Plots: {data_plot}, {sample_plot}")
    print(f"Sample tensor: {tensor_path}")


if __name__ == "__main__":
    main()
