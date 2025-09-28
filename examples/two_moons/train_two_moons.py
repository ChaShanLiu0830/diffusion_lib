#!/usr/bin/env python3
"""Train a DDPM on a toy two moons dataset using diffusion_lib components."""

import argparse
import math
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from diffusion_lib.method.ddpm import DDPM, DDPMScheduler
from diffusion_lib.trainer.base_trainer import BaseTrainer
from diffusion_lib.sampler.base_sampler import BaseSampler


class TwoMoonsDataset(Dataset):
    """Generates a deterministic two moons dataset with optional Gaussian noise."""

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
        if noise > 0:
            data = data + noise * torch.randn(data.shape, generator=generator)

        self.data = data.float()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class ZeroDataset(Dataset):
    """Utility dataset that provides zero tensors for sampler shape hints."""

    def __init__(self, num_samples: int, feature_dim: int) -> None:
        self.data = torch.zeros(num_samples, feature_dim, dtype=torch.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


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


class TimeConditionedMLP(nn.Module):
    """Minimal MLP that predicts diffusion noise conditioned on timesteps."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, time_embed_dim: int = 64) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **unused: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        h = torch.cat((x, t_emb), dim=-1)
        return self.net(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two moons diffusion example")
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


def train_model(dataset: Dataset, args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, DDPM, Dict[str, object], str]:
    train_loader, val_loader = build_dataloaders(dataset, args)

    scheduler = DDPMScheduler(num_timesteps=args.timesteps, device=device)
    method = DDPM(scheduler=scheduler, device=device)
    model = TimeConditionedMLP(hidden_dim=args.hidden_dim, time_embed_dim=args.time_embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    output_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_config: Dict[str, object] = {
        "save_dir": checkpoints_dir,
        "model_name": "two_moons",
        "save_best_only": False,
    }

    trainer = BaseTrainer(
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


def sample_with_base_sampler(
    model: nn.Module,
    method: DDPM,
    device: torch.device,
    config: Dict[str, object],
    num_points: int,
    batch_size: int,
    feature_dim: int,
) -> torch.Tensor:
    """Use BaseSampler to draw samples from the trained DDPM model."""

    sampler = BaseSampler(
        model=model,
        method=method,
        device=device,
        config=config.copy(),
        load_best_model=True,
    )

    sampling_loader = DataLoader(
        ZeroDataset(num_points, feature_dim),
        batch_size=batch_size,
        shuffle=False,
    )

    samples = sampler.sample_batch(num_samples=1, dataloader=sampling_loader)
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

    dataset = TwoMoonsDataset(n_samples=args.n_samples, noise=args.noise, seed=args.seed)
    model, method, checkpoint_config, output_dir = train_model(dataset, args, device)

    samples = sample_with_base_sampler(
        model=model,
        method=method,
        device=device,
        config=checkpoint_config,
        num_points=args.num_generated,
        batch_size=args.batch_size,
        feature_dim=dataset.data.shape[1],
    )

    data_plot = os.path.join(output_dir, "two_moons_data.png")
    sample_plot = os.path.join(output_dir, "two_moons_samples.png")
    tensor_path = os.path.join(output_dir, "two_moons_samples.pt")

    save_scatter(dataset.data, data_plot, "Two moons dataset")
    save_scatter(samples, sample_plot, "DDPM two moons samples")
    torch.save(samples, tensor_path)

    print(f"Artifacts saved in {output_dir}")
    print(f"Plots: {data_plot}, {sample_plot}")
    print(f"Sample tensor: {tensor_path}")


if __name__ == "__main__":
    main()
