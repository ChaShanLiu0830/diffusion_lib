import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons

class MoonDataset(Dataset):
    def __init__(self, n_samples=1000, noise=0.1, transform=None):
        data, labels = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.data = (self.data - torch.mean(self.data, dim = 0, keepdim=True))/torch.std(self.data, dim = 0, keepdim=True)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx, :]
        if self.transform:
            sample = self.transform(sample)
        return {"x": sample, "z": label}