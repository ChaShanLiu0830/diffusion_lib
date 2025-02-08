import os
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn 
import re
import logging
# class BaseTrainer:
#     def __init__(self, model, train_dataloader, valid_dataloader, scheduler, batch_size, model_name="model", device='cpu', lr=1e-4, logger=None):
#         self.model = model.to(device)
#         self.train_dataloader = train_dataloader
#         self.valid_dataloader = valid_dataloader
#         self.scheduler = scheduler
#         self.batch_size = batch_size
#         self.device = device
#         self.optimizer = Adam(self.model.parameters(), lr=lr)
#         self.criterion = MSELoss()
#         self.logger = logger
#         self.model_name = model_name
#         self.checkpoint_dir = f"./ckpt/{self.model_name}/"

#         self.before_training_check()
#         self.train_loss_dict = {"epoch": []}
#         self.valid_loss_dict = {"epoch": []}

#     def loss_fn(self, noise_pred, noise):
#         total_loss = self.criterion(noise_pred, noise)
#         return {"general_loss": total_loss, "total_loss": total_loss}

#     def before_training_check(self):
#         os.makedirs(f"./progress/", exist_ok=True)
#         os.makedirs(self.checkpoint_dir, exist_ok=True)

#     def _one_epoch(self, batch):
#         x_start = batch["x"].to(self.device)
#         z = batch["z"].to(self.device)
#         t = torch.randint(0, self.scheduler.num_timesteps, (self.batch_size,), device=self.device)

#         alpha_bars_sqrt_t = self.scheduler.alpha_bars_sqrt_t[t]
#         one_minus_alpha_bars_sqrt_t = self.scheduler.one_minus_alpha_bars_sqrt_t[t]
#         noise = torch.randn_like(x_start).to(torch.float32).to(self.device)
#         x_noisy = alpha_bars_sqrt_t.view([-1] + [1 for _ in range(0, x_start.dim()-1)]) * x_start + one_minus_alpha_bars_sqrt_t.view([-1] + [1 for _ in range(0, x_start.dim()-1)]) * noise
#         noise_pred = self.model(x_noisy, t, z)
#         return self.loss_fn(noise_pred, noise)

#     def train(self, epochs, save_freq=50):
#         best_val_loss = float('inf')
#         start_epoch = 0
#         for epoch in tqdm(range(start_epoch, epochs)):
#             for batch in self.train_dataloader:
#                 loss_dict = self._one_epoch(batch)
#                 self.optimizer.zero_grad()
#                 loss_dict["total_loss"].backward()
#                 self.optimizer.step()

#             self.save_checkpoint(epoch + 1)
    
#     def save_checkpoint(self, epoch):
#         checkpoint_path = self.checkpoint_dir + self.model_name + f"_ckpt{epoch}.pt"
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#         }, checkpoint_path)


class BaseTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader, scheduler, batch_size: int, model_name:str = "model", device: str = 'cpu', lr = 1e-4, logger = None):
        """
        Initializes the diffusion model trainer.

        :param model: The neural network model for the diffusion process.
        :param train_dataloader: DataLoader for the training dataset.
        :param val_dataloader: DataLoader for the validation dataset.
        :param scheduler: The diffusion scheduler object.
        :param batch_size: The size of each data batch.
        :param device: Device to use for training ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.logger = logger
        self.model_name = model_name
        self.checkpoint_dir = f"./ckpt/{self.model_name}/"

        self.before_training_check()

        self.train_loss_dict = {key:[] for key in self.loss_keys}
        self.valid_loss_dict = {key:[] for key in self.loss_keys}
        self.train_loss_dict.update({"epoch":[]})
        self.valid_loss_dict.update({"epoch":[]})

    def loss_fn(self, noise_pred, noise):
        total_loss = self.criterion(noise_pred, noise)
        return {"general_loss":total_loss, "total_loss": total_loss}

    def before_training_check(self):
        if not Path(f"./progress/").exists():
            Path(f"./progress/").mkdir()
        if not Path(self.checkpoint_dir).exists():
            Path(self.checkpoint_dir).mkdir(parents = True)
        for batch in self.train_dataloader:
            loss_dict = self._one_epoch(batch)
            self.loss_keys = list(loss_dict.keys())
            if "total_loss" not in self.loss_keys:
                raise ValueError("remember to put total_loss into loss function")

    def _one_epoch(self, batch):
        x_start = batch["x"].to(self.device)
        z = batch["z"].to(self.device)
        t = torch.randint(0, self.scheduler.num_timesteps, (self.batch_size,), device=self.device)

        # Get alpha_bar(t) for noise scaling
        alpha_bars_sqrt_t = self.scheduler.alpha_bars_sqrt_t[t]
        one_minus_alpha_bars_sqrt_t = self.scheduler.one_minus_alpha_bars_sqrt_t[t]
        noise = torch.randn_like(x_start).to(torch.float32).to(self.device)
        # print(x_start.shape, alpha_bars_sqrt_t.shape, (1 - one_minus_alpha_bars_sqrt_t).shape, noise.shape)
        # x_noisy = alpha_bars_sqrt_t * x_start + one_minus_alpha_bars_sqrt_t * noise
        x_noisy = alpha_bars_sqrt_t.view([-1] + [1 for _ in range(0, x_start.dim()-1)]) * x_start + one_minus_alpha_bars_sqrt_t.view([-1] + [1 for _ in range(0, x_start.dim()-1)]) * noise
        
        # print(x_noisy.shape)
        noise_pred = self.model(x_noisy, t, z)
        loss_dict = self.loss_fn(noise_pred, noise)
        return loss_dict

    def train(self, epochs: int, early_stopping_patience: int = None, save_freq: int = 50):
        """
        Trains the diffusion model for a specified number of epochs.

        :param epochs: Number of training epochs.
        :param checkpoint_path: Path to save model checkpoints.
        :param early_stopping_patience: Number of epochs with no improvement on validation loss before stopping.
        """
        best_val_loss = float('inf')
        patience_counter = 0

        start_epoch = self.load_last_checkpoint()

        for epoch in tqdm(range(start_epoch, epochs)):
            one_epoch_train_loss = {key: 0 for key in self.loss_keys}
            one_epoch_valid_loss = {key: 0 for key in self.loss_keys}

            # epoch_loss = 0
            self.model.train()
            for batch in self.train_dataloader:

                loss_dict = self._one_epoch(batch)
                # epoch_loss += loss_dict['total_loss'].item()

                self.optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                self.optimizer.step()
                for key in self.loss_keys:
                    one_epoch_train_loss[key] += loss_dict[key].item()

            if self.valid_dataloader is not None:
                with torch.no_grad():
                    for batch in self.valid_dataloader:
                        loss_dict = self._one_epoch(batch)
                        # epoch_loss += loss_dict['total_loss'].item()
                        for key in self.loss_keys:
                            one_epoch_valid_loss[key] += loss_dict[key].item()
            self.train_loss_dict['epoch'].append(epoch)
            self.valid_loss_dict['epoch'].append(epoch)
            for key in self.loss_keys:
                self.train_loss_dict[key].append(one_epoch_train_loss[key]/len(self.train_dataloader))
                self.valid_loss_dict[key].append(one_epoch_valid_loss[key]/len(self.valid_dataloader))

            self.plot_loss()
            # Checkpoint saving
            if (((epoch+1) % save_freq) == 0) or (epoch == epochs) :
                self.save_checkpoint(epoch + 1)


    def plot_loss(self):
        fig, ax = plt.subplots(1, len(self.loss_keys), figsize = (5*len(self.loss_keys), 5))
        for index,  key in enumerate(self.loss_keys):
            ax[index].plot(self.train_loss_dict['epoch'], self.train_loss_dict[key], label = "train")
            ax[index].plot(self.valid_loss_dict['epoch'], self.valid_loss_dict[key], label = "valid")
            ax[index].set_title(key)
            ax[index].legend()
            ax[index].grid()
        plt.tight_layout()
        plt.savefig(f"./progress/{self.model_name}.png")
        plt.close()


    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads the model and optimizer state from a specific checkpoint file.

        :param checkpoint_path: Path to the checkpoint file.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler = checkpoint['scheduler_state_dict']
        self.batch_size = checkpoint['batch_size']
        self.train_loss_dict = checkpoint['train_loss_dict']
        self.valid_loss_dict = checkpoint['valid_loss_dict']
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}.")

    def load_last_checkpoint(self):
        """
        Loads the model and optimizer state from the latest checkpoint in the given directory.

        :param checkpoint_dir: Directory where checkpoints are stored.
        """
        if not Path(self.checkpoint_dir).exists():
            # logger.info("No checkpoint directory found")
            print("No checkpoint directory found")
            # print()

        folder_pattern = re.compile(rf'^{re.escape(self.model_name)}_ckpt(\d+)\.pt$')

        # Initialize variables to keep track of the latest epoch and folder
        latest_epoch = -1

        # List all folders and find the one with the highest epoch number
        # for folder in os.listdir(checkpoint_dir):
        match = folder_pattern.match(self.checkpoint_dir)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch

        if latest_epoch != -1:
            latest_checkpoint = self.checkpoint_dir + self.model_name + f"_ckpt{latest_epoch}.pt"
            self.load_checkpoint(latest_checkpoint)
            # logger.info(f"Loaded the latest checkpoint from {latest_checkpoint}.")
            print(f"Loaded the latest checkpoint from {latest_checkpoint}.")
            
            return latest_epoch
        else:
            return 0

    def save_checkpoint(self, epoch: int):
        """
        Saves the model and optimizer state to a checkpoint file.

        :param checkpoint_path: Path to the checkpoint file.
        """
        checkpoint_path = self.checkpoint_dir + self.model_name + f"_ckpt{epoch}.pt"
        torch.save({
            'epoch':epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler,
            'batch_size': self.batch_size,
            'train_loss_dict':self.train_loss_dict,
            'valid_loss_dict':self.valid_loss_dict
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}.")

# Example usage:
# Define your model, train_dataloader, val_dataloader, and scheduler before using the trainer.
# model = YourModel()
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# scheduler = DiffusionScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)

# trainer = DiffusionModelTrainer(model, train_dataloader, val_dataloader, scheduler, batch_size=64, device='cuda')
# trainer.train(epochs=10, checkpoint_path='best_model.pth', early_stopping_patience=3)
# samples = trainer.sample(num_samples=16)