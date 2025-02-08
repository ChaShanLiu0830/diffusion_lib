
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
import pandas as pd 
import numpy as np 


class BaseSampler:
    def __init__(self, model, scheduler, test_dataloader, device='cpu'):
        """
        Initializes the diffusion model sampler.

        :param model: The trained diffusion model.
        :param scheduler: The diffusion scheduler object.
        :param device: Device to use for sampling ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
        self.dataloader = test_dataloader

    def _sample(self, batch, eta):
        x_t = torch.randn_like(batch['x'], device = self.device)
        z = batch["z"].to(self.device)

        for t in reversed(range(1, self.scheduler.num_timesteps - 1)):
            # Predict noise using the model
            t = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long).to(self.device)
            noise_pred = self.model(x_t, t, z)
            # alpha_t = self.scheduler.alphas[t]
            # alpha_t_next = self.scheduler.alphas[t - 1]

            alpha_bar_t = self.scheduler.alpha_bars[t]
            alpha_bar_t_next = self.scheduler.alpha_bars[t -1]


            # DDIM sampling update
            x0_pred = (x_t - noise_pred * self.scheduler.one_minus_alpha_bars_sqrt_t[t]) / self.scheduler.alpha_bars_sqrt_t[t]

            # print(alpha_bar_t, alpha_bar_t_next)
            c2 = eta * ((1-alpha_bar_t_next) / (1 -alpha_bar_t)).sqrt() * (1 - alpha_bar_t / alpha_bar_t_next).sqrt()
            c1 = torch.sqrt(1 - alpha_bar_t_next - c2**2)

            if (torch.sum(torch.isnan(x0_pred)).item() >0) or (torch.sum(torch.isnan(c1))).item() >0 :
                print(t[0], self.scheduler.one_minus_alpha_bars_sqrt_t[t][0], self.scheduler.alpha_bars_sqrt_t[t].sqrt()[0], self.scheduler.one_minus_alpha_bars_sqrt_t[t][0]/alpha_bar_t[0])
                print(c1[0])
                print(c2[0])

                raise ValueError("x0_pred is nan")

            x_t = alpha_bar_t_next.sqrt() * x0_pred + c2 * torch.randn_like(x_t) + c1 * noise_pred
            # print(t[0], x_t[0, -1], alpha_bar_t_next.sqrt()[0], c1[0], c2[0])
        return x_t


    def sample(self, num_samples: int, save_dir: str = None, file_name: str = 'sample', eta = 0, store_feat = []):
        """
        Samples new data from the trained diffusion model.

        :param num_samples: Number of samples to generate.
        :param img_shape: The shape of the output images (channels, height, width).
        :param save_dir: Directory to save the generated samples. If None, samples are not saved.
        :param filename_prefix: Prefix for the filenames if saving the samples.
        :return: Generated samples as a tensor of shape (num_samples, *img_shape).
        """
        self.model.eval()
        with torch.no_grad():
            gen_data = {feat:[] for feat in store_feat}
            gen_data['gen_data'] = []

            for batch in tqdm(self.dataloader, desc= "Sampling"):
                temp_gen_data = []
                for sample in range(num_samples):
                    x_0 = self._sample(batch = batch, eta = eta)
                    temp_gen_data.append(x_0.unsqueeze(-1))
                temp_gen_data = torch.cat(temp_gen_data, dim = -1)
                # print(temp_gen_data)
                gen_data['gen_data'].append(temp_gen_data.detach().cpu().numpy())
                for feat in store_feat:
                    gen_data[feat].append(batch[feat].detach().cpu().numpy())

            for key in gen_data.keys():
                gen_data[key] = np.concatenate(gen_data[key], axis = 0)
            # print(gen_data['gen_data'])
            # samples = gen_data['gen_data']

            # sample_path = save_dir + file_name
            # Save the samples if a save directory is provided
            if save_dir is None:
                save_dir = "./gen_data/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                # for i, sample in enumerate(samples):
                #     sample_path = os.path.join(save_dir, f"{filename_prefix}_{i}.pt")
                #     torch.save(sample.cpu(), sample_path)
            pd.to_pickle(gen_data, f"{save_dir}{file_name}.pkl")
            # logger.info(f"Sample saved to {save_dir}{file_name}.pkl")
            print(f"Sample saved to {save_dir}{file_name}.pkl")

            return gen_data

    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads the model state from a checkpoint file.

        :param checkpoint_path: Path to the checkpoint file.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # logger.info(f"Model loaded from checkpoint {checkpoint_path}.")
        print(f"Model loaded from checkpoint {checkpoint_path}.")
