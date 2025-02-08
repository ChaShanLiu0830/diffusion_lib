import torch
import matplotlib.pyplot as plt
import numpy as np

class BaseScheduler:
    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float, device: str):
        """
        Initializes the diffusion scheduler.

        :param num_timesteps: Total number of timesteps in the diffusion process.
        :param beta_start: Initial value of beta (variance schedule).
        :param beta_end: Final value of beta.
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self._linear_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        self.alpha_bars_sqrt_t = np.sqrt(self.alpha_bars)
        self.one_minus_alpha_bars_sqrt_t = np.sqrt(1 - self.alpha_bars)
        self.to(device)
    def to(self, device = "cpu"):

        self.betas = torch.FloatTensor(self.betas).to(device).to(torch.float32).unsqueeze(-1)
        self.alphas = torch.FloatTensor(self.alphas).to(device).to(torch.float32).unsqueeze(-1)
        self.alpha_bars = torch.FloatTensor(self.alpha_bars).to(torch.float32).to(device).unsqueeze(-1)
        self.alpha_bars_sqrt_t = torch.FloatTensor(self.alpha_bars_sqrt_t).to(torch.float32).to(device).unsqueeze(-1)
        self.one_minus_alpha_bars_sqrt_t = torch.FloatTensor(self.one_minus_alpha_bars_sqrt_t).to(torch.float32).to(device).unsqueeze(-1)
    def _linear_beta_schedule(self):
        """
        Creates a linearly increasing schedule for beta.
        """
        return np.linspace(self.beta_start, self.beta_end, self.num_timesteps)

    def get_variance(self, t: int):
        """
        Gets the variance (beta) at timestep t.

        :param t: Timestep index.
        :return: Variance at timestep t.
        """
        return self.betas[t]

    def get_alpha(self, t: int):
        """
        Gets the alpha (1 - beta) at timestep t.

        :param t: Timestep index.
        :return: Alpha at timestep t.
        """
        return self.alphas[t]

    def get_alpha_bar(self, t: int):
        """
        Gets the cumulative product of alphas (alpha_bar) up to timestep t.

        :param t: Timestep index.
        :return: Alpha_bar at timestep t.
        """
        return self.alpha_bars[t]

    def get_posterior_mean_variance(self, x_start, x_t, t):
        """
        Calculates the posterior mean and variance for the reverse process.

        :param x_start: The initial (starting) data.
        :param x_t: The data at timestep t.
        :param t: Timestep index.
        :return: Tuple of (posterior mean, posterior variance).
        """
        posterior_mean = (self.get_alpha_bar(t - 1) / self.get_alpha_bar(t)) * x_start + \
                         ((1 - self.get_alpha_bar(t - 1)) / self.get_alpha_bar(t)) * x_t
        posterior_variance = (1 - self.get_alpha_bar(t - 1)) / self.get_alpha_bar(t)
        return posterior_mean, posterior_variance

    def plot_scheduler(self):
        plt.plot(self.alphas, label = "alpahs")
        plt.plot(self.betas, label = "betas")
        plt.plot(self.alpha_bars_sqrt_t, label = "alpha_bars_sqrt_t")
        plt.plot(self.one_minus_alpha_bars_sqrt_t, label = "one_minus_alpha_bars_sqrt_t")
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()


# class BaseScheduler:
#     def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02, device='cpu'):
#         """
#         Initialize the scheduler with a linear noise schedule.

#         :param num_timesteps: Number of timesteps.
#         :param beta_start: Starting beta value.
#         :param beta_end: Ending beta value.
#         :param device: Device to store tensors ('cpu' or 'cuda').
#         """
#         self.num_timesteps = num_timesteps
#         self.device = device
#         self.beta_schedule = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
#         self.alpha_schedule = 1.0 - self.beta_schedule
#         self.alpha_bar_schedule = torch.cumprod(self.alpha_schedule, dim=0)

#         self.alpha_bars_sqrt_t = torch.sqrt(self.alpha_bar_schedule).to(device)
#         self.one_minus_alpha_bars_sqrt_t = torch.sqrt(1 - self.alpha_bar_schedule).to(device)

#     def get_alpha_bar_sqrt(self, t):
#         """
#         Get sqrt(alpha_bar) for timestep t.

#         :param t: Timestep.
#         :return: sqrt(alpha_bar(t)).
#         """
#         return self.alpha_bars_sqrt_t[t]

#     def get_one_minus_alpha_bar_sqrt(self, t):
#         """
#         Get sqrt(1 - alpha_bar) for timestep t.

#         :param t: Timestep.
#         :return: sqrt(1 - alpha_bar(t)).
#         """
#         return self.one_minus_alpha_bars_sqrt_t[t]
#     def get_plot(self):
#         plt.plot(self.alpha_bars_sqrt_t.cpu().numpy(), label = "alpha_bars_sqrt_t")
#         plt.plot(self.one_minus_alpha_bars_sqrt_t.cpu().numpy(), label = "one_minus_alpha_bars_sqrt_t")
#         plt.plot(self.beta_schedule.cpu().numpy(), label = "beta_t")
#         plt.grid()
#         plt.title("scheduler parameter")
        
    