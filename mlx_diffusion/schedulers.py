import torch
import numpy as np

class DDIMScheduler:
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = np.concatenate(([1.0], self.alpha_cumprod[:-1]))
        
    def init_latents(self, batch_size: int, channels: int, height: int, width: int, device: str):
        # Initialize latents with random noise
        return torch.randn(batch_size, channels, height, width, device=device)
    
    def timesteps(self, num_steps: int):
        return np.linspace(self.alpha_cumprod.shape[0]-1,0,num_steps, dtype=int)
    
    def step(self, noise_pred, latents, t, cfg_scale):
        alpha_prod=self.alpha_cumprod[t]
        sqrt_alpha=alpha_prod.sqrt()
        sqrt_one_minus_alpha=(1-alpha_prod).sqrt()
        pred_orig=(latents-sqrt_one_minus_alpha*noise_pred)/sqrt_alpha
        direction=((1-alpha_prod).sqrt())*noise_pred
        prev_latents=sqrt_alpha*pred_orig+direction
        return prev_latents