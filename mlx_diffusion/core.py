from typing import Optional, List, Union
import torch

from .models import UNetWrapper, VAEDecoder
from .tokenizer import Tokenizer
from .schedulers import DDIMScheduler
from .utils import set_seed, tensor2pil

class MLXStableDiffusionPipeline:
    def __init__(self, unet: str, vae: str, tokenizer: str, scheduler: str="ddim"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = Tokenizer(tokenizer).to(self.device)
        self.unet = UNetWrapper(unet).to(self.device)
        self.vae = VAEDecoder(vae).to(self.device)
        self.scheduler = DDIMScheduler().to(self.device)
        
    def generate(self, prompt: str, num_steps: int=50, cfg_scale: float=7.5, height: int=512, width: int=512, seed: Optional[int]=None):
        if seed is not None:
            set_seed(seed)
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt).to(self.device)
        
        # Initialize latents
        latents = self.scheduler.init_latents(batch_size=1, channels=self.unet.in_channels, height=height//8, width=width//8, device=self.device)
        
        # Generate images
        for t in self.scheduler.timesteps(num_steps):
            noise_pred = self.unet(latents, t, encoder_hidden_states=input_ids)
            latents = self.scheduler.step(noise_pred, latents, t, cfg_scale)
        
        # Decode the latents to images
        image = self.vae.decode(latents)
        pil = tensor2pil(image)
        
        return pil
        

