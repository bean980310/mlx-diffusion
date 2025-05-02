import torch
from diffusers import UNet2DConditionModel
import mlx.core as mx

class UNetWrapper(mx.Model):
    def __init__(self, ckpt_path: str):
        super().__init__()
        self.model = mx.load(ckpt_path)
        self.in_channels = self.model.config["in_channels"]
        self.out_channels = self.model.config["out_channels"]
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, encoder_hidden_states: int=None):
        # Call the original UNet forward method
        return self.model(x, t, encoder_hidden_states=encoder_hidden_states)
    
class VAEDecoder(mx.Model):
    def __init__(self, ckpt_path: str):
        super().__init__()
        self.model = mx.load(ckpt_path)

    def forward(self, x: torch.Tensor):
        # Ensure the input tensor is on the same device as the model
        
        # Call the original VAE decoder forward method
        return self.model.decode(x)