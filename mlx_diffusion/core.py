from typing import Optional, List, Union, Dict, Any
import mlx.core as mx
import mlx.nn as nn
from abc import ABC, abstractmethod

from .models import UNetWrapper, VAEDecoder
from .tokenizer import MLXTokenizer
from .schedulers import DDIMScheduler, PNDMScheduler, LMSScheduler
from .utils import set_seed, tensor2pil


class MLXDiffusionPipeline(ABC):
    """Base class for MLX diffusion pipelines"""
    
    def __init__(self, 
                 unet_path: str,
                 vae_path: str, 
                 tokenizer_path: str,
                 scheduler_type: str = "ddim",
                 safety_checker=None):
        
        self.unet = UNetWrapper(unet_path)
        self.vae = VAEDecoder(vae_path)
        self.tokenizer = MLXTokenizer(tokenizer_path)
        self.scheduler = self._load_scheduler(scheduler_type)
        self.safety_checker = safety_checker
        
    def _load_scheduler(self, scheduler_type: str):
        """Load the appropriate scheduler"""
        if scheduler_type.lower() == "ddim":
            return DDIMScheduler()
        elif scheduler_type.lower() == "pndm":
            return PNDMScheduler()
        elif scheduler_type.lower() == "lms":
            return LMSScheduler()
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Main pipeline call method"""
        pass
    
    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        """Enable sliced attention computation for memory efficiency"""
        pass
    
    def disable_attention_slicing(self):
        """Disable sliced attention computation"""
        pass


class MLXStableDiffusionPipeline(MLXDiffusionPipeline):
    """MLX-based Stable Diffusion text-to-image pipeline"""
    
    def __call__(self, 
                 prompt: Union[str, List[str]],
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 height: int = 512,
                 width: int = 512,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 num_images_per_prompt: int = 1,
                 eta: float = 0.0,
                 generator: Optional[mx.random.state] = None,
                 latents: Optional[mx.array] = None,
                 output_type: str = "pil",
                 return_dict: bool = True,
                 callback = None,
                 callback_steps: int = 1,
                 **kwargs):
        
        # Handle batch processing
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)
            
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
        else:
            negative_prompt = [""] * batch_size
            
        # Set random seed if generator provided
        if generator is not None:
            mx.random.seed(generator)
            
        # Encode text prompts
        text_embeddings = self._encode_prompt(prompt, negative_prompt, guidance_scale > 1.0)
        
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Prepare latents
        num_channels_latents = self.unet.in_channels
        latents = self._prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            latents
        )
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = mx.concatenate([latents] * 2, axis=0) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
            
            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = mx.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute the previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, eta=eta)
            
            # Call callback if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        # Decode latents to images
        image = self.vae.decode(latents / 0.18215)
        image = self._postprocess_image(image, output_type)
        
        if not return_dict:
            return (image,)
        
        return {"images": image}
    
    def _encode_prompt(self, prompt: List[str], negative_prompt: List[str], do_classifier_free_guidance: bool):
        """Encode text prompts to embeddings"""
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="mlx"
        ).input_ids
        
        text_embeddings = self.tokenizer.text_encoder(text_input_ids)
        
        if do_classifier_free_guidance:
            uncond_input_ids = self.tokenizer(
                negative_prompt,
                padding="max_length", 
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="mlx"
            ).input_ids
            
            uncond_embeddings = self.tokenizer.text_encoder(uncond_input_ids)
            text_embeddings = mx.concatenate([uncond_embeddings, text_embeddings], axis=0)
            
        return text_embeddings
    
    def _prepare_latents(self, batch_size: int, num_channels: int, height: int, width: int, dtype, latents=None):
        """Prepare initial latents"""
        shape = (batch_size, num_channels, height // 8, width // 8)
        
        if latents is None:
            latents = mx.random.normal(shape, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Expected latents shape {shape}, got {latents.shape}")
                
        # Scale initial noise by scheduler's init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def _postprocess_image(self, image: mx.array, output_type: str):
        """Post-process generated images"""
        if output_type == "pil":
            return tensor2pil(image)
        elif output_type == "mlx":
            return image
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
        

