import os
import json
import random
import importlib
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from tqdm import tqdm

import numpy as np
import mlx.core as mx
from PIL import Image, ImageOps
import requests
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

# Import constants and versions for compatibility
from . import __version__

# MLX-specific constants
MLX_SAFETENSORS_WEIGHTS_NAME = "diffusion_mlx_model.safetensors"
MLX_WEIGHTS_INDEX_NAME = "diffusion_mlx_model.safetensors.index.json"
CONFIG_NAME = "config.json"


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def tensor2pil(tensor: mx.array) -> List[Image.Image]:
    """Convert MLX tensor to PIL images"""
    if len(tensor.shape) == 3:
        # Single image: [C, H, W]
        tensor = tensor[None, :]  # Add batch dimension
    
    # Expect: [B, C, H, W]
    images = []
    for i in range(tensor.shape[0]):
        # Convert from [-1, 1] to [0, 1]
        image = (tensor[i] + 1.0) / 2.0
        # Clamp to valid range
        image = mx.clip(image, 0.0, 1.0)
        # Convert to numpy and transpose to [H, W, C]
        image_np = np.array(image.transpose(1, 2, 0))
        # Scale to [0, 255] and convert to uint8
        image_np = (image_np * 255).round().astype(np.uint8)
        # Create PIL image
        pil_image = Image.fromarray(image_np)
        images.append(pil_image)
    
    return images


def pil2tensor(images: Union[Image.Image, List[Image.Image]]) -> mx.array:
    """Convert PIL images to MLX tensor"""
    if isinstance(images, Image.Image):
        images = [images]
    
    tensors = []
    for img in images:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize to [-1, 1]
        img_array = img_array.astype(np.float32) / 255.0
        img_array = img_array * 2.0 - 1.0
        
        # Transpose to [C, H, W]
        img_array = img_array.transpose(2, 0, 1)
        
        # Convert to MLX
        tensor = mx.array(img_array)
        tensors.append(tensor)
    
    # Stack batch dimension
    return mx.stack(tensors, axis=0)


def resize_image(image: Image.Image, 
                size: Union[int, Tuple[int, int]], 
                resample: int = Image.LANCZOS) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    if isinstance(size, int):
        # Resize shorter side to size
        w, h = image.size
        if w < h:
            new_w, new_h = size, int(size * h / w)
        else:
            new_w, new_h = int(size * w / h), size
        size = (new_w, new_h)
    
    return image.resize(size, resample=resample)


def center_crop(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Center crop image to specified size"""
    return ImageOps.fit(image, size, method=Image.LANCZOS, centering=(0.5, 0.5))


def preprocess_image(image: Union[str, Image.Image], 
                    height: int = 512, 
                    width: int = 512) -> mx.array:
    """Preprocess image for diffusion model input"""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be PIL Image or file path")
    
    # Resize and crop
    image = resize_image(image, min(height, width))
    image = center_crop(image, (width, height))
    
    # Convert to tensor
    return pil2tensor(image)


def postprocess_image(image: mx.array, 
                     output_type: str = "pil") -> Union[mx.array, List[Image.Image]]:
    """Postprocess model output to desired format"""
    if output_type == "pil":
        return tensor2pil(image)
    elif output_type == "mlx":
        return image
    elif output_type == "np":
        return np.array(image)
    else:
        raise ValueError(f"Unsupported output_type: {output_type}")


def load_model_from_hub(repo_id: str, 
                       filename: Optional[str] = None,
                       cache_dir: Optional[str] = None,
                       force_download: bool = False,
                       resume_download: bool = True,
                       token: Optional[str] = None) -> Dict[str, Any]:
    """Load model weights from Hugging Face Hub"""
    
    if filename is None:
        # Try to download the whole repository
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            token=token
        )
        return {"local_dir": local_dir}
    else:
        # Download specific file
        local_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            token=token
        )
        return {"local_file": local_file}


def load_safetensors_weights(file_path: str) -> Dict[str, mx.array]:
    """Load weights from safetensors file"""
    weights = {}
    
    with safe_open(file_path, framework="numpy") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key))
    
    return weights


def save_safetensors_weights(weights: Dict[str, mx.array], file_path: str):
    """Save MLX weights to safetensors format"""
    from safetensors.numpy import save_file
    
    # Convert MLX arrays to numpy
    numpy_weights = {}
    for key, tensor in weights.items():
        numpy_weights[key] = np.array(tensor)
    
    save_file(numpy_weights, file_path)


def convert_diffusers_to_mlx(diffusers_model_path: str, 
                           mlx_output_path: str,
                           model_type: str = "unet") -> str:
    """Convert Diffusers checkpoint to MLX format"""
    
    if not os.path.exists(diffusers_model_path):
        raise FileNotFoundError(f"Diffusers model not found: {diffusers_model_path}")
    
    # Load original weights
    if model_type == "unet":
        from diffusers import UNet2DConditionModel
        model = UNet2DConditionModel.from_pretrained(diffusers_model_path)
    elif model_type == "vae":
        from diffusers import AutoencoderKL
        model = AutoencoderKL.from_pretrained(diffusers_model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Convert to MLX format
    mlx_weights = {}
    for name, param in model.named_parameters():
        mlx_weights[name] = mx.array(param.detach().cpu().numpy())
    
    # Save as safetensors
    os.makedirs(os.path.dirname(mlx_output_path), exist_ok=True)
    save_safetensors_weights(mlx_weights, mlx_output_path)
    
    # Save config
    config_path = os.path.join(os.path.dirname(mlx_output_path), "config.json")
    with open(config_path, 'w') as f:
        json.dump(model.config, f, indent=2)
    
    return mlx_output_path


def create_progress_bar(total: int, desc: str = "Processing") -> tqdm:
    """Create a progress bar for long operations"""
    return tqdm(total=total, desc=desc, unit="step")


def randn_tensor(shape: Tuple[int, ...], 
                dtype: mx.Dtype = mx.float32,
                generator: Optional[mx.random.state] = None) -> mx.array:
    """Generate random normal tensor"""
    if generator is not None:
        mx.random.seed(generator)
    return mx.random.normal(shape, dtype=dtype)


def get_memory_usage() -> Dict[str, float]:
    """Get MLX memory usage statistics"""
    return {
        "peak_memory": mx.metal.get_peak_memory() / 1024**3,  # GB
        "active_memory": mx.metal.get_active_memory() / 1024**3,  # GB
        "cache_memory": mx.metal.get_cache_memory() / 1024**3  # GB
    }


def cleanup_memory():
    """Clean up MLX memory cache"""
    mx.metal.clear_cache()


class DiffusionImageProcessor:
    """Image processor for diffusion pipelines"""
    
    def __init__(self, vae_scale_factor: int = 8):
        self.vae_scale_factor = vae_scale_factor
    
    def preprocess(self, 
                  images: Union[Image.Image, List[Image.Image]],
                  height: int = 512,
                  width: int = 512) -> mx.array:
        """Preprocess images for model input"""
        if isinstance(images, Image.Image):
            images = [images]
        
        processed_images = []
        for img in images:
            # Resize and crop
            img = resize_image(img, min(height, width))
            img = center_crop(img, (width, height))
            processed_images.append(img)
        
        return pil2tensor(processed_images)
    
    def postprocess(self, 
                   images: mx.array,
                   output_type: str = "pil",
                   do_denormalize: List[bool] = None) -> Union[mx.array, List[Image.Image]]:
        """Postprocess model output"""
        if do_denormalize is None:
            do_denormalize = [True] * images.shape[0]
        
        if output_type == "mlx":
            return images
        
        # Convert to PIL
        return tensor2pil(images)


# Utility function to check if model files exist
def check_model_files(model_path: str, 
                     required_files: List[str] = None) -> bool:
    """Check if all required model files exist"""
    if required_files is None:
        required_files = [MLX_SAFETENSORS_WEIGHTS_NAME, CONFIG_NAME]
    
    model_path = Path(model_path)
    
    for file in required_files:
        if not (model_path / file).exists():
            return False
    
    return True


# Factory function to create image processor
def create_image_processor(vae_scale_factor: int = 8) -> DiffusionImageProcessor:
    """Create image processor with specified VAE scale factor"""
    return DiffusionImageProcessor(vae_scale_factor=vae_scale_factor)