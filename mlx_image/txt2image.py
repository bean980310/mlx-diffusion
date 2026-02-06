"""
MLX Diffusion Text-to-Image Pipeline

This module provides a high-level interface for text-to-image generation
using MLX-optimized diffusion models.
"""

import argparse
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from .core import MLXStableDiffusionPipeline
from .models import UNetWrapper, VAEDecoder, TextEncoder
from .tokenizer import MLXTokenizer, create_tokenizer
from .schedulers import create_scheduler
from .utils import (
    set_seed, 
    tensor2pil, 
    get_memory_usage, 
    cleanup_memory,
    create_progress_bar,
    load_model_from_hub
)


class MLXText2ImagePipeline:
    """Complete MLX text-to-image pipeline with optimizations"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 unet_path: Optional[str] = None,
                 vae_path: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 scheduler_type: str = "ddim",
                 float16: bool = True,
                 quantize: bool = False):
        
        # Set default paths if model_path is provided
        if model_path:
            model_path = Path(model_path)
            unet_path = unet_path or str(model_path / "unet")
            vae_path = vae_path or str(model_path / "vae")
            tokenizer_path = tokenizer_path or str(model_path / "tokenizer")
        
        self.float16 = float16
        self.quantize = quantize
        
        # Initialize pipeline components
        self.pipeline = MLXStableDiffusionPipeline(
            unet_path=unet_path,
            vae_path=vae_path,
            tokenizer_path=tokenizer_path,
            scheduler_type=scheduler_type
        )
        
        # Apply optimizations
        if self.quantize:
            self._apply_quantization()
    
    def _apply_quantization(self):
        """Apply quantization to reduce memory usage"""
        # Quantize text encoder
        if hasattr(self.pipeline.tokenizer, 'text_encoder') and self.pipeline.tokenizer.text_encoder:
            nn.quantize(
                self.pipeline.tokenizer.text_encoder,
                class_predicate=lambda _, m: isinstance(m, nn.Linear)
            )
        
        # Quantize UNet
        nn.quantize(self.pipeline.unet, group_size=32, bits=8)
    
    def generate(self,
                prompt: Union[str, List[str]],
                negative_prompt: Optional[Union[str, List[str]]] = None,
                num_images: int = 1,
                height: int = 512,
                width: int = 512,
                num_inference_steps: int = 50,
                guidance_scale: float = 7.5,
                seed: Optional[int] = None,
                output_path: Optional[str] = None,
                grid_rows: int = 1,
                verbose: bool = False) -> List[Image.Image]:
        """Generate images from text prompts"""
        
        if seed is not None:
            set_seed(seed)
        
        if verbose:
            print(f"Generating {num_images} image(s) from prompt: '{prompt}'")
            print(f"Image size: {width}x{height}")
            print(f"Inference steps: {num_inference_steps}")
            print(f"Guidance scale: {guidance_scale}")
        
        # Generate images using the pipeline
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            return_dict=True
        )
        
        images = result["images"]
        
        # Save to file if output path is provided
        if output_path:
            if num_images > 1 and grid_rows > 1:
                grid_image = self._create_image_grid(images, grid_rows)
                grid_image.save(output_path)
                if verbose:
                    print(f"Saved image grid to: {output_path}")
            else:
                for i, img in enumerate(images):
                    if num_images == 1:
                        img.save(output_path)
                    else:
                        base_path = Path(output_path)
                        save_path = base_path.parent / f"{base_path.stem}_{i+1}{base_path.suffix}"
                        img.save(save_path)
                if verbose:
                    print(f"Saved {len(images)} image(s) to: {output_path}")
        
        # Report memory usage if verbose
        if verbose:
            memory_stats = get_memory_usage()
            print(f"Peak memory: {memory_stats['peak_memory']:.2f}GB")
            print(f"Active memory: {memory_stats['active_memory']:.2f}GB")
        
        return images
    
    def _create_image_grid(self, images: List[Image.Image], rows: int) -> Image.Image:
        """Create a grid from multiple images"""
        cols = (len(images) + rows - 1) // rows
        
        # Get image size (assuming all images are same size)
        img_width, img_height = images[0].size
        
        # Create grid image
        grid_width = cols * img_width
        grid_height = rows * img_height
        grid_image = Image.new('RGB', (grid_width, grid_height))
        
        # Paste images into grid
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            x = col * img_width
            y = row * img_height
            grid_image.paste(img, (x, y))
        
        return grid_image
    
    def cleanup(self):
        """Clean up resources"""
        cleanup_memory()


def create_pipeline_from_pretrained(model_name: str, 
                                  cache_dir: Optional[str] = None,
                                  **kwargs) -> MLXText2ImagePipeline:
    """Create pipeline from pretrained model"""
    
    # Download model files
    model_info = load_model_from_hub(
        repo_id=model_name,
        cache_dir=cache_dir
    )
    
    model_path = model_info.get("local_dir")
    
    return MLXText2ImagePipeline(model_path=model_path, **kwargs)


def main():
    """Command line interface for text-to-image generation"""
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using MLX Diffusion"
    )
    
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--model-path", help="Path to model directory")
    parser.add_argument("--model-name", help="Pretrained model name from HuggingFace")
    parser.add_argument("--unet-path", help="Path to UNet model")
    parser.add_argument("--vae-path", help="Path to VAE model") 
    parser.add_argument("--tokenizer-path", help="Path to tokenizer")
    parser.add_argument("--scheduler", default="ddim", choices=["ddim", "pndm", "lms"],
                       help="Scheduler type")
    
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale for CFG")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    parser.add_argument("--output", default="generated_image.png", help="Output file path")
    parser.add_argument("--grid-rows", type=int, default=1, help="Rows in output grid")
    
    parser.add_argument("--no-float16", dest="float16", action="store_false", 
                       help="Disable float16 optimization")
    parser.add_argument("--quantize", "-q", action="store_true", 
                       help="Enable model quantization")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_path and not args.model_name:
        if not all([args.unet_path, args.vae_path, args.tokenizer_path]):
            parser.error("Must provide either --model-path, --model-name, or all of "
                        "--unet-path, --vae-path, --tokenizer-path")
    
    try:
        # Create pipeline
        if args.model_name:
            pipeline = create_pipeline_from_pretrained(
                model_name=args.model_name,
                scheduler_type=args.scheduler,
                float16=args.float16,
                quantize=args.quantize
            )
        else:
            pipeline = MLXText2ImagePipeline(
                model_path=args.model_path,
                unet_path=args.unet_path,
                vae_path=args.vae_path,
                tokenizer_path=args.tokenizer_path,
                scheduler_type=args.scheduler,
                float16=args.float16,
                quantize=args.quantize
            )
        
        # Generate images
        images = pipeline.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            num_images=args.num_images,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_path=args.output,
            grid_rows=args.grid_rows,
            verbose=args.verbose
        )
        
        print(f"Successfully generated {len(images)} image(s)")
        
        # Clean up
        pipeline.cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
