#!/usr/bin/env python3
"""
Example: Text-to-Image Generation with MLX Diffusion

This example demonstrates how to use the MLX Diffusion library
to generate images from text prompts.
"""

import os
import sys
from pathlib import Path

# Add mlx_diffusion to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_diffusion.txt2image import MLXText2ImagePipeline, create_pipeline_from_pretrained
from mlx_diffusion.utils import set_seed, get_memory_usage, cleanup_memory


def basic_example():
    """Basic text-to-image generation example"""
    print("=== Basic Text-to-Image Example ===")
    
    # Note: This example assumes you have model files available
    # In practice, you would specify actual paths to your model files
    
    try:
        # Create pipeline with placeholder paths
        # Replace these with actual model paths
        pipeline = MLXText2ImagePipeline(
            unet_path="path/to/unet/model",  # Replace with actual path
            vae_path="path/to/vae/model",    # Replace with actual path
            tokenizer_path="path/to/tokenizer",  # Replace with actual path
            scheduler_type="ddim",
            quantize=True  # Enable quantization for memory efficiency
        )
        
        # Generate a single image
        images = pipeline.generate(
            prompt="A beautiful landscape with mountains and a lake",
            negative_prompt="blurry, low quality, distorted",
            num_images=1,
            height=512,
            width=512,
            num_inference_steps=25,
            guidance_scale=7.5,
            seed=42,
            verbose=True
        )
        
        # Save the image
        output_path = "generated_landscape.png"
        images[0].save(output_path)
        print(f"Image saved to: {output_path}")
        
        # Clean up
        pipeline.cleanup()
        
    except Exception as e:
        print(f"Error in basic example: {e}")
        print("Make sure to update model paths in the example!")


def batch_generation_example():
    """Example of generating multiple images"""
    print("\n=== Batch Generation Example ===")
    
    try:
        pipeline = MLXText2ImagePipeline(
            unet_path="path/to/unet/model",
            vae_path="path/to/vae/model", 
            tokenizer_path="path/to/tokenizer",
            scheduler_type="ddim",
            quantize=True
        )
        
        # Generate multiple images with different prompts
        prompts = [
            "A cat sitting in a garden",
            "A futuristic city at sunset", 
            "An abstract painting with vibrant colors",
            "A peaceful forest scene"
        ]
        
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/4: '{prompt}'")
            
            images = pipeline.generate(
                prompt=prompt,
                negative_prompt="low quality, blurry",
                num_images=1,
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42 + i,  # Different seed for each image
                verbose=False
            )
            
            # Save image
            output_path = f"batch_image_{i+1}.png"
            images[0].save(output_path)
            print(f"Saved: {output_path}")
        
        pipeline.cleanup()
        print("Batch generation completed!")
        
    except Exception as e:
        print(f"Error in batch generation: {e}")


def grid_generation_example():
    """Example of generating a grid of images"""
    print("\n=== Grid Generation Example ===")
    
    try:
        pipeline = MLXText2ImagePipeline(
            unet_path="path/to/unet/model",
            vae_path="path/to/vae/model",
            tokenizer_path="path/to/tokenizer", 
            scheduler_type="ddim"
        )
        
        # Generate multiple images at once
        images = pipeline.generate(
            prompt="A cute robot in different poses",
            negative_prompt="low quality, blurry, distorted",
            num_images=4,
            height=512,
            width=512,
            num_inference_steps=25,
            guidance_scale=7.5,
            seed=123,
            output_path="robot_grid.png",
            grid_rows=2,  # 2x2 grid
            verbose=True
        )
        
        print(f"Generated grid with {len(images)} images")
        pipeline.cleanup()
        
    except Exception as e:
        print(f"Error in grid generation: {e}")


def scheduler_comparison_example():
    """Example comparing different schedulers"""
    print("\n=== Scheduler Comparison Example ===")
    
    schedulers = ["ddim", "pndm", "lms"]
    prompt = "A serene lake with mountains in the background"
    
    for scheduler in schedulers:
        try:
            print(f"Testing {scheduler.upper()} scheduler...")
            
            pipeline = MLXText2ImagePipeline(
                unet_path="path/to/unet/model",
                vae_path="path/to/vae/model",
                tokenizer_path="path/to/tokenizer",
                scheduler_type=scheduler
            )
            
            images = pipeline.generate(
                prompt=prompt,
                num_images=1,
                height=512,
                width=512,
                num_inference_steps=25,
                guidance_scale=7.5,
                seed=456,  # Same seed for fair comparison
                verbose=False
            )
            
            output_path = f"scheduler_comparison_{scheduler}.png"
            images[0].save(output_path)
            print(f"Saved {scheduler} result: {output_path}")
            
            pipeline.cleanup()
            
        except Exception as e:
            print(f"Error with {scheduler} scheduler: {e}")


def memory_monitoring_example():
    """Example demonstrating memory usage monitoring"""
    print("\n=== Memory Monitoring Example ===")
    
    try:
        print("Initial memory usage:")
        memory_stats = get_memory_usage()
        for key, value in memory_stats.items():
            print(f"  {key}: {value:.2f}GB")
        
        pipeline = MLXText2ImagePipeline(
            unet_path="path/to/unet/model",
            vae_path="path/to/vae/model",
            tokenizer_path="path/to/tokenizer",
            quantize=True  # Enable quantization to reduce memory usage
        )
        
        print("\nMemory after loading models:")
        memory_stats = get_memory_usage()
        for key, value in memory_stats.items():
            print(f"  {key}: {value:.2f}GB")
        
        # Generate image
        images = pipeline.generate(
            prompt="A detailed architectural drawing",
            num_images=1,
            num_inference_steps=30,
            verbose=False
        )
        
        print("\nMemory after generation:")
        memory_stats = get_memory_usage()
        for key, value in memory_stats.items():
            print(f"  {key}: {value:.2f}GB")
        
        # Clean up
        pipeline.cleanup()
        cleanup_memory()
        
        print("\nMemory after cleanup:")
        memory_stats = get_memory_usage()
        for key, value in memory_stats.items():
            print(f"  {key}: {value:.2f}GB")
            
    except Exception as e:
        print(f"Error in memory monitoring: {e}")


def pretrained_model_example():
    """Example using a pretrained model from HuggingFace (if available)"""
    print("\n=== Pretrained Model Example ===")
    
    try:
        # This would download a model from HuggingFace Hub
        # Note: Replace with actual model name when available
        model_name = "runwayml/stable-diffusion-v1-5"  # Example model name
        
        print(f"Loading pretrained model: {model_name}")
        pipeline = create_pipeline_from_pretrained(
            model_name=model_name,
            scheduler_type="ddim",
            quantize=True
        )
        
        images = pipeline.generate(
            prompt="A majestic eagle soaring over mountains",
            negative_prompt="blurry, low quality",
            num_images=1,
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=7.5,
            seed=789,
            output_path="pretrained_eagle.png",
            verbose=True
        )
        
        pipeline.cleanup()
        print("Pretrained model example completed!")
        
    except Exception as e:
        print(f"Error with pretrained model: {e}")
        print("This is expected if the model is not available or paths are not configured")


def main():
    """Run all examples"""
    print("MLX Diffusion Examples")
    print("=====================")
    print()
    print("Note: These examples use placeholder model paths.")
    print("Update the paths to point to your actual model files before running.")
    print()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Run examples
    try:
        basic_example()
        batch_generation_example() 
        grid_generation_example()
        scheduler_comparison_example()
        memory_monitoring_example()
        pretrained_model_example()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up any remaining resources
        cleanup_memory()
        print("\nAll examples completed!")


if __name__ == "__main__":
    main()