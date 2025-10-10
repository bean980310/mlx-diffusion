import os

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mlx_diffusion")

def get_cache_dir():
    return os.getenv('MLX_DIFFUSION_CACHE_DIR', DEFAULT_CACHE_DIR)