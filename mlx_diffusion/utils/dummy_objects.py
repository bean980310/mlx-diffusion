from diffusers.utils import DummyObject, requires_backends

class MLXModelMixin(metaclass=DummyObject):
    """
    Mixin class for MLX models.
    """
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXUNet2DConditionModel(metaclass=DummyObject):
    """
    Mixin class for MLX UNet2DCondition models.
    """
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXAutoencoder(metaclass=DummyObject):
    """
    Mixin class for MLX Autoencoder models.
    """
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXAutoencoderKL(metaclass=DummyObject):
    """
    Mixin class for MLX AutoencoderKL models.
    """
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXStableDiffusionPipeline(metaclass=DummyObject):
    """
    Mixin class for MLX StableDiffusionPipeline models.
    """
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXStableDiffusionXLPipeline(metaclass=DummyObject):
    """
    Mixin class for MLX StableDiffusionXLPipeline models.
    """
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXStableDiffusion3Pipeline(metaclass=DummyObject):
    """
    Mixin class for MLX StableDiffusion3Pipeline models.
    """
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXStableDiffusion3Pipeline(metaclass=DummyObject):
    """
    Mixin class for MLX StableDiffusion3Pipeline models.
    """
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])