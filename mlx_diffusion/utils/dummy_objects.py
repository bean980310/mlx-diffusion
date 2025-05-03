from diffusers.utils import DummyObject, requires_backends

class MLXModelMixin(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXConfigMixin(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXFromOriginalModelMixin(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXPeftAdapterMixin(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
class MLXModuleUtilsMixin(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
class MLXGenerationMixin(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXPushToHubMixin(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])

class MLXSpecialTokensMixin(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXUNet2DConditionLoadersMixin(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXImageProcessingMixin(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXPreTrainedModel(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])

class MLXPreTrainedTokenizerBase(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXPreTrainedTokenizer(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXBaseImageProcessor(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXCLIPPreTrainedModel(metaclass=DummyObject):
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
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXKarrasDiffusionSchedulers(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXCLIPVisionModelWithProjection(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXCLIPTextModel(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXCLIPTokenizer(metaclass=DummyObject):
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
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])

class MLXStableDiffusionSafetyChecker(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
        
class MLXCLIPImageProcessor(metaclass=DummyObject):
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
class MLXCLIPVisionModelWithProjection(metaclass=DummyObject):
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
    _backends=["mlx"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["mlx"])
        
    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["mlx"])
        