import importlib
import os

from huggingface_hub.constants import HF_HOME
from packaging import version

from diffusers.dependency_versions_check import dep_version_check
from diffusers.utils.import_utils import ENV_VARS_TRUE_VALUES, is_peft_available, is_transformers_available

from diffusers.utils.constants import (
    MIN_PEFT_VERSION,
    MIN_TRANSFORMERS_VERSION,
    _CHECK_PEFT,
    CONFIG_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    FLAX_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_FILE_EXTENSION,
    GGUF_FILE_EXTENSION,
    ONNX_EXTERNAL_WEIGHTS_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    DIFFUSERS_DYNAMIC_MODULE_NAME,
    HF_MODULES_CACHE,
    DEPRECATED_REVISION_ARGS,
    DIFFUSERS_REQUEST_TIMEOUT,
    _required_peft_version,
    _required_transformers_version,
    USE_PEFT_BACKEND,
    DECODE_ENDPOINT_SD_V1,
    DECODE_ENDPOINT_SD_XL,
    DECODE_ENDPOINT_FLUX,
    DECODE_ENDPOINT_HUNYUAN_VIDEO,
    ENCODE_ENDPOINT_SD_V1,
    ENCODE_ENDPOINT_SD_XL,
    ENCODE_ENDPOINT_FLUX
)

MLX_SAFETENSORS_WEIGHTS_NAME = "diffusion_mlx_model.safetensors"
MLX_WEIGHTS_INDEX_NAME = "diffusion_mlx_model.safetensors.index.json"

from . import __version__

from typing import List, Optional, Tuple, Union
import mlx.core as mx

import torch
from diffusers.utils import logging

logger = logging.get_logger(__name__)

def maybe_allow_in_graph(cls):
    return cls

import random
import numpy as np
from PIL import Image

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    
def tensor2pil(tensor: torch.Tensor):
    arr=tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    arr=(arr*255).round().astype(np.uint8)
    return Image.fromarray(arr)