import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Union

import numpy as np
import mlx.core as mx
import safetensors
import torch
import torch.nn.functional as F
import mlx.nn as nn
import mlx.nn.layers as nn_layers
from huggingface_hub.utils import validate_hf_hub_args

from diffusers.models.embeddings import (
    ImageProjection,
    IPAdapterFaceIDImageProjection,
    IPAdapterFaceIDPlusImageProjection,
    IPAdapterFullImageProjection,
    IPAdapterPlusImageProjection,
    MultiIPAdapterImageProjection,
)