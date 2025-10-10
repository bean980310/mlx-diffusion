import gc
import os
import json
import math
import torch

import numpy as np
import mlx.nn as nn
import mlx.core as mx
import mlx.utils as mxutils

import numpy as np
from PIL import Image
from tqdm import tqdm

from safetensors.torch import load_file
from ...models.stable_diffusion.tokenizers import CLIPTokenizer
from ...models.stable_diffusion.layers import CLIP, Diffusion, Encoder, Decoder
from ...models.stable_diffusion.scheduler import DDPMSampler, EulerAncestralSampler
from ...models.stable_diffusion.mappings import simplified_clip_lora_mapping, simplified_unet_lora_mapping
from ...models.stable_diffusion.loaders import load_clip_weights, load_clip_lora, load_unet_weights, load_unet_lora, load_decoder_weights,load_encoder_weights