from collections import OrderedDict
from dataclasses import fields, is_dataclass
from typing import Any, Tuple

import numpy as np
import mlx.core as mx

def is_tensor(x) -> bool:
    """
    Tests if `x` is a `torch.Tensor` or `np.ndarray`.
    """
    if isinstance(x, mx.array):
        return True
    
    return isinstance(x, np.ndarray)