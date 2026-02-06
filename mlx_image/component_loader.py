"""Unified diffusion component loader and PyTorch->MLX weight converter.

This module provides:
1. Component loading for UNet/Transformer, VAE, text encoders (CLIP/T5)
2. Optional LoRA loading and fusion on torch components
3. Weight conversion to MLX-friendly layouts and key mappings
4. safetensors export for converted MLX weights

The implementation mirrors patterns used in Apple's mlx-examples and mflux:
- Stable Diffusion mappings are aligned with mlx-examples model_io logic.
- FLUX mappings use the sanitize rules used by mflux/flux loaders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence
import warnings

import numpy as np
import torch
from safetensors.numpy import save_file as save_numpy_safetensors
from safetensors.torch import load_file as load_torch_safetensors

ComponentType = Literal[
    "unet",
    "transformer",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "clip",
    "t5",
]

ModelFamilyType = Literal[
    "stable_diffusion",
    "stable_diffusion_xl",
    "stable_diffusion_3",
    "flux",
    "flux2",
    "z_image",
    "z_image_turbo",
    "qwen_image",
    "qwen_image_edit",
    "stable-diffusion",
    "stable-diffusion-xl",
    "stable-diffusion-3",
    "z-image",
    "z-image-turbo",
    "qwen-image",
    "qwen-image-edit",
]

MappingType = Literal[
    "auto",
    "generic",
    "sd_unet",
    "sd_vae",
    "sd_clip",
    "flux_transformer",
    "flux_vae",
    "flux_clip",
    "flux_t5",
]

MLXFamilyType = ModelFamilyType
DiffusionModelType = Literal["unet", "transformer"]

_FLOAT_DTYPES = {"float16": torch.float16, "float32": torch.float32}

_FLUX_T5_SHARED_REPLACEMENTS = [
    (".block.", ".layers."),
    (".k.", ".key_proj."),
    (".o.", ".out_proj."),
    (".q.", ".query_proj."),
    (".v.", ".value_proj."),
    ("shared.", "wte."),
    ("lm_head.", "lm_head.linear."),
    (".layer.0.layer_norm.", ".ln1."),
    (".layer.1.layer_norm.", ".ln2."),
    (".layer.2.layer_norm.", ".ln3."),
    (".final_layer_norm.", ".ln."),
    (
        "layers.0.layer.0.SelfAttention.relative_attention_bias.",
        "relative_attention_bias.embeddings.",
    ),
]

_FLUX_T5_ENCODER_REPLACEMENTS = [
    (".layer.0.SelfAttention.", ".attention."),
    (".layer.1.DenseReluDense.", ".dense."),
]

_FAMILY_ALIASES = {
    "stable_diffusion": "stable-diffusion",
    "stable_diffusion_xl": "stable-diffusion-xl",
    "stable_diffusion_3": "stable-diffusion-3",
    "flux": "flux",
    "flux2": "flux2",
    "z_image": "z-image",
    "z_image_turbo": "z-image-turbo",
    "qwen_image": "qwen-image",
    "qwen_image_edit": "qwen-image-edit",
    "stable-diffusion": "stable-diffusion",
    "stable-diffusion-xl": "stable-diffusion-xl",
    "stable-diffusion-3": "stable-diffusion-3",
    "z-image": "z-image",
    "z-image-turbo": "z-image-turbo",
    "qwen-image": "qwen-image",
    "qwen-image-edit": "qwen-image-edit",
}


def resolve_model_family(family: Optional[str]) -> str:
    if family is None:
        return "stable-diffusion"
    key = family.strip().lower()
    if key not in _FAMILY_ALIASES:
        raise ValueError(
            f"Unsupported model family: {family!r}. "
            f"Supported values: {sorted(_FAMILY_ALIASES.keys())}"
        )
    return _FAMILY_ALIASES[key]


def recommended_components_for_family(family: ModelFamilyType) -> Sequence[str]:
    canonical = resolve_model_family(family)
    table = {
        "stable-diffusion": ("unet", "vae", "text_encoder"),
        "stable-diffusion-xl": ("unet", "vae", "text_encoder", "text_encoder_2"),
        "stable-diffusion-3": ("transformer", "vae", "text_encoder", "text_encoder_2"),
        "flux": ("transformer", "vae", "text_encoder", "t5"),
        "flux2": ("transformer", "vae", "text_encoder", "t5"),
        "z-image": ("transformer", "vae", "text_encoder", "text_encoder_2"),
        "z-image-turbo": ("transformer", "vae", "text_encoder", "text_encoder_2"),
        "qwen-image": ("transformer", "vae", "text_encoder", "text_encoder_2"),
        "qwen-image-edit": ("transformer", "vae", "text_encoder", "text_encoder_2"),
    }
    return table[canonical]


@dataclass
class LoadedTorchComponents:
    """Container for torch-side diffusion components."""

    components: Dict[str, torch.nn.Module] = field(default_factory=dict)

    def get(self, name: str) -> Optional[torch.nn.Module]:
        return self.components.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.components


@dataclass
class LoadedMLXComponents:
    """Container for MLX-side diffusion components."""

    diffusion_model: Optional[object] = None
    vae: Optional[object] = None
    text_encoder: Optional[object] = None
    text_encoder_2: Optional[object] = None
    tokenizer: Optional[object] = None
    tokenizer_2: Optional[object] = None


def _normalize_component(component: str) -> str:
    normalized = component.strip().lower().replace("-", "_")
    if normalized == "clip":
        return "text_encoder"
    return normalized


def _load_from_pretrained_with_fallbacks(
    cls,
    model_id_or_path: str,
    subfolders: Sequence[Optional[str]],
    revision: Optional[str],
    torch_dtype: torch.dtype,
    local_files_only: bool,
):
    errors = []
    for subfolder in subfolders:
        kwargs = {
            "torch_dtype": torch_dtype,
            "local_files_only": local_files_only,
        }
        if revision:
            kwargs["revision"] = revision
        if subfolder is not None:
            kwargs["subfolder"] = subfolder

        try:
            model = cls.from_pretrained(model_id_or_path, **kwargs)
            model.eval()
            return model
        except Exception as exc:  # pragma: no cover - depends on upstream loaders
            errors.append(f"subfolder={subfolder!r}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(
        f"Failed to load {cls.__name__} from {model_id_or_path!r}.\nTried:\n{joined}"
    )


def _get_transformer_candidate_class_names(family: str) -> Sequence[str]:
    by_family = {
        "stable-diffusion-3": (
            "SD3Transformer2DModel",
            "SD3TransformerModel",
            "Transformer2DModel",
        ),
        "flux": (
            "FluxTransformer2DModel",
            "Transformer2DModel",
        ),
        "flux2": (
            "Flux2Transformer2DModel",
            "FluxTransformer2DModel",
            "Transformer2DModel",
        ),
        "z-image": (
            "ZImageTransformer2DModel",
            "FluxTransformer2DModel",
            "SD3Transformer2DModel",
            "Transformer2DModel",
        ),
        "z-image-turbo": (
            "ZImageTransformer2DModel",
            "FluxTransformer2DModel",
            "SD3Transformer2DModel",
            "Transformer2DModel",
        ),
        "qwen-image": (
            "QwenImageTransformer2DModel",
            "FluxTransformer2DModel",
            "SD3Transformer2DModel",
            "Transformer2DModel",
        ),
        "qwen-image-edit": (
            "QwenImageTransformer2DModel",
            "FluxTransformer2DModel",
            "SD3Transformer2DModel",
            "Transformer2DModel",
        ),
    }
    return by_family.get(
        family, ("FluxTransformer2DModel", "SD3Transformer2DModel", "Transformer2DModel")
    )


def _get_text_encoder_candidates(family: str, component: str):
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel

    if component == "t5":
        return ((T5EncoderModel, ("text_encoder_2", "text_encoder", None)),)

    if component == "text_encoder_2":
        if family in {"stable-diffusion", "stable-diffusion-xl"}:
            return (
                (CLIPTextModelWithProjection, ("text_encoder_2", None)),
                (CLIPTextModel, ("text_encoder_2", None)),
            )
        # SD3 / FLUX-like families typically use T5 as encoder_2.
        return (
            (T5EncoderModel, ("text_encoder_2", None)),
            (CLIPTextModelWithProjection, ("text_encoder_2", None)),
            (CLIPTextModel, ("text_encoder_2", None)),
        )

    # component == text_encoder
    if family in {"stable-diffusion", "stable-diffusion-xl"}:
        return (
            (CLIPTextModel, ("text_encoder", None)),
            (CLIPTextModelWithProjection, ("text_encoder", None)),
        )

    # Transformer-centric families can still expose CLIP-like text_encoder.
    return (
        (CLIPTextModelWithProjection, ("text_encoder", None)),
        (CLIPTextModel, ("text_encoder", None)),
        (T5EncoderModel, ("text_encoder", None)),
    )


def load_torch_component(
    model_id_or_path: str,
    component: ComponentType,
    *,
    family: Optional[ModelFamilyType] = None,
    revision: Optional[str] = None,
    dtype: Literal["float16", "float32"] = "float16",
    local_files_only: bool = False,
) -> torch.nn.Module:
    """Load one diffusion component from a HF repo ID or local path.

    Args:
        model_id_or_path: HF repo id (e.g. runwayml/stable-diffusion-v1-5)
            or local model directory.
        component: One of unet/transformer/vae/text_encoder/text_encoder_2/clip/t5.
        family: Model family hint. Supported:
            stable_diffusion, stable_diffusion_xl, stable_diffusion_3,
            flux, flux2, z_image, z_image_turbo, qwen_image, qwen_image_edit.
        revision: Optional HF revision.
        dtype: float16 or float32.
        local_files_only: Force local cache/files only.
    """

    normalized = _normalize_component(component)
    canonical_family = resolve_model_family(family)
    torch_dtype = _FLOAT_DTYPES[dtype]

    if normalized == "unet":
        from diffusers import UNet2DConditionModel

        return _load_from_pretrained_with_fallbacks(
            UNet2DConditionModel,
            model_id_or_path,
            subfolders=("unet", None),
            revision=revision,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

    if normalized == "transformer":
        import diffusers

        class_names = _get_transformer_candidate_class_names(canonical_family)

        last_error = None
        for class_name in class_names:
            cls = getattr(diffusers, class_name, None)
            if cls is None:
                continue
            try:
                return _load_from_pretrained_with_fallbacks(
                    cls,
                    model_id_or_path,
                    subfolders=("transformer", "unet", None),
                    revision=revision,
                    torch_dtype=torch_dtype,
                    local_files_only=local_files_only,
                )
            except Exception as exc:  # pragma: no cover - depends on model family
                last_error = exc

        raise RuntimeError(
            f"Failed to load transformer diffusion model from {model_id_or_path!r}. "
            f"Last error: {last_error}"
        )

    if normalized == "vae":
        from diffusers import AutoencoderKL

        return _load_from_pretrained_with_fallbacks(
            AutoencoderKL,
            model_id_or_path,
            subfolders=("vae", None),
            revision=revision,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

    if normalized in {"text_encoder", "text_encoder_2", "t5"}:
        candidates = _get_text_encoder_candidates(canonical_family, normalized)

        last_error = None
        for cls, subfolders in candidates:
            try:
                return _load_from_pretrained_with_fallbacks(
                    cls,
                    model_id_or_path,
                    subfolders=subfolders,
                    revision=revision,
                    torch_dtype=torch_dtype,
                    local_files_only=local_files_only,
                )
            except Exception as exc:  # pragma: no cover - depends on model family
                last_error = exc

        raise RuntimeError(
            f"Failed to load text encoder component {normalized!r} "
            f"from {model_id_or_path!r}. Last error: {last_error}"
        )

    raise ValueError(f"Unsupported component type: {component!r}")


def load_torch_components(
    model_id_or_path: str,
    components: Iterable[ComponentType],
    *,
    family: Optional[ModelFamilyType] = None,
    revision: Optional[str] = None,
    dtype: Literal["float16", "float32"] = "float16",
    local_files_only: bool = False,
) -> LoadedTorchComponents:
    """Load multiple diffusion components in torch format."""

    loaded: Dict[str, torch.nn.Module] = {}
    for component in components:
        key = _normalize_component(component)
        loaded[key] = load_torch_component(
            model_id_or_path,
            key,  # type: ignore[arg-type]
            family=family,
            revision=revision,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    return LoadedTorchComponents(components=loaded)


def _try_fuse_lora(model: torch.nn.Module, lora_scale: float) -> None:
    if not hasattr(model, "fuse_lora"):
        return
    try:
        model.fuse_lora(lora_scale=lora_scale)
        return
    except TypeError:
        pass
    try:
        model.fuse_lora(scale=lora_scale)
        return
    except TypeError:
        pass
    # Some implementations expose fuse_lora without scale args.
    model.fuse_lora()


def apply_lora_to_torch_component(
    model: torch.nn.Module,
    lora_path: str,
    *,
    lora_scale: float = 1.0,
) -> torch.nn.Module:
    """Apply LoRA weights to a torch component and fuse them if possible."""

    errors = []

    if hasattr(model, "load_lora_adapter"):
        try:
            model.load_lora_adapter(lora_path, adapter_name="mlx_loader")
            _try_fuse_lora(model, lora_scale)
            return model
        except Exception as exc:  # pragma: no cover - depends on backend support
            errors.append(f"load_lora_adapter: {exc}")

    if hasattr(model, "load_attn_procs"):
        try:
            model.load_attn_procs(lora_path)
            _try_fuse_lora(model, lora_scale)
            return model
        except Exception as exc:  # pragma: no cover - depends on model family
            errors.append(f"load_attn_procs: {exc}")

    try:
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(model, lora_path)
        merged = peft_model.merge_and_unload(progressbar=False)
        return merged
    except Exception as exc:  # pragma: no cover - optional dependency path
        errors.append(f"peft.merge_and_unload: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to apply LoRA from {lora_path!r}.\n{joined}")


def _as_float_tensor(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    value = value.detach().cpu()
    if torch.is_floating_point(value):
        value = value.to(dtype)
    return value


def _conv_oihw_to_ohwc(value: torch.Tensor) -> torch.Tensor:
    return value.permute(0, 2, 3, 1).contiguous()


def _map_sd_unet_weight(key: str, value: torch.Tensor):
    if "downsamplers" in key:
        key = key.replace("downsamplers.0.conv", "downsample")
    if "upsamplers" in key:
        key = key.replace("upsamplers.0.conv", "upsample")

    if "mid_block.resnets.0" in key:
        key = key.replace("mid_block.resnets.0", "mid_blocks.0")
    if "mid_block.attentions.0" in key:
        key = key.replace("mid_block.attentions.0", "mid_blocks.1")
    if "mid_block.resnets.1" in key:
        key = key.replace("mid_block.resnets.1", "mid_blocks.2")

    if "to_k" in key:
        key = key.replace("to_k", "key_proj")
    if "to_out.0" in key:
        key = key.replace("to_out.0", "out_proj")
    if "to_q" in key:
        key = key.replace("to_q", "query_proj")
    if "to_v" in key:
        key = key.replace("to_v", "value_proj")

    if "ff.net.2" in key:
        key = key.replace("ff.net.2", "linear3")

    if "ff.net.0" in key:
        k1 = key.replace("ff.net.0.proj", "linear1")
        k2 = key.replace("ff.net.0.proj", "linear2")
        v1, v2 = value.chunk(2, dim=0)
        return [(k1, v1), (k2, v2)]

    if "conv_shortcut.weight" in key:
        value = value.squeeze(-1).squeeze(-1)

    if value.ndim == 4 and ("proj_in" in key or "proj_out" in key):
        value = value.squeeze(-1).squeeze(-1)

    if value.ndim == 4:
        value = _conv_oihw_to_ohwc(value)

    return [(key, value)]


def _map_sd_vae_weight(key: str, value: torch.Tensor):
    if "downsamplers" in key:
        key = key.replace("downsamplers.0.conv", "downsample")
    if "upsamplers" in key:
        key = key.replace("upsamplers.0.conv", "upsample")

    if "to_k" in key:
        key = key.replace("to_k", "key_proj")
    if "to_out.0" in key:
        key = key.replace("to_out.0", "out_proj")
    if "to_q" in key:
        key = key.replace("to_q", "query_proj")
    if "to_v" in key:
        key = key.replace("to_v", "value_proj")

    if "mid_block.resnets.0" in key:
        key = key.replace("mid_block.resnets.0", "mid_blocks.0")
    if "mid_block.attentions.0" in key:
        key = key.replace("mid_block.attentions.0", "mid_blocks.1")
    if "mid_block.resnets.1" in key:
        key = key.replace("mid_block.resnets.1", "mid_blocks.2")

    if "quant_conv" in key:
        key = key.replace("quant_conv", "quant_proj")
        value = value.squeeze(-1).squeeze(-1)

    if "conv_shortcut.weight" in key:
        value = value.squeeze(-1).squeeze(-1)

    if value.ndim == 4:
        value = _conv_oihw_to_ohwc(value)

    return [(key, value)]


def _map_sd_clip_weight(key: str, value: torch.Tensor):
    if key.startswith("text_model."):
        key = key[11:]
    if key.startswith("embeddings."):
        key = key[11:]
    if key.startswith("encoder."):
        key = key[8:]

    if "self_attn." in key:
        key = key.replace("self_attn.", "attention.")
    if "q_proj." in key:
        key = key.replace("q_proj.", "query_proj.")
    if "k_proj." in key:
        key = key.replace("k_proj.", "key_proj.")
    if "v_proj." in key:
        key = key.replace("v_proj.", "value_proj.")

    if "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "linear1")
    if "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "linear2")

    return [(key, value)]


def _map_flux_transformer_weight(key: str, value: torch.Tensor):
    if key.startswith("model.diffusion_model."):
        key = key[22:]
    if key.endswith(".scale"):
        key = key[:-6] + ".weight"

    for seq in ("img_mlp", "txt_mlp", "adaLN_modulation"):
        if f".{seq}." in key:
            key = key.replace(f".{seq}.", f".{seq}.layers.")
            break

    return [(key, value)]


def _map_flux_clip_weight(key: str, value: torch.Tensor):
    if key.startswith("text_model."):
        key = key[11:]
    if key.startswith("embeddings."):
        key = key[11:]
    if key.startswith("encoder."):
        key = key[8:]

    if "self_attn." in key:
        key = key.replace("self_attn.", "attention.")
    if "q_proj." in key:
        key = key.replace("q_proj.", "query_proj.")
    if "k_proj." in key:
        key = key.replace("k_proj.", "key_proj.")
    if "v_proj." in key:
        key = key.replace("v_proj.", "value_proj.")

    if "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "linear1")
    if "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "linear2")

    return [(key, value)]


def _map_flux_vae_weight(key: str, value: torch.Tensor):
    if value.ndim == 4:
        value = _conv_oihw_to_ohwc(value)
        if value.shape[1:3] == (1, 1):
            value = value.squeeze((1, 2))

    return [(key, value)]


def _map_flux_t5_weight(key: str, value: torch.Tensor):
    for old, new in _FLUX_T5_SHARED_REPLACEMENTS:
        key = key.replace(old, new)
    if key.startswith("encoder."):
        for old, new in _FLUX_T5_ENCODER_REPLACEMENTS:
            key = key.replace(old, new)
    return [(key, value)]


def _map_generic_weight(key: str, value: torch.Tensor):
    if value.ndim == 4:
        value = _conv_oihw_to_ohwc(value)
    return [(key, value)]


def _infer_mapping(
    component: str,
    state_dict: Dict[str, torch.Tensor],
    *,
    family: Optional[ModelFamilyType] = None,
) -> MappingType:
    canonical_family = resolve_model_family(family)
    keys = list(state_dict.keys())

    if component == "unet":
        return "sd_unet"

    if component == "transformer":
        if canonical_family in {
            "flux",
            "flux2",
            "z-image",
            "z-image-turbo",
            "qwen-image",
            "qwen-image-edit",
        }:
            return "flux_transformer"
        flux_like = any(
            k.startswith("model.diffusion_model.") or ".img_mlp." in k for k in keys
        )
        return "flux_transformer" if flux_like else "generic"

    if component == "vae":
        if canonical_family in {"flux", "flux2"}:
            return "flux_vae"
        flux_like = any(
            k.startswith("encoder.") and (".down." in k or ".mid." in k)
            for k in keys
        )
        return "flux_vae" if flux_like else "sd_vae"

    if component in {"text_encoder", "text_encoder_2", "clip"}:
        if canonical_family in {
            "stable-diffusion",
            "stable-diffusion-xl",
        }:
            return "sd_clip"
        if canonical_family in {"flux", "flux2"} and component == "text_encoder":
            return "flux_clip"
        if canonical_family in {
            "stable-diffusion-3",
            "z-image",
            "z-image-turbo",
            "qwen-image",
            "qwen-image-edit",
        } and component == "text_encoder_2":
            return "flux_t5"
        t5_like = any(k.startswith("shared.") or ".block." in k for k in keys)
        if t5_like:
            return "flux_t5"
        clip_like = any("self_attn" in k or k.startswith("text_model.") for k in keys)
        if clip_like:
            # Works for SD/SDXL CLIP encoders and Flux CLIP text encoder.
            return "sd_clip"
        return "generic"

    if component == "t5":
        return "flux_t5"

    return "generic"


def _get_mapper(mapping: MappingType):
    mapping_table = {
        "generic": _map_generic_weight,
        "sd_unet": _map_sd_unet_weight,
        "sd_vae": _map_sd_vae_weight,
        "sd_clip": _map_sd_clip_weight,
        "flux_transformer": _map_flux_transformer_weight,
        "flux_vae": _map_flux_vae_weight,
        "flux_clip": _map_flux_clip_weight,
        "flux_t5": _map_flux_t5_weight,
    }
    if mapping not in mapping_table:
        raise ValueError(f"Unsupported mapping: {mapping!r}")
    return mapping_table[mapping]


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def convert_torch_state_dict_to_mlx(
    state_dict: Dict[str, torch.Tensor],
    *,
    component: ComponentType,
    family: Optional[ModelFamilyType] = None,
    mapping: MappingType = "auto",
    dtype: Literal["float16", "float32"] = "float16",
) -> Dict[str, np.ndarray]:
    """Convert a torch state dict into MLX-friendly numpy weights."""

    normalized_component = _normalize_component(component)
    effective_mapping = (
        _infer_mapping(normalized_component, state_dict, family=family)
        if mapping == "auto"
        else mapping
    )
    mapper = _get_mapper(effective_mapping)
    target_dtype = _FLOAT_DTYPES[dtype]

    converted: Dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        value = _as_float_tensor(value, target_dtype)
        for mapped_key, mapped_value in mapper(key, value):
            converted[mapped_key] = _tensor_to_numpy(mapped_value)

    return converted


def save_mlx_safetensors(
    weights: Dict[str, np.ndarray],
    output_path: str | Path,
    *,
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """Save converted MLX weights to a safetensors file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_numpy_safetensors(weights, str(output_path), metadata=metadata)
    return output_path


def convert_component_from_pretrained(
    model_id_or_path: str,
    component: ComponentType,
    *,
    output_path: str | Path,
    family: Optional[ModelFamilyType] = None,
    mapping: MappingType = "auto",
    revision: Optional[str] = None,
    dtype: Literal["float16", "float32"] = "float16",
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    local_files_only: bool = False,
) -> Path:
    """Load one component and export converted MLX safetensors."""

    model = load_torch_component(
        model_id_or_path,
        component,
        family=family,
        revision=revision,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    if lora_path:
        model = apply_lora_to_torch_component(
            model, lora_path=lora_path, lora_scale=lora_scale
        )

    state_dict = model.state_dict()
    converted = convert_torch_state_dict_to_mlx(
        state_dict,
        component=component,
        family=family,
        mapping=mapping,
        dtype=dtype,
    )
    meta = {
        "source_model": model_id_or_path,
        "component": _normalize_component(component),
        "family": resolve_model_family(family),
        "mapping": mapping,
        "dtype": dtype,
    }
    if lora_path:
        meta["lora_path"] = lora_path
        meta["lora_scale"] = str(lora_scale)

    return save_mlx_safetensors(converted, output_path, metadata=meta)


def convert_components_from_pretrained(
    model_id_or_path: str,
    components: Iterable[ComponentType],
    *,
    output_dir: str | Path,
    family: Optional[ModelFamilyType] = None,
    mapping: MappingType = "auto",
    revision: Optional[str] = None,
    dtype: Literal["float16", "float32"] = "float16",
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    local_files_only: bool = False,
) -> Dict[str, Path]:
    """Convert multiple components from a model into MLX safetensors files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}
    for component in components:
        normalized = _normalize_component(component)
        out_path = output_dir / f"{normalized}.safetensors"
        saved = convert_component_from_pretrained(
            model_id_or_path=model_id_or_path,
            component=normalized,  # type: ignore[arg-type]
            output_path=out_path,
            family=family,
            mapping=mapping,
            revision=revision,
            dtype=dtype,
            lora_path=lora_path,
            lora_scale=lora_scale,
            local_files_only=local_files_only,
        )
        outputs[normalized] = saved

    return outputs


def load_torch_checkpoint_state_dict(checkpoint_path: str | Path) -> Dict[str, torch.Tensor]:
    """Load a raw torch/safetensors checkpoint into a state_dict-like dict."""

    checkpoint_path = Path(checkpoint_path)
    suffix = checkpoint_path.suffix.lower()

    if suffix == ".safetensors":
        return load_torch_safetensors(str(checkpoint_path))

    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(
        state["state_dict"], dict
    ):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unsupported checkpoint object type at {checkpoint_path}: {type(state)}")


def convert_checkpoint_to_mlx(
    checkpoint_path: str | Path,
    *,
    component: ComponentType,
    output_path: str | Path,
    family: Optional[ModelFamilyType] = None,
    mapping: MappingType = "auto",
    dtype: Literal["float16", "float32"] = "float16",
) -> Path:
    """Convert a local checkpoint file to MLX safetensors."""

    state_dict = load_torch_checkpoint_state_dict(checkpoint_path)
    converted = convert_torch_state_dict_to_mlx(
        state_dict,
        component=component,
        family=family,
        mapping=mapping,
        dtype=dtype,
    )
    meta = {
        "source_checkpoint": str(checkpoint_path),
        "component": _normalize_component(component),
        "family": resolve_model_family(family),
        "mapping": mapping,
        "dtype": dtype,
    }
    return save_mlx_safetensors(converted, output_path, metadata=meta)


def load_mlx_components(
    *,
    family: MLXFamilyType,
    model: str,
    diffusion_model: DiffusionModelType = "unet",
    load_vae: bool = True,
    load_text_encoder: bool = True,
    load_text_encoder_2: bool = False,
    load_tokenizer: bool = False,
    float16: bool = True,
    hf_download: bool = True,
    lora_path: Optional[str] = None,
) -> LoadedMLXComponents:
    """Load MLX-native components for supported model families.

    Notes:
    - `stable_diffusion` and `stable_diffusion_xl` route to `stable_diffusion.model_io`.
    - `flux`, `flux2`, `z_image`, `z_image_turbo`, `qwen_image`, `qwen_image_edit`
      route to `flux.utils` loaders.
    - `stable_diffusion_3` currently has no dedicated MLX-native loader in this repo;
      use torch loading + conversion APIs in this module.
    - LoRA merging is recommended on the torch side using
      `convert_component_from_pretrained(..., lora_path=...)`.
    """

    if lora_path:
        warnings.warn(
            "MLX-native LoRA merge is not performed in load_mlx_components(). "
            "Apply LoRA during torch conversion instead.",
            stacklevel=2,
        )

    bundle = LoadedMLXComponents()
    canonical_family = resolve_model_family(family)

    if canonical_family in {"stable-diffusion", "stable-diffusion-xl"}:
        if diffusion_model != "unet":
            raise ValueError("Stable Diffusion MLX loader currently supports diffusion_model='unet' only.")

        from .stable_diffusion.model_io import (
            load_autoencoder,
            load_text_encoder as load_sd_text_encoder,
            load_tokenizer as load_sd_tokenizer,
            load_unet,
        )

        bundle.diffusion_model = load_unet(model, float16=float16)
        if load_vae:
            bundle.vae = load_autoencoder(model, float16=False)
        if load_text_encoder:
            bundle.text_encoder = load_sd_text_encoder(model, float16=float16)
        if load_text_encoder_2 or canonical_family == "stable-diffusion-xl":
            bundle.text_encoder_2 = load_sd_text_encoder(
                model,
                float16=float16,
                model_key="text_encoder_2",
            )
        if load_tokenizer:
            bundle.tokenizer = load_sd_tokenizer(model)
            if load_text_encoder_2 or canonical_family == "stable-diffusion-xl":
                bundle.tokenizer_2 = load_sd_tokenizer(
                    model,
                    vocab_key="tokenizer_2_vocab",
                    merges_key="tokenizer_2_merges",
                )
        return bundle

    if canonical_family in {
        "flux",
        "flux2",
        "z-image",
        "z-image-turbo",
        "qwen-image",
        "qwen-image-edit",
    }:
        if diffusion_model != "transformer":
            raise ValueError(
                "FLUX-like MLX loader expects diffusion_model='transformer'."
            )

        from .flux.utils import (
            load_ae,
            load_clip,
            load_clip_tokenizer,
            load_flow_model,
            load_t5,
            load_t5_tokenizer,
        )

        bundle.diffusion_model = load_flow_model(model, hf_download=hf_download)
        if load_vae:
            bundle.vae = load_ae(model, hf_download=hf_download)
        if load_text_encoder:
            bundle.text_encoder = load_clip(model)
        if load_text_encoder_2 or canonical_family in {
            "flux",
            "flux2",
            "z-image",
            "z-image-turbo",
            "qwen-image",
            "qwen-image-edit",
        }:
            bundle.text_encoder_2 = load_t5(model)
        if load_tokenizer:
            bundle.tokenizer = load_clip_tokenizer(model)
            if load_text_encoder_2 or canonical_family in {
                "flux",
                "flux2",
                "z-image",
                "z-image-turbo",
                "qwen-image",
                "qwen-image-edit",
            }:
                bundle.tokenizer_2 = load_t5_tokenizer(model)
        return bundle

    if canonical_family == "stable-diffusion-3":
        raise NotImplementedError(
            "stable_diffusion_3 MLX-native loader is not implemented in this repository yet. "
            "Use `load_torch_component(..., family='stable_diffusion_3')` + "
            "`convert_component_from_pretrained(...)` paths."
        )

    raise ValueError(f"Unsupported MLX family: {family!r} -> {canonical_family!r}")


__all__ = [
    "ModelFamilyType",
    "LoadedMLXComponents",
    "LoadedTorchComponents",
    "apply_lora_to_torch_component",
    "convert_checkpoint_to_mlx",
    "convert_component_from_pretrained",
    "convert_components_from_pretrained",
    "convert_torch_state_dict_to_mlx",
    "load_mlx_components",
    "load_torch_checkpoint_state_dict",
    "load_torch_component",
    "load_torch_components",
    "recommended_components_for_family",
    "resolve_model_family",
    "save_mlx_safetensors",
]
