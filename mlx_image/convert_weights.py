"""CLI for converting diffusion model components to MLX safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

from .component_loader import (
    convert_components_from_pretrained,
    recommended_components_for_family,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert diffusion components (UNet/Transformer/VAE/Text Encoder) "
        "to MLX-friendly safetensors."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF repo id or local path containing model subfolders.",
    )
    parser.add_argument(
        "--family",
        default="stable_diffusion",
        choices=[
            "stable_diffusion",
            "stable_diffusion_xl",
            "stable_diffusion_3",
            "flux",
            "flux2",
            "z_image",
            "z_image_turbo",
            "qwen_image",
            "qwen_image_edit",
        ],
        help="Model family hint (controls class candidates and auto mapping).",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Components to convert. Choices: unet transformer vae text_encoder text_encoder_2 clip t5",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for converted .safetensors files.",
    )
    parser.add_argument(
        "--mapping",
        default="auto",
        choices=[
            "auto",
            "generic",
            "sd_unet",
            "sd_vae",
            "sd_clip",
            "flux_transformer",
            "flux_vae",
            "flux_clip",
            "flux_t5",
        ],
        help="Weight mapping rule.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Target dtype for exported weights.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF model revision.",
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="Optional LoRA checkpoint or adapter path to fuse before conversion.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA merge scale.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only from local files/cache.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    components = args.components or list(recommended_components_for_family(args.family))

    outputs = convert_components_from_pretrained(
        model_id_or_path=args.model,
        components=components,
        output_dir=Path(args.outdir),
        family=args.family,
        mapping=args.mapping,
        revision=args.revision,
        dtype=args.dtype,
        lora_path=args.lora,
        lora_scale=args.lora_scale,
        local_files_only=args.local_files_only,
    )

    for component, path in outputs.items():
        print(f"{component}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
