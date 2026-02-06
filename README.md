# MLX Image Diffusion

## Component Loader / Converter

`mlx_image/component_loader.py` adds:

- `UNet` or `Transformer` (diffusion model) loading
- `VAE` loading
- `Text Encoder / CLIP` loading
- optional LoRA merge on torch components
- PyTorch checkpoint -> MLX-friendly `.safetensors` conversion

### Python Example

```python
from mlx_image.component_loader import (
    load_torch_components,
    convert_components_from_pretrained,
    load_mlx_components,
)

# 1) Load torch-side components (HF repo or local path)
torch_bundle = load_torch_components(
    "runwayml/stable-diffusion-v1-5",
    components=["unet", "vae", "text_encoder"],
    dtype="float16",
)

# 2) Convert to MLX-friendly safetensors (optionally with LoRA)
converted = convert_components_from_pretrained(
    model_id_or_path="runwayml/stable-diffusion-v1-5",
    components=["unet", "vae", "text_encoder"],
    output_dir="./mlx_weights/sd15",
    mapping="auto",
    dtype="float16",
    # lora_path="./my_lora",
    # lora_scale=0.8,
)

# 3) Load MLX-native components from this repository
mlx_bundle = load_mlx_components(
    family="stable-diffusion",
    model="runwayml/stable-diffusion-v1-5",
    diffusion_model="unet",
    load_vae=True,
    load_text_encoder=True,
)
```

### CLI Example

```bash
python -m mlx_image.convert_weights \
  --model runwayml/stable-diffusion-v1-5 \
  --components unet vae text_encoder \
  --outdir ./mlx_weights/sd15 \
  --mapping auto \
  --dtype float16
```
