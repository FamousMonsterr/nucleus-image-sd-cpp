# Nucleus-Image on stable-diffusion.cpp (Vulkan, AMD gfx906)

First implementation of Nucleus-Image (Sparse MoE image DiT, 17B total / ~2B active) running outside PyTorch, via stable-diffusion.cpp with the Vulkan backend on an AMD Radeon Pro VII (gfx906, 16 GB VRAM).

PR: leejet/stable-diffusion.cpp (link pending)

## Hardware

| Component | Spec |
|---|---|
| GPU | AMD Radeon Pro VII 16 GB (gfx906 / Vega 20, Vulkan only — no ZLUDA, no ROCm on Windows) |
| CPU | Ryzen 5 5600X (6c/12t) |
| RAM | 32 GB DDR4 |
| OS | Windows 11 |

## Command

```
sd-cli --diffusion-model Nucleus-Image-Q5_1.safetensors \
  --llm Qwen3-VL-8B-Instruct-Q4_K_M.gguf \
  --vae qwen_image_vae.safetensors \
  --tokenizer nucleus_tokenizer.json \
  -p "prompt" -n "negative" \
  --steps 24 --cfg-scale 4.5 -H 768 -W 768 --seed 42
```

For Q8_0 (18 GB, exceeds 16 GB VRAM): add `NUCLEUS_EXPERTS_CPU=1` env to offload MoE experts to system RAM.

## Reference images

| File | Quant | Prompt | Resolution | Steps | Time | Notes |
|---|---|---|---|---|---|---|
| q5_cabin.png | Q5_1 | "a cozy wooden cabin in a snowy forest at dusk, warm light in the windows, cinematic" | 768×768 | 24 | 27 min (67 s/step) | |
| q8_cabin.png | Q8_0 (CPU experts) | same | 768×768 | 24 | 35 min (87 s/step) | |
| q5_apple_success.png | Q5_1 | "a red apple on a wooden table" | 768×768 | 20 | ~3.5 min | |
| q5_lighthouse_cfg1.png | Q5_1 | "a majestic lighthouse on a cliff at sunset" | 512×512 | 20 | ~2 min | cfg=1 (no guidance) |
| q5_lighthouse_cfg4.png | Q5_1 | same | 512×512 | 20 | ~4 min | cfg=4 (renormalized) — notice the improvement |
| q8_cat.png | Q8_0 (CPU experts) | "a cat sitting on a car in the rain" | 512×512 | 20 | 15 min | |

## Performance summary (gfx906 16 GB, Vulkan)

| Quant | VRAM | RAM | Speed | Notes |
|---|---|---|---|---|
| Q5_1 | 12.1 GB | 4.5 GB | 5-11 s/step | Corrupted blocks in upstream quant (layers 9, 21, 30, 31) — built-in sanitizer zeroes them at load (~30 s overhead) |
| Q8_0 + `NUCLEUS_EXPERTS_CPU=1` | ~2 GB | ~16 GB | 45-87 s/step | Clean weights, higher quality |
| BF16 singlefile (34 GB) | — | — | untested | Exceeds 32 GB system RAM |

## Implementation notes

- **Expert-choice routing** implemented using existing ggml ops: `argsort_top_k` (column-wise) → `get_rows` → `set_rows` (replaces `scatter_add`) → gate renormalization → token-major inversion → `mul_mat_id` with R=8 slots
- **Quantized expert tensors** are transposed at load: dequant → transpose → re-quant (one-time, ~30-110 s)
- **Renormalized CFG**: `comb = neg + scale*(cond-neg)`, then L2-renorm to cond norm (matches diffusers pipeline)
- **F16 overflow in matmul kernels**: Nucleus activations reach ~2e5, overflowing F16 accumulators in quantized matmuls — fixed with exact power-of-2 pre/post rescales
- **Text encoder**: Qwen3-VL-8B, hidden_states[-8] (8th from last layer, no final norm), pipeline system prompt
- **Known limitation**: CFG > 1 renormalization implemented; full CFG pipeline with separate uncond pass doubles generation time

## Credits

- [NucleusAI](https://huggingface.co/NucleusAI/Nucleus-Image) for the model
- [stduhpf](https://huggingface.co/stduhpf/Nucleus-Image-sdcpp) for the Q5_1/Q8_0 quants (Q5_1 has corrupted blocks in layers 9/21/30/31 — the sanitizer handles this)
- [leejet](https://github.com/leejet/stable-diffusion.cpp) for stable-diffusion.cpp
