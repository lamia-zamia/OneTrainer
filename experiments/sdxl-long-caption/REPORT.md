# Pretrained SDXL long-caption experiment report

This report is generated from `pretrained_results.json`. All model-forward results use the local pretrained Illustrious XL checkpoint named in that JSON. No randomly initialized model is used as evidence.

Checkpoint: `D:\ComfyUI\ComfyUI\ComfyUI\models\checkpoints\illustriousXL_v01.safetensors`
SHA-256: `3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9`
OneTrainer revision: `1906424bc4b064f07c94e2f64d3d3d28ba100edd`

## ComfyUI compatibility

The comparisons are saved in [`comfy_compat.json`](comfy_compat.json) and [`comfy_compat_fp32.json`](comfy_compat_fp32.json). They use the local ComfyUI source, the same checkpoint, and realistic Danbooru-style tags. Both files store full token IDs per section for CLIP-L and CLIP-G, chunk counts, effective conditioning shapes, padding masks, hidden padding values, and pooled outputs.
For these unweighted captions, ComfyUI groups the full text as one long tokenizer segment; its boundaries match the 75-content-token splits exactly. ComfyUI also trims dual-encoder conditioning to the smaller CLIP-L/CLIP-G section count and uses CLIP-G pooled output. Both implementations omit a text padding attention mask for SDXL; OneTrainer therefore encodes native EOS/PAD tokens and trims inactive sections before the UNet, matching the measured reference conditioning.

For the 40-token caption, ComfyUI encodes 77 positions; OneTrainer returns a fixed-capacity 231-position tensor with 154 masked positions whose hidden values are exactly zero. Its effective sequence is 77 positions.

Token IDs match for both encoders across all shared sections in the FP32 parity run: `True`. ComfyUI section counts are 1 at 20/74/75 tokens, 2 at 76/100/149/150, 3 at 151/200/225, and 4 at 232; OneTrainer's configured 225-token cap retains three sections at 232.
With both implementations' CLIP parameters normalized to FP32 from the same checkpoint, maximum relative L2 differences across the tested captions are `5.79e-06` for sequence conditioning and `6.58e-06` for pooled conditioning. With both parameter sets normalized to FP16, the maxima are `0.00179` and `0.00241`; ComfyUI returns `torch.float32` conditioning while OneTrainer returns `torch.float16`, so the remaining gap is measured with its FP32 output/accumulation path.
The ComfyUI source revision is `3c1b7a17fdf239d35ce98cc777756f572d089795`. Both comparisons run that source inside the OneTrainer Python/Torch environment so the same CUDA-enabled Torch build is used.

## Short-caption compatibility

- 40-token caption at 75 vs 150: CLIP-L hidden relative L2 `0`, CLIP-G hidden relative L2 `0`, pooled relative L2 `0`.
- 40-token caption at 75 vs 225: CLIP-L hidden relative L2 `0`, CLIP-G hidden relative L2 `0`, pooled relative L2 `0`.
- Pretrained UNet prediction relative L2 at 75 vs 225: `0`; same noise `True`, same timestep `True`.

## Pretrained UNet LoRA gradients

- `75_caption_A_vs_B`: absolute L2 `0`, relative `0`.
- `75_repeat_control`: absolute L2 `0`, relative `0`.
- `225_caption_A_vs_B_third_chunk`: absolute L2 `0.00014797404`, relative `0.26555352`.
- `225_caption_A_vs_B_chunk_two_only`: absolute L2 `0.00024760211`, relative `0.36212287`.

All cases used the same pretrained UNet, initial OneTrainer LoRA state, image-derived latent, noise, timestep, batch size, loss, and FP16 autocast. The text encoders were frozen. The 75-token pair differs only after the retained prefix; the 225-token pair uses different third-chunk tags. A second pair differs only in chunk two.

## One-step optimizer updates

- `A1_75`: gradient L2 `0.00037219041`, update L2 `0.014936328`, finite `True`.
- `A2_75`: gradient L2 `0.00037219041`, update L2 `0.014936328`, finite `True`.
- `B1_225`: gradient L2 `0.00055722863`, update L2 `0.01491699`, finite `True`.
- `B2_225`: gradient L2 `0.00056189124`, update L2 `0.01496115`, finite `True`.
- Update difference `75_A_vs_B`: absolute L2 `0`, relative `0`.
- Update difference `225_A_vs_B`: absolute L2 `0.010809511`, relative `0.72464441`.
- Update difference `75_vs_225_same_caption_A`: absolute L2 `0.014996584`, relative `1.0040342`.

## Controlled LoRA training

Eight OneTrainer updates per treatment used one image, matched optimizer settings, identical initial LoRA weights, and the same per-step noise and timestep. Final 75-vs-225 adapter parameter delta: absolute L2 `0.074038959`, relative `0.017782679`.

- Evaluation seed `42`: mean absolute pixel difference `0.064334787`, PSNR `18.355 dB`.
- Evaluation seed `123`: mean absolute pixel difference `0.11080035`, PSNR `15.178 dB`.

The parameter and fixed-seed image comparisons show a detectable update effect in this bounded run. They do not establish that the 225-token treatment is perceptually better. The training contact sheet is `images/training_effect_contact_sheet_512.png`; the eight-step outputs are visibly different but still have sampler artifacts.

## Base-model image comparisons

Generated 10 images at 512px with Euler, 8 steps, CFG 7.0, and fixed seeds [42, 123]. Contact sheet: [`images/contact_sheet_512.png`](images/contact_sheet_512.png).

- Seed `42`, `short_75_vs_225`: mean absolute pixel difference `0`, PSNR `exact match`.
- Seed `42`, `long_75_vs_225`: mean absolute pixel difference `0.31934363`, PSNR `7.6333 dB`.
- Seed `42`, `third_chunk_A_vs_B_225`: mean absolute pixel difference `0.25243753`, PSNR `8.2737 dB`.
- Seed `123`, `short_75_vs_225`: mean absolute pixel difference `0`, PSNR `exact match`.
- Seed `123`, `long_75_vs_225`: mean absolute pixel difference `0.42722031`, PSNR `5.8574 dB`.
- Seed `123`, `third_chunk_A_vs_B_225`: mean absolute pixel difference `0.3065199`, PSNR `8.247 dB`.

Visual inspection shows exact short-caption matches across 75/225 modes, with clear layout and content changes for long-caption versus truncated prompts and for third-chunk variants. Pixel differences are not quality scores; the eight-step Euler images are diagnostic comparisons, not quality evaluations.

## Performance

The microsteps ran on the NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GiB) with batch size 1, 256px images (32×32 latent), FP16 UNet autocast, FP32 LoRA, and OneTrainer native SDP with PyTorch default backend selection.
Text-encoder cache preparation measured 15 prompt/capacity combinations in 1.1490s (0.0616–0.0836s each).

| Case | Seconds / microstep | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|
| short40_75 | 0.8090 | 5090.2 | 5304.0 |
| short40_150 | 0.5652 | 5090.5 | 5304.0 |
| short40_225 | 0.5610 | 5090.9 | 5304.0 |
| longA_75 | 0.5549 | 5090.2 | 5304.0 |
| longA_150 | 0.5917 | 5090.5 | 5304.0 |
| longA_225 | 0.5444 | 5090.9 | 5304.0 |

Text-cache preparation peak: 1694.9 MiB allocated / 1926.0 MiB reserved. Shared GPU memory: `Not reported separately by the available nvidia-smi WDDM query`.
CUDA allocation and reserved-memory counters confirm these forwards ran on the GPU. The WDDM query on this laptop does not expose shared GPU memory as a separate measurement.

## Reproduction

From the repository root with the existing OneTrainer Python environment:

```powershell
.\venv\Scripts\python.exe experiments\sdxl-long-caption\comfy_compat.py --output experiments\sdxl-long-caption\comfy_compat.json
.\venv\Scripts\python.exe experiments\sdxl-long-caption\comfy_compat.py --encoder-precision fp32 --output experiments\sdxl-long-caption\comfy_compat_fp32.json
.\venv\Scripts\python.exe experiments\sdxl-long-caption\short_condition_debug.py
.\venv\Scripts\python.exe experiments\sdxl-long-caption\pretrained_experiment.py
```

The scripts write machine-readable results and generated images under `experiments/sdxl-long-caption/`. The JSON records package versions, checkpoint hash, ComfyUI revision, GPU, precision, seeds, and settings.
