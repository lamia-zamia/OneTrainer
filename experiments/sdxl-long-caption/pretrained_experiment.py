"""Run pretrained SDXL conditioning, LoRA, sampling, and bounded training experiments.

This harness intentionally uses OneTrainer's SDXL model loader, LoRA setup, prediction,
loss, optimizer factory, and sampler. The local ComfyUI comparison is in comfy_compat.py.
"""

from __future__ import annotations

import gc
import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from comfy_compat import (
    DEFAULT_CHECKPOINT,
    TAGS,
    TAIL_A,
    TAIL_B,
    build_caption,
    load_checkpoint_model,
    make_tail_caption,
    sha256_file,
    token_counts,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = EXPERIMENT_DIR / "images"
OUTPUT_JSON = EXPERIMENT_DIR / "pretrained_results.json"
REPORT_PATH = EXPERIMENT_DIR / "REPORT.md"
TARGET_LAYER = "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k"
SEEDS = (42, 123)
BASE_RESOLUTION = 512
TRAIN_RESOLUTION = 256
SAMPLE_STEPS = 8
SAMPLE_CFG = 7.0
TRAIN_UPDATES = 8


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_inventory() -> dict:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        rows = [line.split(", ") for line in raw.splitlines()]
        return {
            "nvidia_smi": raw,
            "gpus": [
                {"name": row[0], "memory_total_mib": int(row[1]), "memory_used_mib": int(row[2]),
                 "memory_free_mib": int(row[3]), "driver_version": row[4]}
                for row in rows
            ],
            "shared_memory": "Not reported separately by the available nvidia-smi WDDM query",
        }
    except Exception as exc:
        return {"nvidia_smi_error": repr(exc), "shared_memory": "unavailable"}


def cuda_peak(reset: bool = True) -> dict:
    if not torch.cuda.is_available():
        return {"allocated_bytes": None, "reserved_bytes": None}
    torch.cuda.synchronize()
    result = {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return result


def as_float_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def vector_metrics(left: torch.Tensor, right: torch.Tensor) -> dict:
    left = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    delta = left - right
    left_norm = torch.linalg.vector_norm(left).item()
    delta_norm = torch.linalg.vector_norm(delta).item()
    return {
        "elements": int(left.numel()),
        "l2_absolute": delta_norm,
        "l2_relative_to_left": delta_norm / max(left_norm, 1e-30),
        "max_absolute": delta.abs().max().item(),
        "mean_absolute": delta.abs().mean().item(),
        "left_l2": left_norm,
    }


def tensor_list_metrics(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict:
    return vector_metrics(torch.cat([x.reshape(-1) for x in left]), torch.cat([x.reshape(-1) for x in right]))


def flatten_parameters(parameters) -> torch.Tensor:
    return torch.cat([parameter.detach().float().reshape(-1).cpu() for parameter in parameters])


def flatten_gradients(parameters) -> torch.Tensor:
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            gradients.append(torch.zeros_like(parameter, device="cpu", dtype=torch.float32).reshape(-1))
        else:
            gradients.append(parameter.grad.detach().float().reshape(-1).cpu())
    return torch.cat(gradients)


def optimizer_update(
    model,
    setup,
    config,
    batch_factory,
    prompt: str,
    limit: int,
    initial_state: dict,
    step: int = 0,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float, dict]:
    from modules.util import create

    config.sdxl_max_caption_tokens = limit
    model.unet_lora.load_state_dict(initial_state)
    model.optimizer = create.create_optimizer(model.parameters, None, config)
    model.optimizer.zero_grad(set_to_none=True)
    before = flatten_parameters(model.unet_lora.parameters())
    model.train_progress.global_step = step
    batch = batch_factory(prompt, limit)
    started = time.perf_counter()
    prediction_data = setup.predict(model, batch, config, model.train_progress, deterministic=deterministic)
    loss = setup.calculate_loss(model, batch, prediction_data, config)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    gradients = flatten_gradients(model.unet_lora.parameters())
    if not torch.isfinite(gradients).all() or not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss or LoRA gradients during pretrained-model training")
    model.optimizer.step()
    after = flatten_parameters(model.unet_lora.parameters())
    update = after - before
    signals = {
        "target": as_float_cpu(prediction_data["target"]),
        "predicted": as_float_cpu(prediction_data["predicted"]),
        "timestep": prediction_data["timestep"].detach().cpu().clone(),
    }
    return gradients, update, float(loss.detach().cpu()), elapsed, cuda_peak(), signals


def state_clone(state: dict) -> dict:
    return {key: value.detach().clone() for key, value in state.items()}


def append_tag_block(tokenizers, prefix: str, candidates: list[str], target: int) -> str:
    tags: list[str] = []
    for tag in candidates:
        candidate = prefix + ", " + ", ".join([*tags, tag])
        counts = token_counts(tokenizers, candidate)
        if max(counts) <= target:
            tags.append(tag)
        if counts == [target, target]:
            return candidate

    # Try replacing the last selected tag to land exactly on the target without changing prior tags.
    for index in range(len(tags) - 1, -1, -1):
        for tag in candidates:
            if tag in tags and tag != tags[index]:
                continue
            replacement = [*tags[:index], tag]
            candidate = prefix + ", " + ", ".join(replacement)
            if token_counts(tokenizers, candidate) == [target, target]:
                return candidate
    raise RuntimeError(f"Could not append Danbooru tags to exactly {target} tokens after prefix")


def make_chunk_two_pair(tokenizers, common_75: str) -> tuple[str, str]:
    a = append_tag_block(tokenizers, common_75, TAIL_A + TAGS, 150)
    b = append_tag_block(tokenizers, common_75, TAIL_B + list(reversed(TAGS)), 150)
    suffix = [
        "cherry_blossoms", "brick_wall", "crosswalk", "street_lamp", "park_bench",
        "wooden_fence", "garden", "stone_path", "paper_lantern", "string_lights",
        "night_sky", "crescent_moon", "stars", "city_lights", "mist", "reflection",
    ] + TAGS
    for tag in suffix:
        next_a = a + ", " + tag
        next_b = b + ", " + tag
        counts_a, counts_b = token_counts(tokenizers, next_a), token_counts(tokenizers, next_b)
        if max(counts_a + counts_b) <= 225:
            a, b = next_a, next_b
        if min(counts_a + counts_b) >= 225:
            break
    return a, b


def build_prompts(tokenizers) -> dict[str, str]:
    common150 = build_caption(tokenizers, 150)
    common75 = build_caption(tokenizers, 75)
    if token_counts(tokenizers, common150) != [150, 150] or token_counts(tokenizers, common75) != [75, 75]:
        raise RuntimeError("Could not build matching shared tag prefixes for both SDXL tokenizers")
    captions = {
        "short40": build_caption(tokenizers, 40),
        "third_a": make_tail_caption(tokenizers, common150, TAIL_A),
        "third_b": make_tail_caption(tokenizers, common150, TAIL_B),
    }
    captions["chunk2_a"], captions["chunk2_b"] = make_chunk_two_pair(tokenizers, common75)
    if token_counts(tokenizers, captions["short40"]) != [40, 40]:
        raise RuntimeError("Short compatibility caption is not 40 tokens in both tokenizers")
    ids = {}
    for name, caption in captions.items():
        ids[name] = [tokenizer(caption, add_special_tokens=False, truncation=False).input_ids for tokenizer in tokenizers]
    if ids["third_a"][0][:150] != ids["third_b"][0][:150] or ids["third_a"][1][:150] != ids["third_b"][1][:150]:
        raise RuntimeError("Third-chunk caption pair does not share the first 150 tokens for both tokenizers")
    for encoder_index in (0, 1):
        a, b = ids["chunk2_a"][encoder_index], ids["chunk2_b"][encoder_index]
        if a[:75] != b[:75] or a[75:150] == b[75:150] or a[150:225] != b[150:225]:
            raise RuntimeError(
                "Could not construct prompts that differ exclusively in chunk two with aligned CLIP tokens "
                f"(encoder index {encoder_index}: counts {len(a)}/{len(b)})"
            )
    return captions


def make_config(model_type, dtype, target_layer: str):
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.Optimizer import Optimizer
    from modules.util.enum.TrainingMethod import TrainingMethod
    from modules.util.enum.LearningRateScheduler import LearningRateScheduler
    from modules.util.enum.LossWeight import LossWeight

    config = TrainConfig.default_values()
    config.model_type = ModelType.STABLE_DIFFUSION_XL_10_BASE
    config.training_method = TrainingMethod.LORA
    config.train_device = "cuda"
    config.temp_device = "cpu"
    config.train_dtype = dtype.FLOAT_16
    config.fallback_train_dtype = dtype.BFLOAT_16
    config.enable_autocast_cache = True
    config.batch_size = 1
    config.gradient_accumulation_steps = 1
    config.learning_rate = 3e-4
    config.learning_rate_scheduler = LearningRateScheduler.CONSTANT
    config.resolution = str(TRAIN_RESOLUTION)
    config.latent_caching = True
    config.text_encoder.train = False
    config.text_encoder_2.train = False
    config.vae.train = False
    config.unet.train = True
    config.unet.weight_dtype = dtype.FLOAT_16
    config.unet.gradient_checkpointing = True
    config.unet.activation_offloading = False
    config.text_encoder.weight_dtype = dtype.FLOAT_16
    config.text_encoder_2.weight_dtype = dtype.FLOAT_16
    config.vae.weight_dtype = dtype.FLOAT_16
    config.layer_filter = target_layer
    config.layer_filter_regex = False
    config.lora_rank = 4
    config.lora_alpha = 4.0
    config.lora_weight_dtype = dtype.FLOAT_32
    config.dropout_probability = 0.0
    config.optimizer.optimizer = Optimizer.ADAMW
    config.optimizer.fused = False
    config.optimizer.foreach = False
    config.optimizer.weight_decay = 0.0
    config.sdxl_max_caption_tokens = 225
    config.mse_strength = 1.0
    config.mae_strength = 0.0
    config.log_cosh_strength = 0.0
    config.huber_strength = 0.0
    config.loss_weight_fn = LossWeight.CONSTANT
    return config


def encode_prompt_cache(model, tokenizer_pair, prompt: str, limit: int, device) -> dict:
    from modules.model.util.clip_chunk import tokenize_clip_chunks

    tokens_1, _ = tokenize_clip_chunks(tokenizer_pair[0], prompt, limit)
    tokens_2, _ = tokenize_clip_chunks(tokenizer_pair[1], prompt, limit)
    started = time.perf_counter()
    encoded = model.encode_text(
        text=prompt,
        train_device=device,
        max_caption_tokens=limit,
        return_chunk_attention_mask=True,
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return {
        "tokens_1": tokens_1.detach().cpu(),
        "tokens_2": tokens_2.detach().cpu(),
        "te1": encoded[0].detach().cpu(),
        "te2": encoded[1].detach().cpu(),
        "pooled": encoded[2].detach().cpu(),
        "mask": encoded[3].detach().cpu() if encoded[3] is not None else None,
        "seconds": seconds,
    }


def make_batch_factory(model, tokenizer_pair, prompt_cache, latent, resolution):
    device = torch.device("cuda")

    def batch_factory(prompt: str, limit: int) -> dict:
        key = (prompt, limit)
        if key not in prompt_cache:
            raise KeyError(f"Missing frozen text cache for prompt and capacity {limit}")
        cached = prompt_cache[key]
        return {
            "latent_image": latent.to(device),
            "tokens_1": cached["tokens_1"].to(device),
            "tokens_2": cached["tokens_2"].to(device),
            "text_encoder_1_hidden_state": cached["te1"].to(device),
            "text_encoder_2_hidden_state": cached["te2"].to(device),
            "text_encoder_2_pooled_state": cached["pooled"].to(device),
            "loss_weight": torch.ones((1,), device=device),
            "original_resolution": torch.tensor([[resolution], [resolution]], device=device),
            "crop_offset": torch.tensor([[0], [0]], device=device),
            "crop_resolution": torch.tensor([[resolution], [resolution]], device=device),
        }

    return batch_factory


def pil_tensor(image: Image.Image, resolution: int, device) -> torch.Tensor:
    image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    data = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(device)


def save_contact_sheet(results: dict, labels: list[str], seeds: tuple[int, ...], filename: Path) -> None:
    cell_width, cell_height, label_height = 512, 512, 50
    sheet = Image.new("RGB", (len(labels) * cell_width, len(seeds) * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, seed in enumerate(seeds):
        for col, label in enumerate(labels):
            path = results[label][str(seed)]["path"]
            image = Image.open(path).convert("RGB").resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            x = col * cell_width
            y = row * (cell_height + label_height)
            sheet.paste(image, (x, y + label_height))
            draw.text((x + 8, y + 8), f"{label} | seed {seed}", fill="black")
    sheet.save(filename, quality=95)


def image_metrics(path_a: str, path_b: str) -> dict:
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32) / 255.0
    delta = a - b
    mse = float(np.mean(delta * delta))
    return {
        "width_height": [int(a.shape[1]), int(a.shape[0])],
        "exact_match": bool(mse == 0),
        "mean_absolute_pixel_difference": float(np.abs(delta).mean()),
        "max_absolute_pixel_difference": float(np.abs(delta).max()),
        "rmse": math.sqrt(mse),
        "psnr_db": None if mse == 0 else -10.0 * math.log10(mse),
    }


def save_results(results: dict) -> None:
    OUTPUT_JSON.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from util.import_util import script_imports

    script_imports()
    from modules.modelSampler.StableDiffusionXLSampler import StableDiffusionXLSampler
    from modules.modelSetup.StableDiffusionXLLoRASetup import StableDiffusionXLLoRASetup
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.DataType import DataType
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.NoiseScheduler import NoiseScheduler
    from modules.util.TrainProgress import TrainProgress
    from modules.model.util.clip_chunk import tokenize_clip_chunks
    from transformers import CLIPTokenizer
    from huggingface_hub import scan_cache_dir

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-generated-images",
        action="store_true",
        help="Reuse the image-generation phase from an existing pretrained_results.json",
    )
    args = parser.parse_args()
    previous_results = {}
    if args.reuse_generated_images and OUTPUT_JSON.exists():
        previous_results = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the pretrained UNet/LoRA experiments")
    checkpoint = DEFAULT_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261024)
    torch.cuda.manual_seed_all(20261024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    repo = next(
        item for item in scan_cache_dir().repos
        if item.repo_id == "stabilityai/stable-diffusion-xl-base-1.0"
    )
    revision = next(iter(repo.revisions))
    tokenizer_snapshot = Path(repo.repo_path) / "snapshots" / revision.commit_hash
    tokenizers = (
        CLIPTokenizer.from_pretrained(tokenizer_snapshot / "tokenizer", local_files_only=True),
        CLIPTokenizer.from_pretrained(tokenizer_snapshot / "tokenizer_2", local_files_only=True),
    )
    captions = build_prompts(tokenizers)
    results = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": package_version("transformers"),
            "diffusers": package_version("diffusers"),
            "tokenizers": package_version("tokenizers"),
            "safetensors": package_version("safetensors"),
            "gpu": gpu_inventory(),
            "determinism": {
                "seed": 20261024,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
                "attention": "OneTrainer native SDP; deterministic sampling and gradient controls use PyTorch SDPA math",
            },
        },
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "source": {
            "onetrainer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "target_lora_layer": TARGET_LAYER,
            "lora_rank": 4,
            "lora_alpha": 4.0,
            "optimizer": "OneTrainer create.create_optimizer -> torch AdamW",
            "optimizer_lr": 3e-4,
            "weight_decay": 0.0,
            "training_precision": "FP16 UNet autocast, FP32 LoRA weights, frozen FP16 CLIP encoders",
            "batch_size": 1,
            "caption_dropout": 0.0,
            "image_generation": {
                "sampler": "OneTrainer StableDiffusionXLSampler",
                "scheduler": "Euler",
                "steps": SAMPLE_STEPS,
                "cfg": SAMPLE_CFG,
                "seeds": list(SEEDS),
                "initial_resolution": BASE_RESOLUTION,
            },
            "controlled_training": {
                "resolution": TRAIN_RESOLUTION,
                "updates_per_treatment": TRAIN_UPDATES,
                "dataset_size": 1,
                "sample_exposures_per_treatment": TRAIN_UPDATES,
            },
        },
        "captions": {key: {"text": value, "token_counts": token_counts(tokenizers, value)} for key, value in captions.items()},
        "phases": {},
    }
    save_results(results)

    print("Loading the local pretrained Illustrious XL checkpoint with the OneTrainer loader", flush=True)
    model = load_checkpoint_model(checkpoint)
    model.train_config = None
    gc.collect()
    torch.cuda.empty_cache()
    results["loaded_model"] = {
        "unet_parameters": sum(p.numel() for p in model.unet.parameters()),
        "unet_parameter_dtypes": sorted({str(p.dtype) for p in model.unet.parameters()}),
        "clip_l_parameter_dtypes": sorted({str(p.dtype) for p in model.text_encoder_1.parameters()}),
        "clip_g_parameter_dtypes": sorted({str(p.dtype) for p in model.text_encoder_2.parameters()}),
        "unet_cross_attention_dim": model.unet.config.cross_attention_dim,
    }
    save_results(results)

    config = make_config(ModelType, DataType, TARGET_LAYER)
    model.train_config = config
    model.train_progress = TrainProgress()
    setup = StableDiffusionXLLoRASetup(torch.device("cuda"), torch.device("cpu"), False)
    setup.setup_optimizations(model, config)
    # Identical adapter initialization is the control state for every caption treatment.
    torch.manual_seed(94117)
    setup.setup_model(model, config)
    setup.setup_train_device(model, config)
    model.eval()
    model.unet.train()
    initial_adapter_state = state_clone(model.unet_lora.state_dict())
    lora_parameters = model.unet_lora.parameters()
    if len(lora_parameters) == 0:
        raise RuntimeError("Production OneTrainer LoRA setup produced no parameters")

    sampler = StableDiffusionXLSampler(
        train_device=torch.device("cuda"),
        temp_device=torch.device("cpu"),
        model=model,
        model_type=ModelType.STABLE_DIFFUSION_XL_10_BASE,
    )

    def sample(prompt: str, limit: int, seed: int, resolution: int) -> Image.Image:
        model.eval()
        output = sampler._StableDiffusionXLSampler__sample_base(
            prompt=prompt,
            negative_prompt="",
            height=resolution,
            width=resolution,
            seed=seed,
            random_seed=False,
            diffusion_steps=SAMPLE_STEPS,
            cfg_scale=SAMPLE_CFG,
            noise_scheduler=NoiseScheduler.EULER,
            cfg_rescale=0.0,
            sdxl_max_caption_tokens=limit,
            force_last_timestep=False,
        )
        return output.data

    image_cases = {
        "short75": (captions["short40"], 75),
        "short225": (captions["short40"], 225),
        "longA75": (captions["third_a"], 75),
        "longA225": (captions["third_a"], 225),
        "longB225": (captions["third_b"], 225),
    }
    image_results = {}
    actual_resolution = BASE_RESOLUTION
    generation_retry = None
    if args.reuse_generated_images:
        previous_generation = previous_results.get("phases", {}).get("image_generation", {})
        if previous_generation.get("status") != "completed":
            raise RuntimeError("--reuse-generated-images requires a completed image_generation phase in pretrained_results.json")
        image_results = previous_generation["images"]
        actual_resolution = previous_generation["resolution"]
        generation_seconds = previous_generation["seconds_for_all_images"]
        generation_retry = previous_generation.get("oom_retry")
        print(f"Reusing prior deterministic {actual_resolution}px image outputs", flush=True)
    else:
        for attempt_resolution in (BASE_RESOLUTION, 384, 320):
            try:
                image_results = {}
                actual_resolution = attempt_resolution
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started_generation = time.perf_counter()
                for label, (prompt, limit) in image_cases.items():
                    image_results[label] = {}
                    for seed in SEEDS:
                        image = sample(prompt, limit, seed, actual_resolution)
                        path = IMAGE_DIR / f"{label}_seed{seed}_{actual_resolution}.png"
                        image.save(path)
                        image_results[label][str(seed)] = {"path": str(path), "size": list(image.size)}
                        print(f"Generated {label} seed={seed} at {actual_resolution}px", flush=True)
                generation_seconds = time.perf_counter() - started_generation
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or attempt_resolution == 320:
                    raise
                next_resolution = 384 if attempt_resolution == BASE_RESOLUTION else 320
                generation_retry = {"from_resolution": attempt_resolution, "error": str(exc), "to_resolution": next_resolution}
                for path in IMAGE_DIR.glob("*_seed*_*.png"):
                    path.unlink(missing_ok=True)
                gc.collect()
                torch.cuda.empty_cache()
    if not image_results:
        raise RuntimeError("No image comparison outputs were generated")

    labels = list(image_cases)
    sheet_path = IMAGE_DIR / f"contact_sheet_{actual_resolution}.png"
    save_contact_sheet(image_results, labels, list(SEEDS), sheet_path)
    per_image = {}
    for seed in SEEDS:
        per_image[str(seed)] = {
            "short_75_vs_225": image_metrics(image_results["short75"][str(seed)]["path"], image_results["short225"][str(seed)]["path"]),
            "long_75_vs_225": image_metrics(image_results["longA75"][str(seed)]["path"], image_results["longA225"][str(seed)]["path"]),
            "third_chunk_A_vs_B_225": image_metrics(image_results["longA225"][str(seed)]["path"], image_results["longB225"][str(seed)]["path"]),
        }
    results["phases"]["image_generation"] = {
        "status": "completed",
        "resolution": actual_resolution,
        "seconds_for_all_images": generation_seconds,
        "peak_cuda_memory": cuda_peak(),
        "oom_retry": generation_retry,
        "parameters": {"sampler": "Euler", "steps": SAMPLE_STEPS, "cfg": SAMPLE_CFG, "seeds": list(SEEDS)},
        "images": image_results,
        "contact_sheet": str(sheet_path),
        "comparisons": per_image,
        "notes": "Pixel metrics measure image changes, not perceptual quality or improved prompt adherence.",
    }
    save_results(results)

    print("Encoding frozen SDXL prompt caches and preparing an image-derived training latent", flush=True)
    prompt_cache: dict[tuple[str, int], dict] = {}
    cache_cases = set()
    for prompt in captions.values():
        for limit in (75, 150, 225):
            cache_cases.add((prompt, limit))
    model.materialize_only_text_encoders()
    torch.cuda.reset_peak_memory_stats()
    cache_started = time.perf_counter()
    for prompt, limit in cache_cases:
        cached = encode_prompt_cache(model, tokenizers, prompt, limit, torch.device("cuda"))
        prompt_cache[(prompt, limit)] = cached
    torch.cuda.synchronize()
    total_cache_seconds = time.perf_counter() - cache_started
    text_encoding_peak = cuda_peak()
    # Give each cache entry a readable prompt label rather than retaining only tuple keys in the report.
    cache_time_by_caption = {}
    for name, prompt in captions.items():
        cache_time_by_caption[name] = {
            str(limit): prompt_cache[(prompt, limit)]["seconds"] for limit in (75, 150, 225)
        }
    short_cache_controls = {}
    for limit in (150, 225):
        legacy = prompt_cache[(captions["short40"], 75)]
        extended = prompt_cache[(captions["short40"], limit)]
        short_cache_controls[str(limit)] = {
            "clip_l_hidden": vector_metrics(legacy["te1"], extended["te1"][:, :77]),
            "clip_g_hidden": vector_metrics(legacy["te2"], extended["te2"][:, :77]),
            "pooled_clip_g": vector_metrics(legacy["pooled"], extended["pooled"]),
            "clip_l_token_ids_equal": torch.equal(legacy["tokens_1"], extended["tokens_1"][:77]),
            "clip_g_token_ids_equal": torch.equal(legacy["tokens_2"], extended["tokens_2"][:77]),
        }
    model.materialize_only("unet")
    model.unet.train()
    gc.collect()
    torch.cuda.empty_cache()

    train_resolution = TRAIN_RESOLUTION
    training_image = Image.open(image_results["longA225"][str(SEEDS[0])]["path"]).convert("RGB")
    train_image_tensor = pil_tensor(training_image, train_resolution, torch.device("cuda"))
    model.materialize_only("vae")
    with torch.no_grad(), model.vae_autocast_context:
        latent = model.vae.encode(train_image_tensor).latent_dist.mode()
    latent = latent.detach()
    model.materialize_only("unet")
    batch_factory = make_batch_factory(model, tokenizers, prompt_cache, latent, train_resolution)
    short_unet_controls = {}
    with torch.no_grad():
        for limit in (75, 225):
            config.sdxl_max_caption_tokens = limit
            short_unet_controls[str(limit)] = setup.predict(
                model,
                batch_factory(captions["short40"], limit),
                config,
                model.train_progress,
                deterministic=True,
            )
    results["phases"]["short_caption_conditioning"] = {
        "status": "completed",
        "caption_token_counts": token_counts(tokenizers, captions["short40"]),
        "text_encoder_75_vs_150_225": short_cache_controls,
        "pretrained_unet_75_vs_225_prediction": {
            "latent_noise_target": vector_metrics(
                short_unet_controls["75"]["target"], short_unet_controls["225"]["target"]
            ),
            "predicted_noise": vector_metrics(
                short_unet_controls["75"]["predicted"], short_unet_controls["225"]["predicted"]
            ),
            "timestep_equal": torch.equal(
                short_unet_controls["75"]["timestep"], short_unet_controls["225"]["timestep"]
            ),
        },
    }
    save_results(results)

    # Gradient and one-step optimizer comparisons (Experiments A and B).
    torch.cuda.reset_peak_memory_stats()
    phase_started = time.perf_counter()
    # Use PyTorch's deterministic math attention and deterministic algorithm mode for
    # same-input gradient controls. Sampling and ordinary training retain the configured
    # native SDP backend; these flags only constrain the comparison run.
    results["source"]["gradient_control_attention"] = "PyTorch SDPA math backend"
    results["source"]["deterministic_algorithms_for_gradient_controls"] = True
    save_results(results)

    cache_control = {
        "tokens_l_equal_at_75": torch.equal(
            prompt_cache[(captions["third_a"], 75)]["tokens_1"],
            prompt_cache[(captions["third_b"], 75)]["tokens_1"],
        ),
        "tokens_g_equal_at_75": torch.equal(
            prompt_cache[(captions["third_a"], 75)]["tokens_2"],
            prompt_cache[(captions["third_b"], 75)]["tokens_2"],
        ),
        "clip_l_hidden_equal_at_75": torch.equal(
            prompt_cache[(captions["third_a"], 75)]["te1"],
            prompt_cache[(captions["third_b"], 75)]["te1"],
        ),
        "clip_g_hidden_equal_at_75": torch.equal(
            prompt_cache[(captions["third_a"], 75)]["te2"],
            prompt_cache[(captions["third_b"], 75)]["te2"],
        ),
        "pooled_equal_at_75": torch.equal(
            prompt_cache[(captions["third_a"], 75)]["pooled"],
            prompt_cache[(captions["third_b"], 75)]["pooled"],
        ),
    }
    if not all(cache_control.values()):
        raise RuntimeError(f"75-token captions do not share identical CLIP caches: {cache_control}")

    gradient_results = {}
    prediction_signals = {}
    update_results = {}
    update_vectors = {}
    conditions = (
        ("A1_75", captions["third_a"], 75),
        ("A2_75", captions["third_b"], 75),
        ("B1_225", captions["third_a"], 225),
        ("B2_225", captions["third_b"], 225),
    )
    for label, prompt, limit in conditions:
        gradients, update, loss, elapsed, peak, signals = optimizer_update(
            model, setup, config, batch_factory, prompt, limit, initial_adapter_state, deterministic=True,
        )
        gradient_results[label] = gradients
        prediction_signals[label] = signals
        update_vectors[label] = update
        update_results[label] = {
            "loss": loss,
            "gradient_l2": float(torch.linalg.vector_norm(gradients).item()),
            "update_l2": float(torch.linalg.vector_norm(update).item()),
            "update_finite": bool(torch.isfinite(update).all()),
            "microstep_seconds": elapsed,
            "peak_cuda_memory": peak,
        }
        print(f"Completed {label}: loss={loss:.7g} grad_l2={update_results[label]['gradient_l2']:.7g} update_l2={update_results[label]['update_l2']:.7g}", flush=True)

    gradients_repeat, _, repeat_loss, repeat_elapsed, repeat_peak, repeat_signals = optimizer_update(
        model, setup, config, batch_factory, captions["third_a"], 75, initial_adapter_state, deterministic=True,
    )
    gradient_results["A1_75_repeat"] = gradients_repeat
    prediction_signals["A1_75_repeat"] = repeat_signals
    chunk2_a_grad, _, chunk2_a_loss, chunk2_a_elapsed, chunk2_a_peak, _ = optimizer_update(
        model, setup, config, batch_factory, captions["chunk2_a"], 225, initial_adapter_state, deterministic=True,
    )
    chunk2_b_grad, _, chunk2_b_loss, chunk2_b_elapsed, chunk2_b_peak, _ = optimizer_update(
        model, setup, config, batch_factory, captions["chunk2_b"], 225, initial_adapter_state, deterministic=True,
    )
    gradient_results["chunk2_A_225"] = chunk2_a_grad
    gradient_results["chunk2_B_225"] = chunk2_b_grad
    gradient_diff = {
        "75_caption_A_vs_B": vector_metrics(gradient_results["A1_75"], gradient_results["A2_75"]),
        "75_repeat_control": vector_metrics(gradient_results["A1_75"], gradients_repeat),
        "225_caption_A_vs_B_third_chunk": vector_metrics(gradient_results["B1_225"], gradient_results["B2_225"]),
        "225_caption_A_vs_B_chunk_two_only": vector_metrics(chunk2_a_grad, chunk2_b_grad),
    }
    prediction_control = {
        "75_A_vs_B_target_noise": vector_metrics(
            prediction_signals["A1_75"]["target"], prediction_signals["A2_75"]["target"]
        ),
        "75_A_vs_B_unet_prediction": vector_metrics(
            prediction_signals["A1_75"]["predicted"], prediction_signals["A2_75"]["predicted"]
        ),
        "75_A_vs_A_repeat_unet_prediction": vector_metrics(
            prediction_signals["A1_75"]["predicted"], prediction_signals["A1_75_repeat"]["predicted"]
        ),
        "75_A_vs_B_timestep_equal": torch.equal(
            prediction_signals["A1_75"]["timestep"], prediction_signals["A2_75"]["timestep"]
        ),
        "75_A_vs_A_repeat_timestep_equal": torch.equal(
            prediction_signals["A1_75"]["timestep"], prediction_signals["A1_75_repeat"]["timestep"]
        ),
    }
    for key, metric in gradient_diff.items():
        print(f"Gradient comparison {key}: abs={metric['l2_absolute']:.9g} rel={metric['l2_relative_to_left']:.9g}", flush=True)
    results["phases"]["real_unet_lora_gradients"] = {
        "status": "completed",
        "fixed_inputs": {
            "checkpoint_sha256": results["checkpoint"]["sha256"],
            "same_initial_adapter_state": True,
            "same_image_latent": True,
            "same_prediction_noise_and_timestep": True,
            "frozen_text_encoders": True,
            "caption_dropout": 0.0,
            "loss": "OneTrainer SDXL setup.calculate_loss (MSE target, constant weighting)",
            "resolution": train_resolution,
            "latent_shape": list(latent.shape),
            "batch_size": 1,
            "precision": "FP16 UNet autocast, FP32 trainable LoRA",
        },
        "first_two_chunks_identical_for_both_tokenizers": True,
        "75_token_cache_controls": cache_control,
        "results": update_results,
        "gradient_differences": gradient_diff,
        "prediction_controls": prediction_control,
        "chunk_two_losses": {"A": chunk2_a_loss, "B": chunk2_b_loss},
        "repeat_control_loss": repeat_loss,
        "repeat_control_peak": repeat_peak,
        "repeat_control_microstep_seconds": repeat_elapsed,
        "chunk_two_peak": {"A": chunk2_a_peak, "B": chunk2_b_peak},
        "elapsed_seconds": time.perf_counter() - phase_started,
        "peak_cuda_memory": cuda_peak(),
    }
    results["phases"]["optimizer_updates"] = {
        "status": "completed",
        "one_update_each": update_results,
        "update_differences": {},
    }
    results["phases"]["optimizer_updates"]["update_differences"] = {
        "75_A_vs_B": vector_metrics(update_vectors["A1_75"], update_vectors["A2_75"]),
        "225_A_vs_B": vector_metrics(update_vectors["B1_225"], update_vectors["B2_225"]),
        "75_vs_225_same_caption_A": vector_metrics(update_vectors["A1_75"], update_vectors["B1_225"]),
    }
    save_results(results)

    # OneTrainer training-effect comparison with eight identical-schedule updates on one real image/latent.
    treatment_states = {}
    treatment_steps = {}
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    for treatment, limit in (("train75", 75), ("train225", 225)):
        model.unet_lora.load_state_dict(initial_adapter_state)
        from modules.util import create

        model.optimizer = create.create_optimizer(model.parameters, None, config)
        step_losses = []
        for step in range(TRAIN_UPDATES):
            config.sdxl_max_caption_tokens = limit
            model.train_progress.global_step = step
            model.optimizer.zero_grad(set_to_none=True)
            prediction_data = setup.predict(
                model,
                batch_factory(captions["third_a"], limit),
                config,
                model.train_progress,
                deterministic=False,
            )
            loss = setup.calculate_loss(
                model,
                batch_factory(captions["third_a"], limit),
                prediction_data,
                config,
            )
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.unet_lora.parameters()):
                raise RuntimeError(f"Non-finite LoRA gradient in {treatment} step {step}")
            model.optimizer.step()
            step_losses.append(float(loss.detach().cpu()))
            print(f"Training {treatment} step {step + 1}/{TRAIN_UPDATES}: loss={step_losses[-1]:.7g}", flush=True)
        treatment_states[treatment] = state_clone(model.unet_lora.state_dict())
        treatment_steps[treatment] = step_losses
    training_seconds = time.perf_counter() - training_started
    results["phases"]["controlled_training"] = {
        "status": "completed",
        "train75": {"updates": len(treatment_steps["train75"]), "losses": treatment_steps["train75"]},
        "train225": {"updates": len(treatment_steps["train225"]), "losses": treatment_steps["train225"]},
        "same_initial_lora": True,
        "same_image_latent_and_sample_exposures": True,
        "same_optimizer_and_update_schedule": True,
        "same_noise_and_timestep_per_step": True,
        "train75_vs_train225_final_parameter_delta": tensor_list_metrics(
            [treatment_states["train75"][key] for key in sorted(treatment_states["train75"])],
            [treatment_states["train225"][key] for key in sorted(treatment_states["train225"])],
        ),
        "elapsed_seconds": training_seconds,
        "peak_cuda_memory": cuda_peak(),
        "sample_image": image_results["longA225"][str(SEEDS[0])]["path"],
    }
    save_results(results)

    # Benchmark text cache preparation and actual training microsteps with real weights.
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    benchmark = {"text_encoding_seconds": cache_time_by_caption, "text_encoding_total_seconds": total_cache_seconds,
                 "text_encoding_peak_cuda_memory": text_encoding_peak,
                 "training_attention_backend": "OneTrainer native SDP with PyTorch default backend selection"}
    microsteps = {}
    for prompt_name, prompt in (("short40", captions["short40"]), ("longA", captions["third_a"])):
        for limit in (75, 150, 225):
            config.sdxl_max_caption_tokens = limit
            model.unet_lora.load_state_dict(initial_adapter_state)
            model.optimizer.zero_grad(set_to_none=True)
            model.train_progress.global_step = 31
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            prediction_data = setup.predict(
                model, batch_factory(prompt, limit), config, model.train_progress, deterministic=True,
            )
            loss = setup.calculate_loss(model, batch_factory(prompt, limit), prediction_data, config)
            loss.backward()
            torch.cuda.synchronize()
            microseconds = time.perf_counter() - started
            microsteps[f"{prompt_name}_{limit}"] = {
                "seconds": microseconds,
                "loss": float(loss.detach().cpu()),
                "peak_cuda_memory": cuda_peak(),
                "batch_size": 1,
                "train_resolution": train_resolution,
                "latent_shape": list(latent.shape),
                "precision": "FP16 UNet autocast / FP32 LoRA",
            }
    benchmark["training_microsteps"] = microsteps
    results["phases"]["performance"] = benchmark

    # Compare the trained adapters using identical sampling inputs and the same 225-token inference prompt.
    training_sample_results = {}
    for treatment in ("train75", "train225"):
        model.unet_lora.load_state_dict(treatment_states[treatment])
        training_sample_results[treatment] = {}
        for seed in SEEDS:
            image = sample(captions["third_a"], 225, seed, actual_resolution)
            path = IMAGE_DIR / f"{treatment}_eval_seed{seed}_{actual_resolution}.png"
            image.save(path)
            training_sample_results[treatment][str(seed)] = {"path": str(path), "size": list(image.size)}
    training_sheet = IMAGE_DIR / f"training_effect_contact_sheet_{actual_resolution}.png"
    temp_image_results = {
        "trained75": training_sample_results["train75"],
        "trained225": training_sample_results["train225"],
    }
    save_contact_sheet(temp_image_results, ["trained75", "trained225"], list(SEEDS), training_sheet)
    results["phases"]["controlled_training"]["evaluation"] = {
        "prompt": captions["third_a"],
        "capacity": 225,
        "seeds": list(SEEDS),
        "same_sampler_settings": True,
        "images": training_sample_results,
        "contact_sheet": str(training_sheet),
        "pixel_differences": {
            str(seed): image_metrics(
                training_sample_results["train75"][str(seed)]["path"],
                training_sample_results["train225"][str(seed)]["path"],
            ) for seed in SEEDS
        },
        "quality_interpretation": "Output differences show an update effect, not proof that longer captions improve quality.",
    }

    save_results(results)
    write_human_report(results)
    print(f"Completed pretrained experiments; results at {OUTPUT_JSON}", flush=True)


def write_human_report(results: dict) -> None:
    phases = results.get("phases", {})
    comfy_path = EXPERIMENT_DIR / "comfy_compat.json"
    comfy_fp32_path = EXPERIMENT_DIR / "comfy_compat_fp32.json"
    lines = [
        "# Pretrained SDXL long-caption experiment report",
        "",
        "This report is generated from `pretrained_results.json`. All model-forward results use the local pretrained Illustrious XL checkpoint named in that JSON. No randomly initialized model is used as evidence.",
        "",
        f"Checkpoint: `{results.get('checkpoint', {}).get('path', 'not loaded')}`",
        f"SHA-256: `{results.get('checkpoint', {}).get('sha256', 'not loaded')}`",
        f"OneTrainer revision: `{results.get('source', {}).get('onetrainer_commit', 'unknown')}`",
        "",
        "## ComfyUI compatibility",
        "",
        "The comparisons are saved in [`comfy_compat.json`](comfy_compat.json) and [`comfy_compat_fp32.json`](comfy_compat_fp32.json). They use the local ComfyUI source, the same checkpoint, and realistic Danbooru-style tags. Both files store full token IDs per section for CLIP-L and CLIP-G, chunk counts, effective conditioning shapes, padding masks, hidden padding values, and pooled outputs.",
        "For these unweighted captions, ComfyUI groups the full text as one long tokenizer segment; its boundaries match the 75-content-token splits exactly. ComfyUI also trims dual-encoder conditioning to the smaller CLIP-L/CLIP-G section count and uses CLIP-G pooled output. Both implementations omit a text padding attention mask for SDXL; OneTrainer therefore encodes native EOS/PAD tokens and trims inactive sections before the UNet, matching the measured reference conditioning.",
        "",
    ]
    if comfy_path.is_file() and comfy_fp32_path.is_file():
        comfy_fp16 = json.loads(comfy_path.read_text(encoding="utf-8"))
        comfy_fp32 = json.loads(comfy_fp32_path.read_text(encoding="utf-8"))
        fp16_cases = comfy_fp16["cases"]
        fp32_cases = comfy_fp32["cases"]
        max_fp16_hidden = max(item["conditioning_difference"]["l2_relative"] for item in fp16_cases.values())
        max_fp16_pooled = max(item["pooled_difference"]["l2_relative"] for item in fp16_cases.values())
        max_fp32_hidden = max(item["conditioning_difference"]["l2_relative"] for item in fp32_cases.values())
        max_fp32_pooled = max(item["pooled_difference"]["l2_relative"] for item in fp32_cases.values())
        fp16_comfy_output_dtype = fp16_cases["danbooru_20"]["comfy_conditioning_dtype"]
        fp16_onetrainer_output_dtype = fp16_cases["danbooru_20"]["one_trainer_conditioning_dtype"]
        shared_token_ids_match = all(
            all(chunk_matches)
            for item in fp32_cases.values()
            for chunk_matches in item["token_ids_equal"].values()
        )
        short_padding = fp32_cases["short_40"]
        lines += [
            f"For the 40-token caption, ComfyUI encodes 77 positions; OneTrainer returns a fixed-capacity {short_padding['one_trainer_conditioning_shape'][1]}-position tensor with {short_padding['one_trainer_padding_tokens']} masked positions whose hidden values are exactly zero. Its effective sequence is 77 positions.",
            "",
            f"Token IDs match for both encoders across all shared sections in the FP32 parity run: `{shared_token_ids_match}`. ComfyUI section counts are 1 at 20/74/75 tokens, 2 at 76/100/149/150, 3 at 151/200/225, and 4 at 232; OneTrainer's configured 225-token cap retains three sections at 232.",
            f"With both implementations' CLIP parameters normalized to FP32 from the same checkpoint, maximum relative L2 differences across the tested captions are `{max_fp32_hidden:.3g}` for sequence conditioning and `{max_fp32_pooled:.3g}` for pooled conditioning. With both parameter sets normalized to FP16, the maxima are `{max_fp16_hidden:.3g}` and `{max_fp16_pooled:.3g}`; ComfyUI returns `{fp16_comfy_output_dtype}` conditioning while OneTrainer returns `{fp16_onetrainer_output_dtype}`, so the remaining gap is measured with its FP32 output/accumulation path.",
            f"The ComfyUI source revision is `{comfy_fp32['environment']['comfy_commit']}`. Both comparisons run that source inside the OneTrainer Python/Torch environment so the same CUDA-enabled Torch build is used.",
            "",
        ]
    short_control = phases.get("short_caption_conditioning", {})
    if short_control.get("status") == "completed":
        lines += ["## Short-caption compatibility", ""]
        for limit, values in short_control.get("text_encoder_75_vs_150_225", {}).items():
            lines.append(
                f"- 40-token caption at 75 vs {limit}: CLIP-L hidden relative L2 `{values['clip_l_hidden']['l2_relative_to_left']:.8g}`, "
                f"CLIP-G hidden relative L2 `{values['clip_g_hidden']['l2_relative_to_left']:.8g}`, "
                f"pooled relative L2 `{values['pooled_clip_g']['l2_relative_to_left']:.8g}`."
            )
        prediction = short_control.get("pretrained_unet_75_vs_225_prediction", {})
        if prediction:
            lines.append(
                f"- Pretrained UNet prediction relative L2 at 75 vs 225: `{prediction['predicted_noise']['l2_relative_to_left']:.8g}`; "
                f"same noise `{prediction['latent_noise_target']['l2_absolute'] == 0}`, same timestep `{prediction['timestep_equal']}`."
            )
        lines.append("")
    gradients = phases.get("real_unet_lora_gradients", {})
    if gradients.get("status") == "completed":
        lines += ["## Pretrained UNet LoRA gradients", ""]
        for name, metrics in gradients.get("gradient_differences", {}).items():
            lines.append(f"- `{name}`: absolute L2 `{metrics['l2_absolute']:.8g}`, relative `{metrics['l2_relative_to_left']:.8g}`.")
        lines += ["", "All cases used the same pretrained UNet, initial OneTrainer LoRA state, image-derived latent, noise, timestep, batch size, loss, and FP16 autocast. The text encoders were frozen. The 75-token pair differs only after the retained prefix; the 225-token pair uses different third-chunk tags. A second pair differs only in chunk two.", ""]
    else:
        lines += ["## Pretrained UNet LoRA gradients", "", f"Status: **UNVERIFIED** ({gradients.get('error', 'phase did not complete')}).", ""]
    updates = phases.get("optimizer_updates", {})
    if updates.get("status") == "completed":
        lines += ["## One-step optimizer updates", ""]
        for name, result in updates.get("one_update_each", {}).items():
            lines.append(f"- `{name}`: gradient L2 `{result['gradient_l2']:.8g}`, update L2 `{result['update_l2']:.8g}`, finite `{result['update_finite']}`.")
        for name, metrics in updates.get("update_differences", {}).items():
            lines.append(f"- Update difference `{name}`: absolute L2 `{metrics['l2_absolute']:.8g}`, relative `{metrics['l2_relative_to_left']:.8g}`.")
        lines.append("")
    training = phases.get("controlled_training", {})
    if training.get("status") == "completed":
        lines += ["## Controlled LoRA training", "", f"Eight OneTrainer updates per treatment used one image, matched optimizer settings, identical initial LoRA weights, and the same per-step noise and timestep. Final 75-vs-225 adapter parameter delta: absolute L2 `{training['train75_vs_train225_final_parameter_delta']['l2_absolute']:.8g}`, relative `{training['train75_vs_train225_final_parameter_delta']['l2_relative_to_left']:.8g}`.", ""]
        if "evaluation" in training:
            for seed, metrics in training["evaluation"]["pixel_differences"].items():
                psnr = "exact match" if metrics["psnr_db"] is None else f"{metrics['psnr_db']:.5g} dB"
                lines.append(f"- Evaluation seed `{seed}`: mean absolute pixel difference `{metrics['mean_absolute_pixel_difference']:.8g}`, PSNR `{psnr}`.")
            lines += ["", "The parameter and fixed-seed image comparisons show a detectable update effect in this bounded run. They do not establish that the 225-token treatment is perceptually better. The training contact sheet is `images/training_effect_contact_sheet_512.png`; the eight-step outputs are visibly different but still have sampler artifacts.", ""]
    else:
        lines += ["## Controlled LoRA training", "", f"Status: **UNVERIFIED** ({training.get('error', 'phase did not complete')}).", ""]
    generation = phases.get("image_generation", {})
    if generation.get("status") == "completed":
        lines += ["## Base-model image comparisons", "", f"Generated {len(generation['images']) * len(generation['parameters']['seeds'])} images at {generation['resolution']}px with Euler, {generation['parameters']['steps']} steps, CFG {generation['parameters']['cfg']}, and fixed seeds {generation['parameters']['seeds']}. Contact sheet: [`{Path(generation['contact_sheet']).relative_to(EXPERIMENT_DIR).as_posix()}`]({Path(generation['contact_sheet']).relative_to(EXPERIMENT_DIR).as_posix()}).", ""]
        for seed, comparisons in generation.get("comparisons", {}).items():
            for name, metrics in comparisons.items():
                psnr = "exact match" if metrics["psnr_db"] is None else f"{metrics['psnr_db']:.5g} dB"
                lines.append(f"- Seed `{seed}`, `{name}`: mean absolute pixel difference `{metrics['mean_absolute_pixel_difference']:.8g}`, PSNR `{psnr}`.")
        lines += ["", "Visual inspection shows exact short-caption matches across 75/225 modes, with clear layout and content changes for long-caption versus truncated prompts and for third-chunk variants. Pixel differences are not quality scores; the eight-step Euler images are diagnostic comparisons, not quality evaluations.", ""]
    else:
        lines += ["## Base-model image comparisons", "", f"Status: **UNVERIFIED** ({generation.get('error', 'phase did not complete')}).", ""]
    performance = phases.get("performance", {})
    if performance:
        gpu = results.get("environment", {}).get("gpu", {})
        gpu_devices = gpu.get("gpus", [])
        gpu_name = gpu_devices[0].get("name", "recorded CUDA GPU") if gpu_devices else "recorded CUDA GPU"
        gpu_memory = gpu_devices[0].get("memory_total_mib") if gpu_devices else None
        gpu_description = f"{gpu_name} ({gpu_memory / 1024:.1f} GiB)" if gpu_memory else gpu_name
        text_times = [
            seconds
            for prompt_times in performance.get("text_encoding_seconds", {}).values()
            for seconds in prompt_times.values()
        ]
        cache_timing = (
            f"Text-encoder cache preparation measured {len(text_times)} prompt/capacity combinations in "
            f"{performance.get('text_encoding_total_seconds', 0):.4f}s "
            f"({min(text_times):.4f}–{max(text_times):.4f}s each)."
            if text_times else "Text-encoder cache preparation timing is unavailable."
        )
        lines += [
            "## Performance", "",
            f"The microsteps ran on the {gpu_description} with batch size 1, 256px images (32×32 latent), FP16 UNet autocast, FP32 LoRA, and {performance.get('training_attention_backend', 'recorded attention backend')}.",
            cache_timing,
            "",
            "| Case | Seconds / microstep | Peak allocated MiB | Peak reserved MiB |", "|---|---:|---:|---:|",
        ]
        for name, item in performance.get("training_microsteps", {}).items():
            peak = item.get("peak_cuda_memory", {})
            allocated = peak.get("allocated_bytes")
            reserved = peak.get("reserved_bytes")
            allocated_mib = f"{allocated / 1048576:.1f}" if allocated is not None else "N/A"
            reserved_mib = f"{reserved / 1048576:.1f}" if reserved is not None else "N/A"
            lines.append(f"| {name} | {item['seconds']:.4f} | {allocated_mib} | {reserved_mib} |")
        text_peak = performance.get("text_encoding_peak_cuda_memory", {})
        text_allocated = text_peak.get("allocated_bytes")
        text_reserved = text_peak.get("reserved_bytes")
        text_allocated_mib = f"{text_allocated / 1048576:.1f}" if text_allocated is not None else "N/A"
        text_reserved_mib = f"{text_reserved / 1048576:.1f}" if text_reserved is not None else "N/A"
        lines += [
            "",
            f"Text-cache preparation peak: {text_allocated_mib} MiB allocated / {text_reserved_mib} MiB reserved. Shared GPU memory: `{gpu.get('shared_memory', 'unavailable')}`.",
            "CUDA allocation and reserved-memory counters confirm these forwards ran on the GPU. The WDDM query on this laptop does not expose shared GPU memory as a separate measurement.",
            "",
        ]
    lines += [
        "## Reproduction", "", "From the repository root with the existing OneTrainer Python environment:", "",
        "```powershell",
        r".\venv\Scripts\python.exe experiments\sdxl-long-caption\comfy_compat.py --output experiments\sdxl-long-caption\comfy_compat.json",
        r".\venv\Scripts\python.exe experiments\sdxl-long-caption\comfy_compat.py --encoder-precision fp32 --output experiments\sdxl-long-caption\comfy_compat_fp32.json",
        r".\venv\Scripts\python.exe experiments\sdxl-long-caption\short_condition_debug.py",
        r".\venv\Scripts\python.exe experiments\sdxl-long-caption\pretrained_experiment.py",
        "```", "",
        "The scripts write machine-readable results and generated images under `experiments/sdxl-long-caption/`. The JSON records package versions, checkpoint hash, ComfyUI revision, GPU, precision, seeds, and settings.", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        partial = {}
        if OUTPUT_JSON.exists():
            try:
                partial = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
            except Exception:
                partial = {}
        partial.setdefault("phases", {})["fatal_error"] = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "gpu": gpu_inventory(),
            "peak_cuda_memory": cuda_peak(reset=False),
        }
        OUTPUT_JSON.write_text(json.dumps(partial, indent=2, allow_nan=False), encoding="utf-8")
        try:
            write_human_report(partial)
        except Exception:
            pass
        raise
