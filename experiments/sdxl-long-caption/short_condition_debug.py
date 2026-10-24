"""Focused pretrained check for short-caption token, embedding, UNet, and sampler parity."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from comfy_compat import DEFAULT_CHECKPOINT, build_caption, sha256_file
from pretrained_experiment import TARGET_LAYER, image_metrics, make_config, package_version


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "images"
OUT_JSON = Path(__file__).resolve().parent / "short_condition_debug.json"


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    from util.import_util import script_imports

    script_imports()
    from modules.modelSampler.StableDiffusionXLSampler import StableDiffusionXLSampler
    from modules.modelSetup.StableDiffusionXLLoRASetup import StableDiffusionXLLoRASetup
    from modules.util.enum.DataType import DataType
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.NoiseScheduler import NoiseScheduler
    from modules.util import create
    from modules.model.util.clip_chunk import sdxl_chunk_aware_unet_prediction
    from huggingface_hub import scan_cache_dir
    from transformers import CLIPTokenizer

    repo = next(item for item in scan_cache_dir().repos if item.repo_id == "stabilityai/stable-diffusion-xl-base-1.0")
    revision = next(iter(repo.revisions))
    snapshot = Path(repo.repo_path) / "snapshots" / revision.commit_hash
    tokenizers = (
        CLIPTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True),
        CLIPTokenizer.from_pretrained(snapshot / "tokenizer_2", local_files_only=True),
    )
    prompt = build_caption(tokenizers, 40)
    config = make_config(ModelType, DataType, TARGET_LAYER)
    model = __import__("comfy_compat").load_checkpoint_model(DEFAULT_CHECKPOINT)
    model.train_config = config
    setup = StableDiffusionXLLoRASetup(torch.device("cuda"), torch.device("cpu"), False)
    setup.setup_optimizations(model, config)
    torch.manual_seed(94117)
    setup.setup_model(model, config)
    setup.setup_train_device(model, config)
    model.eval()
    model.unet.train()

    torch.manual_seed(20261024)
    torch.cuda.manual_seed_all(20261024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    encodings = {}
    for limit in (75, 225):
        encodings[str(limit)] = {}
        for label, text in (("positive", prompt), ("negative", "")):
            encoded = model.encode_text(
                text=text,
                train_device=torch.device("cuda"),
                max_caption_tokens=limit,
                return_chunk_attention_mask=True,
            )
            encodings[str(limit)][label] = [value.detach().cpu() if value is not None else None for value in encoded]

    comparisons = {}
    for label in ("positive", "negative"):
        left = encodings["75"][label]
        right = encodings["225"][label]
        comparisons[label] = {
            "clip_l_hidden_equal": torch.equal(left[0], right[0][:, :77]),
            "clip_g_hidden_equal": torch.equal(left[1], right[1][:, :77]),
            "pooled_equal": torch.equal(left[2], right[2]),
            "clip_l_max_abs": float((left[0] - right[0][:, :77]).abs().max()),
            "clip_g_max_abs": float((left[1] - right[1][:, :77]).abs().max()),
            "pooled_max_abs": float((left[2] - right[2]).abs().max()),
        }

    sampler_inputs = {}
    sampler_predictions = {}
    sampler_latents = {}
    for limit in (75, 225):
        model.eval()
        model.materialize_only_text_encoders()
        positive = model.encode_text(text=prompt, train_device=torch.device("cuda"), max_caption_tokens=limit, return_chunk_attention_mask=True)
        negative = model.encode_text(text="", train_device=torch.device("cuda"), max_caption_tokens=limit, return_chunk_attention_mask=True)
        positive_hidden, positive_pooled = model.combine_text_encoder_output(*positive[:3])
        negative_hidden, negative_pooled = model.combine_text_encoder_output(*negative[:3])
        combined_hidden = torch.cat([negative_hidden, positive_hidden]).to(dtype=model.train_dtype.torch_dtype())
        combined_mask = torch.cat([negative[3], positive[3]], dim=0) if positive[3] is not None else None
        added = {
            "text_embeds": torch.cat([negative_pooled, positive_pooled], dim=0),
            "time_ids": torch.tensor([[512, 512, 0, 0, 512, 512]], device="cuda").repeat(2, 1).to(dtype=model.train_dtype.torch_dtype()),
        }
        scheduler = create.create_noise_scheduler(NoiseScheduler.EULER, model.noise_scheduler, 8)
        scheduler.set_timesteps(8, device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(42)
        latent = torch.randn((1, 4, 64, 64), generator=generator, device="cuda", dtype=model.train_dtype.torch_dtype())
        latent = latent * scheduler.init_noise_sigma
        model.materialize_only("unet")
        per_step = []
        with torch.no_grad(), model.autocast_context:
            for timestep in scheduler.timesteps:
                model_input = torch.cat([latent] * 2)
                model_input = scheduler.scale_model_input(model_input, timestep)
                noise = sdxl_chunk_aware_unet_prediction(
                    model.unet, model_input, timestep, combined_hidden, added, combined_mask
                )
                per_step.append(noise.detach().cpu())
                noise_negative, noise_positive = noise.chunk(2)
                guided = noise_negative + 7.0 * (noise_positive - noise_negative)
                latent = scheduler.step(guided, timestep, latent, return_dict=False)[0]
        sampler_predictions[str(limit)] = per_step
        sampler_latents[str(limit)] = latent.detach().cpu()
        sampler_inputs[str(limit)] = {
            "latent_initial": "seeded with torch.Generator(cuda).manual_seed(42)",
            "encoder_hidden": combined_hidden.detach().cpu(),
            "pooled": added["text_embeds"].detach().cpu(),
            "attention_mask": combined_mask.detach().cpu() if combined_mask is not None else None,
        }
    cfg_denoise = {
        "latent_initial_equal": True,
        "encoder_hidden_first_77_equal": torch.equal(
            sampler_inputs["75"]["encoder_hidden"], sampler_inputs["225"]["encoder_hidden"][:, :77]
        ),
        "pooled_equal": torch.equal(sampler_inputs["75"]["pooled"], sampler_inputs["225"]["pooled"]),
        "per_step_prediction_max_abs": [
            float((left - right).abs().max()) for left, right in zip(sampler_predictions["75"], sampler_predictions["225"], strict=True)
        ],
        "final_latent_max_abs": float((sampler_latents["75"] - sampler_latents["225"]).abs().max()),
        "final_latent_mean_abs": float((sampler_latents["75"] - sampler_latents["225"]).abs().mean()),
    }

    sampler = StableDiffusionXLSampler(torch.device("cuda"), torch.device("cpu"), model, ModelType.STABLE_DIFFUSION_XL_10_BASE)
    saved = {}
    for label, limit in (("short75_first", 75), ("short75_repeat", 75), ("short225", 225), ("short225_repeat", 225)):
        model.eval()
        image = sampler._StableDiffusionXLSampler__sample_base(
            prompt=prompt,
            negative_prompt="",
            height=512,
            width=512,
            seed=42,
            random_seed=False,
            diffusion_steps=8,
            cfg_scale=7.0,
            noise_scheduler=NoiseScheduler.EULER,
            cfg_rescale=0.0,
            sdxl_max_caption_tokens=limit,
            force_last_timestep=False,
        ).data
        path = OUT_DIR / f"debug_{label}_seed42.png"
        image.save(path)
        saved[label] = str(path)

    try:
        onetrainer_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        onetrainer_commit = "unknown"

    result = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "diffusers": package_version("diffusers"),
            "transformers": package_version("transformers"),
            "checkpoint": str(DEFAULT_CHECKPOINT),
            "checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT),
            "onetrainer_commit": onetrainer_commit,
            "gpu": torch.cuda.get_device_name(0),
        },
        "sampling_parameters": {
            "sampler": "OneTrainer StableDiffusionXLSampler / Euler",
            "steps": 8,
            "cfg": 7.0,
            "height": 512,
            "width": 512,
            "seed": 42,
            "negative_prompt": "",
            "attention": "deterministic PyTorch SDPA math backend",
        },
        "caption": prompt,
        "token_counts": [len(t(prompt, add_special_tokens=False, truncation=False).input_ids) for t in tokenizers],
        "conditioning_comparisons": comparisons,
        "cfg_denoising_controls": cfg_denoise,
        "image_files": saved,
        "image_comparisons": {
            "75_first_vs_repeat": image_metrics(saved["short75_first"], saved["short75_repeat"]),
            "225_first_vs_repeat": image_metrics(saved["short225"], saved["short225_repeat"]),
            "75_vs_225": image_metrics(saved["short75_first"], saved["short225"]),
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
