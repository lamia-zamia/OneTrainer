"""Compare OneTrainer SDXL chunking/conditioning with the local ComfyUI implementation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch


DEFAULT_CHECKPOINT = Path(
    r"D:\ComfyUI\ComfyUI\ComfyUI\models\checkpoints\illustriousXL_v01.safetensors"
)
DEFAULT_COMFY_ROOT = Path(r"D:\ComfyUI\ComfyUI\ComfyUI")
DEFAULT_COMFY_SITE = Path(r"D:\ComfyUI\ComfyUI\standalone-env\Lib\site-packages")

TAGS = [
    "1girl", "solo", "portrait", "looking_at_viewer", "smile", "blush", "long_hair",
    "brown_hair", "blue_eyes", "eyelashes", "hair_ribbon", "side_ponytail", "braid",
    "loose_hair", "windblown_hair", "school_uniform", "white_shirt", "red_necktie",
    "pleated_skirt", "black_socks", "loafers", "backpack", "hand_on_hip", "peace_sign",
    "open_mouth", "teeth", "small_mouth", "earrings", "hair_ornament", "star_hair_ornament",
    "sunlight", "blue_sky", "clouds", "city_street", "school", "window", "curtain",
    "wooden_desk", "book", "notebook", "pen", "flower", "potted_plant", "green_leaves",
    "cafe", "tea_cup", "plate", "strawberry", "cake", "table", "chair", "indoors",
    "outdoors", "sitting", "standing", "walking", "upper_body", "cowboy_shot", "full_body",
    "from_above", "from_below", "profile", "three_quarter_view", "depth_of_field",
    "soft_lighting", "rim_lighting", "warm_lighting", "evening", "sunset", "golden_hour",
    "morning", "rain", "wet_pavement", "reflection", "bokeh", "shallow_depth_of_field",
    "anime_coloring", "illustration", "detailed_background", "highly_detailed", "clean_lineart",
    "painterly", "watercolor", "cel_shading", "soft_shading", "subtle_gradient", "pastel_colors",
    "vivid_colors", "blue_cardigan", "yellow_scarf", "denim_jacket", "red_coat", "white_dress",
    "black_ribbon", "striped_socks", "hairclip", "hairband", "gloves", "umbrella", "scarf",
    "sweater", "hoodie", "sailor_collar", "pleated_dress", "lace_trim", "buttons", "sleeves_rolled_up",
    "closed_eyes", "sparkling_eyes", "looking_back", "head_tilt", "arms_behind_back", "holding_book",
    "holding_umbrella", "holding_flower", "holding_cup", "one_hand_up", "fingers_touching",
    "wind", "falling_leaves", "cherry_blossoms", "rose_petals", "autumn_leaves", "snowflakes",
    "brick_wall", "crosswalk", "traffic_light", "storefront", "train_station", "park_bench",
    "wooden_fence", "garden", "stone_path", "lanterns", "paper_lantern", "string_lights",
    "night_sky", "crescent_moon", "stars", "city_lights", "neon_sign", "rainbow", "mist",
]

TAIL_A = [
    "red_hair", "fox_ears", "black_kimono", "red_sash", "katana", "sword", "shrine_gate",
    "stone_lantern", "maple_leaves", "autumn_shrine", "sunset_glow", "embers", "wind_motions",
    "dramatic_pose", "confident_smile", "golden_eyes", "ornate_sheath", "red_tassel",
]
TAIL_B = [
    "silver_hair", "cat_ears", "maid_uniform", "white_apron", "teacup", "tea_room",
    "moonlight", "blue_curtains", "porcelain_vase", "blue_roses", "quiet_expression",
    "relaxed_pose", "violet_eyes", "lace_cuffs", "ribbon_bow", "silver_bell", "soft_candlelight",
]


def content_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False, truncation=False).input_ids


def token_counts(tokenizers, text: str) -> list[int]:
    return [len(content_ids(tokenizer, text)) for tokenizer in tokenizers]


def build_caption(tokenizers, target: int) -> str:
    selected: list[str] = []
    for tag in TAGS:
        candidate = ", ".join([*selected, tag])
        counts = token_counts(tokenizers, candidate)
        if max(counts) <= target:
            selected.append(tag)
        if min(counts) >= target:
            break

    # If the greedy pass stopped just short, try a replacement tag; punctuation and BPE lengths can
    # make the final one-token adjustment impossible by appending another complete tag.
    if token_counts(tokenizers, ", ".join(selected)) != [target, target]:
        remaining = [tag for tag in TAGS if tag not in selected]
        for index in range(len(selected) - 1, -1, -1):
            for tag in remaining:
                candidate_tags = [*selected[:index], tag, *selected[index + 1:]]
                if token_counts(tokenizers, ", ".join(candidate_tags)) == [target, target]:
                    selected = candidate_tags
                    break
            if token_counts(tokenizers, ", ".join(selected)) == [target, target]:
                break
    return ", ".join(selected)


def make_tail_caption(tokenizers, common: str, tail: list[str]) -> str:
    selected: list[str] = []
    for tag in tail:
        candidate = common + ", " + ", ".join([*selected, tag])
        selected.append(tag)
        if min(token_counts(tokenizers, candidate)) >= 225:
            break
    return common + ", " + ", ".join(selected)


def load_checkpoint_model(checkpoint: Path, encoder_precision: str = "fp16"):
    # This is the same OneTrainer loader used by production training and sampling.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from util.import_util import script_imports

    script_imports()
    from modules.util import create  # noqa: F401  (registers model loaders in production order)
    from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
    from modules.modelLoader.stableDiffusionXL.StableDiffusionXLModelLoader import StableDiffusionXLModelLoader
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.DataType import DataType
    from modules.util.enum.ModelType import ModelType

    config = TrainConfig.default_values()
    config.model_type = ModelType.STABLE_DIFFUSION_XL_10_BASE
    config.base_model_name = str(checkpoint)
    config.train_dtype = DataType.FLOAT_16
    config.fallback_train_dtype = DataType.FLOAT_16
    config.unet.weight_dtype = DataType.FLOAT_16
    config.vae.weight_dtype = DataType.FLOAT_16
    encoder_dtype = DataType.FLOAT_32 if encoder_precision == "fp32" else DataType.FLOAT_16
    config.text_encoder.weight_dtype = encoder_dtype
    config.text_encoder_2.weight_dtype = encoder_dtype

    model = StableDiffusionXLModel(config.model_type)
    StableDiffusionXLModelLoader().load(
        model,
        config.model_type,
        config.model_names(),
        config.weight_dtypes(),
        config.quantization,
    )
    model.eval()
    return model


def load_comfy_clip(checkpoint: Path, comfy_root: Path, comfy_site: Path, encoder_precision: str):
    sys.path.insert(0, str(comfy_root))
    if comfy_site.exists():
        # ComfyUI Desktop's site-packages supplies its non-Torch dependencies; Torch stays OneTrainer's CUDA build.
        sys.path.append(str(comfy_site))
    import comfy.sd

    cpu = torch.device("cpu")
    encoder_dtype = torch.float32 if encoder_precision == "fp32" else torch.float16
    model_options = {
        "load_device": cpu,
        "offload_device": cpu,
        "initial_device": cpu,
        "dtype": encoder_dtype,
    }
    _, clip, *_ = comfy.sd.load_checkpoint_guess_config(
        str(checkpoint),
        output_vae=False,
        output_clip=True,
        output_clipvision=False,
        output_model=False,
        model_options=model_options,
        disable_dynamic=True,
    )
    if clip is None:
        raise RuntimeError("ComfyUI did not return CLIP for the selected checkpoint")
    clip.cond_stage_model.to(dtype=encoder_dtype)
    return clip


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def comfy_ids(tokens, key: str) -> list[list[int]]:
    return [[int(pair[0]) for pair in section] for section in tokens[key]]


def difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    left = left.float()
    right = right.float()
    delta = left - right
    left_norm = torch.linalg.vector_norm(left).item()
    return {
        "shape_left": list(left.shape),
        "shape_right": list(right.shape),
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "l2_abs": torch.linalg.vector_norm(delta).item(),
        "l2_relative": torch.linalg.vector_norm(delta).item() / max(left_norm, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--comfy-site", type=Path, default=DEFAULT_COMFY_SITE)
    parser.add_argument("--encoder-precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("comfy_compat.json"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not (args.comfy_root / "comfy" / "sdxl_clip.py").is_file():
        raise FileNotFoundError(args.comfy_root / "comfy" / "sdxl_clip.py")

    # Load the ComfyUI reference first. It reads the checkpoint once for its CLIP-only path.
    tokenizer_cache = None
    from huggingface_hub import scan_cache_dir
    from transformers import CLIPTokenizer

    repo = next(
        item for item in scan_cache_dir().repos
        if item.repo_id == "stabilityai/stable-diffusion-xl-base-1.0"
    )
    revision = next(iter(repo.revisions))
    snapshot = Path(repo.repo_path) / "snapshots" / revision.commit_hash
    tokenizer_cache = (
        CLIPTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True),
        CLIPTokenizer.from_pretrained(snapshot / "tokenizer_2", local_files_only=True),
    )

    target_counts = (20, 74, 75, 76, 100, 149, 150, 151, 200, 225, 232)
    captions = {f"danbooru_{target}": build_caption(tokenizer_cache, target) for target in target_counts}
    common = build_caption(tokenizer_cache, 150)
    common_counts = token_counts(tokenizer_cache, common)
    if common_counts != [150, 150]:
        raise RuntimeError(f"Could not construct a shared 150-token tag prefix: {common_counts}")
    captions["third_chunk_A"] = make_tail_caption(tokenizer_cache, common, TAIL_A)
    captions["third_chunk_B"] = make_tail_caption(tokenizer_cache, common, TAIL_B)
    captions["short_40"] = build_caption(tokenizer_cache, 40)
    captions["empty_negative"] = ""

    # Get the true ComfyUI tokenizer and pretrained conditioning for every case.
    comfy_clip = load_comfy_clip(args.checkpoint, args.comfy_root, args.comfy_site, args.encoder_precision)
    comfy_encoder_dtypes = sorted({str(parameter.dtype) for parameter in comfy_clip.cond_stage_model.parameters()})
    comfy_results = {}
    for name, caption in captions.items():
        token_pairs = comfy_clip.tokenize(caption)
        section_ids = {key: comfy_ids(token_pairs, key) for key in ("l", "g")}
        with torch.inference_mode():
            encoded = comfy_clip.encode_from_tokens(token_pairs, return_pooled=True, return_dict=True)
        comfy_results[name] = {
            "caption": caption,
            "token_ids": section_ids,
            "conditioning": encoded["cond"].detach().cpu().float(),
            "pooled": encoded["pooled_output"].detach().cpu().float(),
            "conditioning_dtype": str(encoded["cond"].dtype),
            "pooled_dtype": str(encoded["pooled_output"].dtype),
        }
        print(f"ComfyUI encoded {name}: sections={len(section_ids['l'])}", flush=True)

    del comfy_clip
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    comfy_site = str(args.comfy_site)
    if comfy_site in sys.path:
        # Avoid loading Comfy Desktop's CPython 3.13 binary wheels in OneTrainer's CPython 3.12 runtime.
        sys.path.remove(comfy_site)

    model = load_checkpoint_model(args.checkpoint, args.encoder_precision)
    onetrainer_encoder_dtypes = {
        "clip_l": str(next(model.text_encoder_1.parameters()).dtype),
        "clip_g": str(next(model.text_encoder_2.parameters()).dtype),
    }
    results = {}
    from modules.model.util.clip_chunk import sdxl_chunk_attention_mask, tokenize_clip_chunks

    for name, caption in captions.items():
        tokens_l, _ = tokenize_clip_chunks(model.tokenizer_1, caption, 225)
        tokens_g, _ = tokenize_clip_chunks(model.tokenizer_2, caption, 225)
        mask = sdxl_chunk_attention_mask(
            model.tokenizer_1, tokens_l, model.tokenizer_2, tokens_g, 225
        )
        with torch.inference_mode():
            encoded = model.encode_text(
                text=caption,
                train_device=torch.device("cpu"),
                max_caption_tokens=225,
                return_chunk_attention_mask=True,
            )
            combined, pooled = model.combine_text_encoder_output(*encoded[:3])

        one_chunks_l = tokens_l.reshape(-1, 77).tolist()
        one_chunks_g = tokens_g.reshape(-1, 77).tolist()
        comfy = comfy_results[name]
        comfy_conditioning = comfy["conditioning"]
        active_token_count = int(mask.sum().item())
        compare_length = min(comfy_conditioning.shape[1], active_token_count)
        padding_hidden = combined[:, active_token_count:]
        results[name] = {
            "caption": caption,
            "token_counts": token_counts(tokenizer_cache, caption),
            "one_trainer_active_chunks": int(mask.sum().item() // 77),
            "comfy_chunk_counts": {key: len(value) for key, value in comfy["token_ids"].items()},
            "comfy_token_ids": comfy["token_ids"],
            "one_trainer_token_ids": {"l": one_chunks_l, "g": one_chunks_g},
            "token_ids_equal": {
                "l": [chunk == other for chunk, other in zip(comfy["token_ids"]["l"], one_chunks_l)],
                "g": [chunk == other for chunk, other in zip(comfy["token_ids"]["g"], one_chunks_g)],
            },
            "token_id_mismatch_positions": {
                key: [
                    sum(a != b for a, b in zip(chunk, other))
                    for chunk, other in zip(comfy["token_ids"][key], one_chunks)
                ]
                for key, one_chunks in (("l", one_chunks_l), ("g", one_chunks_g))
            },
            "conditioning_difference": difference(
                comfy_conditioning[:, :compare_length], combined[:, :compare_length]
            ),
            "pooled_difference": difference(comfy["pooled"], pooled),
            "comfy_conditioning_shape": list(comfy_conditioning.shape),
            "one_trainer_conditioning_shape": list(combined.shape),
            "one_trainer_effective_conditioning_shape": [combined.shape[0], active_token_count, combined.shape[-1]],
            "one_trainer_mask_shape": list(mask.shape),
            "one_trainer_padding_tokens": int(mask.shape[1] - active_token_count),
            "one_trainer_padding_hidden_max_abs": (
                padding_hidden.abs().max().item() if padding_hidden.numel() else 0.0
            ),
            "one_trainer_conditioning_dtype": str(combined.dtype),
            "one_trainer_pooled_dtype": str(pooled.dtype),
            "comfy_conditioning_dtype": comfy["conditioning_dtype"],
            "comfy_pooled_dtype": comfy["pooled_dtype"],
        }
        if name == "short_40":
            with torch.inference_mode():
                legacy = model.encode_text(
                    text=caption,
                    train_device=torch.device("cpu"),
                    max_caption_tokens=75,
                )
                legacy_combined, legacy_pooled = model.combine_text_encoder_output(*legacy)
            results[name]["75_vs_225_hidden_difference"] = difference(
                legacy_combined, combined[:, :77]
            )
            results[name]["75_vs_225_pooled_difference"] = difference(legacy_pooled, pooled)
        print(
            f"Compared {name}: counts={results[name]['token_counts']} "
            f"active={results[name]['one_trainer_active_chunks']} "
            f"cond_rel={results[name]['conditioning_difference']['l2_relative']:.6g}",
            flush=True,
        )

    third_a = results["third_chunk_A"]
    third_b = results["third_chunk_B"]
    for encoder in ("l", "g"):
        third_a["first_two_chunks_identical_" + encoder] = all(
            third_a["token_ids_equal"][encoder][:2]
        ) and all(third_b["token_ids_equal"][encoder][:2]) and all(
            comfy_results["third_chunk_A"]["token_ids"][encoder][i]
            == comfy_results["third_chunk_B"]["token_ids"][encoder][i]
            for i in range(min(2, len(comfy_results["third_chunk_A"]["token_ids"][encoder])))
        )

    try:
        comfy_commit = subprocess.check_output(
            ["git", "-C", str(args.comfy_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        comfy_commit = "unknown"
    try:
        onetrainer_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        onetrainer_commit = "unknown"

    payload = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "transformers": package_version("transformers"),
            "diffusers": package_version("diffusers"),
            "tokenizers": package_version("tokenizers"),
            "safetensors": package_version("safetensors"),
            "comfy_aimdo": package_version("comfy-aimdo"),
            "comfy_kitchen": package_version("comfy-kitchen"),
            "comfy_commit": comfy_commit,
            "onetrainer_commit": onetrainer_commit,
            "precision": f"CLIP parameters and OneTrainer encoders normalized to {args.encoder_precision}; checkpoint weights from the same file",
            "comfy_encoder_parameter_dtypes": comfy_encoder_dtypes,
            "onetrainer_encoder_parameter_dtypes": onetrainer_encoder_dtypes,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
        },
        "comfyui_source": str(args.comfy_root),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
