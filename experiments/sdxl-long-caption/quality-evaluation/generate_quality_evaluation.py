from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(EXPERIMENT_DIR))

from comfy_compat import DEFAULT_CHECKPOINT, load_checkpoint_model  # noqa: E402


RUNS = {
    "after_75": Path(r"D:\ComfyUI\ComfyUI\ComfyUI\models\loras\lora_manager\alt_test_after_75"),
    "after_225": Path(r"D:\ComfyUI\ComfyUI\ComfyUI\models\loras\lora_manager\alt_test_after_225"),
}
BASE_SHA256 = "3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9"
SEEDS = (42, 123, 456, 789)
STRENGTHS = (1.0, 1.5, 2.0)
PROGRESSION_STEPS = (1199, 2399, 3199, 3999)
SAMPLE_STEPS = 28
CFG = 3.0
WIDTH = 1024
HEIGHT = 1024
MAX_CAPTION_TOKENS = 225
PAIR_RANDOM_SEED = 20261027


def load_prompts() -> dict:
    return json.loads((HERE / "prompt_suite.json").read_text(encoding="utf-8"))


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(run: str, step: int) -> Path:
    exact = RUNS[run] / f"alt_test_{run}-{step:05}.safetensors"
    if exact.is_file():
        return exact
    matches = sorted(RUNS[run].glob(f"alt_test_{run}-{step:05}-*.safetensors"))
    if not matches:
        raise FileNotFoundError(exact)
    return matches[0]


def content_counts(tokenizers, text: str) -> list[int]:
    return [
        len(tokenizer(text, add_special_tokens=False, truncation=False).input_ids)
        for tokenizer in tokenizers
    ]


def marker_positions(tokenizers, text: str, markers: list[str]) -> list[dict]:
    records = []
    for marker in markers:
        char_start = text.find(marker)
        if char_start < 0:
            raise ValueError(f"Late prompt marker not found: {marker}")
        start_positions = []
        end_positions = []
        for tokenizer in tokenizers:
            encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
            positions = [
                index + 1
                for index, (start, end) in enumerate(encoded.offset_mapping)
                if end > start and start < char_start + len(marker) and end > char_start
            ]
            start_positions.append(positions[0] if positions else None)
            end_positions.append(positions[-1] if positions else None)
        records.append({
            "marker": marker,
            "token_start_positions": start_positions,
            "token_end_positions": end_positions,
        })
    return records


def stage_plan(stage: str, prompts: dict) -> list[dict]:
    items = []
    if stage == "pilot":
        prompt_ids, steps, strengths, seeds = ["P6"], [3999], [1.0], [42]
    elif stage == "screen":
        prompt_ids = [p["id"] for p in prompts["prompts"] if p.get("screen")]
        steps, strengths, seeds = [3999], list(STRENGTHS), list(SEEDS)
    else:
        prompt_ids, steps, strengths, seeds = ["P6", "P7"], list(PROGRESSION_STEPS), list(STRENGTHS), list(SEEDS)
    for run in RUNS:
        for step in steps:
            for strength in strengths:
                for prompt_id in prompt_ids:
                    for seed in seeds:
                        items.append({
                            "run": run,
                            "step": step,
                            "strength": strength,
                            "prompt_id": prompt_id,
                            "seed": seed,
                        })
    return items


def plan_key(item: dict) -> tuple:
    return (
        item["run"], int(item["step"]), float(item["strength"]),
        item["prompt_id"], int(item["seed"]),
    )


def record_key(record: dict) -> tuple:
    return (
        record["run"], int(record["checkpoint_step"]), float(record["strength"]),
        record["prompt_id"], int(record["seed"]),
    )


def load_resume_records(output_dir: Path, stage: str, planned_items: list[dict], base_sha256: str) -> list[dict]:
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Cannot resume without metadata: {metadata_path}")
    planned_keys = {plan_key(item) for item in planned_items}
    prompts = load_prompts()
    if stage == "screen":
        base_keys = {
            ("base", 3999, 0.0, prompt["id"], seed)
            for prompt in prompts["prompts"] if prompt.get("base_control")
            for seed in SEEDS
        }
    else:
        base_keys = set()
    allowed_keys = planned_keys | base_keys
    records = []
    seen = set()
    for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = record_key(record)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid metadata record at {metadata_path}:{line_number}: {exc}") from exc
        if record.get("stage") != stage or key not in allowed_keys:
            raise ValueError(f"Unexpected stage or image assignment at {metadata_path}:{line_number}: {key}")
        if key in seen:
            raise ValueError(f"Duplicate completed image assignment at {metadata_path}:{line_number}: {key}")
        seen.add(key)
        if record.get("base_checkpoint_sha256") != base_sha256:
            raise ValueError(f"Base checkpoint hash mismatch in {metadata_path}:{line_number}")
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Recorded image is missing at {metadata_path}:{line_number}: {image_path}")
        if record["run"] != "base":
            checkpoint = checkpoint_path(record["run"], record["checkpoint_step"])
            stat = checkpoint.stat()
            if (
                str(checkpoint) != record.get("checkpoint_path")
                or stat.st_size != record.get("checkpoint_size_bytes")
                or stat.st_mtime_ns != record.get("checkpoint_mtime_ns")
            ):
                raise ValueError(f"Checkpoint changed since this image was generated: {checkpoint}")
        records.append(record)
    return records


def build_config():
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.DataType import DataType
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    config = TrainConfig.default_values()
    config.model_type = ModelType.STABLE_DIFFUSION_XL_10_BASE
    config.training_method = TrainingMethod.LORA
    config.train_device = "cuda"
    config.temp_device = "cpu"
    config.train_dtype = DataType.FLOAT_16
    config.fallback_train_dtype = DataType.BFLOAT_16
    config.enable_autocast_cache = True
    config.unet.weight_dtype = DataType.FLOAT_16
    # OneTrainer's setup_model creates its optimizer even for inference. Keep
    # the run's trained adapter groups enabled so the optimizer has parameters;
    # the model is switched to eval mode before sampling.
    config.unet.train = True
    config.unet.gradient_checkpointing = False
    config.text_encoder.weight_dtype = DataType.FLOAT_16
    config.text_encoder.train = True
    config.text_encoder.gradient_checkpointing = False
    config.text_encoder_2.weight_dtype = DataType.FLOAT_16
    config.text_encoder_2.train = False
    config.text_encoder_2.gradient_checkpointing = False
    config.vae.weight_dtype = DataType.FLOAT_16
    config.vae.train = False
    config.layer_filter = "attentions"
    config.layer_filter_regex = False
    config.lora_rank = 32
    config.lora_alpha = 32.0
    config.lora_weight_dtype = DataType.FLOAT_32
    config.dropout_probability = 0.0
    config.sdxl_max_caption_tokens = MAX_CAPTION_TOKENS
    return config


def adapter_modules(model):
    return [
        module for module in (
            model.text_encoder_1_lora,
            model.text_encoder_2_lora,
            model.unet_lora,
        )
        if module is not None
    ]


def up_parameters(model):
    parameters = []
    for wrapper in adapter_modules(model):
        for module in wrapper.lora_modules.values():
            lora_up = getattr(module, "lora_up", None)
            if lora_up is not None:
                parameters.append(lora_up.weight)
    return parameters


@contextmanager
def scaled_adapter(model, strength: float):
    params = up_parameters(model)
    originals = [(parameter, parameter.detach().to("cpu").clone()) for parameter in params]
    with torch.no_grad():
        for parameter, original in originals:
            parameter.copy_(original.to(device=parameter.device, dtype=parameter.dtype) * strength)
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, original in originals:
                parameter.copy_(original.to(device=parameter.device, dtype=parameter.dtype))


def apply_checkpoint(model, loader, model_names, setup, config, path: Path, first: bool) -> int:
    loader.load(model, model_names(lora=str(path)))
    state = model.lora_state_dict
    if not state:
        raise RuntimeError(f"No LoRA tensors loaded from {path}")
    tensor_count = len(state)
    if first:
        setup.setup_model(model, config)
        setup.setup_train_device(model, config)
        model.eval()
        return tensor_count

    for module in adapter_modules(model):
        # LoRAModuleWrapper is a project wrapper, not nn.Module; its strict
        # per-layer loading raises on incompatibilities and returns None.
        module.load_state_dict(state)
    model.lora_state_dict = None
    return tensor_count


def save_pair_image(image: Image.Image, output_dir: Path, image_id: str) -> str:
    path = output_dir / "images" / f"{image_id}.png"
    image.convert("RGB").save(path, format="PNG", optimize=True)
    return str(path)


def sample_one(sampler, model, prompt: str, negative: str, seed: int, strength: float) -> tuple[Image.Image, dict]:
    from modules.util.enum.NoiseScheduler import NoiseScheduler

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 1024px evaluation")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with scaled_adapter(model, strength):
        result = sampler._StableDiffusionXLSampler__sample_base(
            prompt=prompt,
            negative_prompt=negative,
            height=HEIGHT,
            width=WIDTH,
            seed=seed,
            random_seed=False,
            diffusion_steps=SAMPLE_STEPS,
            cfg_scale=CFG,
            noise_scheduler=NoiseScheduler.EULER_A,
            cfg_rescale=0.0,
            sdxl_max_caption_tokens=MAX_CAPTION_TOKENS,
            force_last_timestep=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return result.data, {
        "generation_seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def append_metadata(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_blind_assets(output_dir: Path, records: list[dict], prompt_map: dict) -> None:
    image_by_id = {record["image_id"]: record["image_path"] for record in records}
    comparisons = {}
    for record in records:
        if record["run"] not in RUNS:
            continue
        key = (record["prompt_id"], record["checkpoint_step"], record["strength"], record["seed"])
        comparisons.setdefault(key, {})[record["run"]] = record

    rng = random.Random(PAIR_RANDOM_SEED)
    key_rows = []
    score_rows = []
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(exist_ok=True)
    grouped = {}
    for key, treatment_records in sorted(comparisons.items()):
        if set(treatment_records) != set(RUNS):
            continue
        prompt_id, step, strength, seed = key
        orientations = [("A", treatment_records["after_75"]), ("B", treatment_records["after_225"])]
        rng.shuffle(orientations)
        pair_id = f"pair_{len(key_rows) + 1:04}"
        assignment = {label: record for label, record in orientations}
        key_rows.append({
            "pair_id": pair_id,
            "prompt_id": prompt_id,
            "checkpoint_step": step,
            "strength": strength,
            "seed": seed,
            "A_image_id": assignment["A"]["image_id"],
            "A_condition": assignment["A"]["run"],
            "B_image_id": assignment["B"]["image_id"],
            "B_condition": assignment["B"]["run"],
        })
        prompt = prompt_map[prompt_id]
        score_rows.append({
            "pair_id": pair_id,
            "prompt_id": prompt_id,
            "prompt": prompt["text"],
            "seed": seed,
            "checkpoint_step": step,
            "strength": strength,
            "A_image_id": assignment["A"]["image_id"],
            "B_image_id": assignment["B"]["image_id"],
            "A_prompt_compliance_1to5": "",
            "B_prompt_compliance_1to5": "",
            "A_style_fidelity_1to5": "",
            "B_style_fidelity_1to5": "",
            "A_anatomy_artifact_0to3": "",
            "B_anatomy_artifact_0to3": "",
            "A_unwanted_prior_0to3": "",
            "B_unwanted_prior_0to3": "",
            "pair_style_preference_A_B_tie": "",
            "notes": "",
        })
        grouped.setdefault((prompt_id, step, strength), []).append((pair_id, assignment))

    font = ImageFont.load_default()
    for (prompt_id, step, strength), pairs in grouped.items():
        cell = 384
        label_h = 30
        margin = 14
        width = margin * 3 + cell * 2
        height = margin + len(pairs) * (cell + label_h + margin)
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        y = margin
        for pair_id, assignment in pairs:
            for column, label in enumerate(("A", "B")):
                record = assignment[label]
                x = margin + column * (cell + margin)
                with Image.open(image_by_id[record["image_id"]]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((cell, cell))
                    sheet.paste(image, (x + (cell - image.width) // 2, y + label_h))
                draw.text((x, y), f"{pair_id}  {label}  {record['image_id']}", fill="black", font=font)
            y += cell + label_h + margin
        sheet.save(sheets_dir / f"{prompt_id}_step{step}_strength{strength:g}.png", optimize=True)

    if score_rows and not (output_dir / "human_scoring.csv").exists():
        with (output_dir / "human_scoring.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(score_rows[0]))
            writer.writeheader()
            writer.writerows(score_rows)
    (output_dir / "blind_key.json").write_text(json.dumps(key_rows, indent=2) + "\n", encoding="utf-8")

    extended_rows = []
    extended_keys = []
    for prompt in prompt_map.values():
        if not prompt.get("base_control"):
            continue
        prompt_id = prompt["id"]
        seeds = sorted({r["seed"] for r in records if r["prompt_id"] == prompt_id and r["run"] in RUNS})
        for seed in seeds:
            candidates = [
                record for record in records
                if record["prompt_id"] == prompt_id and record["seed"] == seed
                and record["checkpoint_step"] == 3999
                and (record["run"] == "base" or record["strength"] == 1.5)
                and record["run"] in {"base", "after_75", "after_225"}
            ]
            by_run = {record["run"]: record for record in candidates}
            if set(by_run) != {"base", "after_75", "after_225"}:
                continue
            columns = [("A", by_run["base"]), ("B", by_run["after_75"]), ("C", by_run["after_225"])]
            rng.shuffle(columns)
            triple_id = f"long_{len(extended_keys) + 1:03}"
            assignment = {label: record for label, record in columns}
            extended_keys.append({
                "comparison_id": triple_id,
                "prompt_id": prompt_id,
                "seed": seed,
                "A_image_id": assignment["A"]["image_id"],
                "A_condition": assignment["A"]["run"],
                "B_image_id": assignment["B"]["image_id"],
                "B_condition": assignment["B"]["run"],
                "C_image_id": assignment["C"]["image_id"],
                "C_condition": assignment["C"]["run"],
            })
            extended_rows.append({
                "comparison_id": triple_id,
                "prompt_id": prompt_id,
                "prompt": prompt["text"],
                "seed": seed,
                "A_image_id": assignment["A"]["image_id"],
                "B_image_id": assignment["B"]["image_id"],
                "C_image_id": assignment["C"]["image_id"],
                "A_late_detail_compliance_1to5": "",
                "B_late_detail_compliance_1to5": "",
                "C_late_detail_compliance_1to5": "",
                "notes": "",
            })
            images = [image_by_id[assignment[label]["image_id"]] for label in ("A", "B", "C")]
            cell = 384
            margin = 14
            label_h = 30
            sheet = Image.new("RGB", (margin * 4 + cell * 3, cell + label_h + margin * 2), "white")
            draw = ImageDraw.Draw(sheet)
            for column, label in enumerate(("A", "B", "C")):
                x = margin + column * (cell + margin)
                with Image.open(images[column]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((cell, cell))
                    sheet.paste(image, (x + (cell - image.width) // 2, margin + label_h))
                draw.text((x, margin), f"{triple_id}  {label}", fill="black", font=font)
            sheet.save(sheets_dir / f"{triple_id}_{prompt_id}.png", optimize=True)

    if extended_rows and not (output_dir / "human_extended_scoring.csv").exists():
        with (output_dir / "human_extended_scoring.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(extended_rows[0]))
            writer.writeheader()
            writer.writerows(extended_rows)
    (output_dir / "extended_blind_key.json").write_text(json.dumps(extended_keys, indent=2) + "\n", encoding="utf-8")


def environment_info(base_sha256: str) -> dict:
    packages = {}
    for package in ("torch", "transformers", "diffusers", "safetensors", "tokenizers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "packages": packages,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "base_checkpoint": str(DEFAULT_CHECKPOINT),
        "base_checkpoint_sha256": base_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "screen", "progression"), default="pilot")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted stage from its metadata.jsonl")
    args = parser.parse_args()
    prompts = load_prompts()
    prompt_map = {prompt["id"]: prompt for prompt in prompts["prompts"]}
    output_dir = args.output_dir or HERE / "results" / (
        args.stage + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if args.resume and (not output_dir.exists() or not any(output_dir.iterdir())):
        raise FileNotFoundError(f"Cannot resume an empty or missing output directory: {output_dir}")
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "contact_sheets").mkdir(exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    if not DEFAULT_CHECKPOINT.is_file():
        raise FileNotFoundError(DEFAULT_CHECKPOINT)
    base_sha256 = checkpoint_sha256(DEFAULT_CHECKPOINT)
    if base_sha256 != BASE_SHA256:
        raise RuntimeError(
            "The current base checkpoint hash differs from the checkpoint used in the prior training audit: "
            f"{base_sha256} != {BASE_SHA256}"
        )
    planned_items = stage_plan(args.stage, prompts)
    existing_records = load_resume_records(output_dir, args.stage, planned_items, base_sha256) if args.resume else []
    completed_keys = {record_key(record) for record in existing_records}
    items = [item for item in planned_items if plan_key(item) not in completed_keys]
    lora_files = {
        (item["run"], item["step"]): checkpoint_path(item["run"], item["step"])
        for item in planned_items
    }

    print(
        f"Stage {args.stage}: {len(items)} images remain of {len(planned_items)} paired-LoRA images "
        f"at {WIDTH}px"
    )
    model = load_checkpoint_model(DEFAULT_CHECKPOINT)
    config = build_config()
    model.train_config = config
    from modules.modelLoader.stableDiffusionXL.StableDiffusionXLLoRALoader import StableDiffusionXLLoRALoader
    from modules.modelSetup.StableDiffusionXLLoRASetup import StableDiffusionXLLoRASetup
    from modules.modelSampler.StableDiffusionXLSampler import StableDiffusionXLSampler
    from modules.util.ModelNames import ModelNames
    from modules.util.TrainProgress import TrainProgress
    from modules.util.enum.ModelType import ModelType
    import modules.util.create  # noqa: F401

    model.train_progress = TrainProgress()
    setup = StableDiffusionXLLoRASetup(torch.device("cuda"), torch.device("cpu"), False)
    setup.setup_optimizations(model, config)
    loader = StableDiffusionXLLoRALoader()
    first_item = items[0] if items else planned_items[0]
    first_path = lora_files[(first_item["run"], first_item["step"])]
    loaded_tensor_count = apply_checkpoint(model, loader, ModelNames, setup, config, first_path, first=True)
    sampler = StableDiffusionXLSampler(
        train_device=torch.device("cuda"),
        temp_device=torch.device("cpu"),
        model=model,
        model_type=ModelType.STABLE_DIFFUSION_XL_10_BASE,
    )
    tokenizer_pair = (model.tokenizer_1, model.tokenizer_2)
    for tokenizer in tokenizer_pair:
        tokenizer.model_max_length = 10**9
    tokenizer_lengths = {}
    late_positions = {}
    for prompt_id in {item["prompt_id"] for item in planned_items}:
        prompt = prompt_map[prompt_id]
        positive = prompts["style_prefix"] + ", " + prompt["text"]
        tokenizer_lengths[prompt_id] = content_counts(tokenizer_pair, positive)
        if prompt.get("late_markers"):
            positions = marker_positions(tokenizer_pair, positive, prompt["late_markers"])
            late_positions[prompt_id] = positions
            target = 75 if prompt_id == "late_after_75" else 150
            if any(
                any(position is None or position <= target for position in row["token_start_positions"])
                or any(position is None or position > MAX_CAPTION_TOKENS for position in row["token_end_positions"])
                for row in positions
            ):
                raise RuntimeError(
                    f"Late marker for {prompt_id} must start after position {target} and end by "
                    f"{MAX_CAPTION_TOKENS}: {positions}"
                )

    records = list(existing_records)
    metadata_path = output_dir / "metadata.jsonl"
    loaded_path = first_path
    for run in RUNS:
        for item in [case for case in items if case["run"] == run]:
            path = lora_files[(run, item["step"])]
            if path != loaded_path:
                loaded_tensor_count = apply_checkpoint(model, loader, ModelNames, setup, config, path, first=False)
                loaded_path = path
            prompt = prompt_map[item["prompt_id"]]
            positive = prompts["style_prefix"] + ", " + prompt["text"]
            image, timing = sample_one(
                sampler, model, positive, prompts["negative_prompt"], item["seed"], item["strength"]
            )
            image_id = "img_" + uuid.uuid4().hex[:10]
            image_path = save_pair_image(image, output_dir, image_id)
            record = {
                "image_id": image_id,
                "image_path": image_path,
                "stage": args.stage,
                "run": run,
                "training_caption_dataset": "original" if run == "after_75" else "extended",
                "effective_training_token_limit": 75 if run == "after_75" else 225,
                "checkpoint_step": item["step"],
                "checkpoint_path": str(path),
                "checkpoint_size_bytes": path.stat().st_size,
                "checkpoint_mtime_ns": path.stat().st_mtime_ns,
                "loaded_lora_tensor_count": loaded_tensor_count,
                "base_checkpoint": str(DEFAULT_CHECKPOINT),
                "base_checkpoint_sha256": base_sha256,
                "prompt_id": prompt["id"],
                "prompt_group": prompt["group"],
                "prompt_text": prompt["text"],
                "positive_prompt": positive,
                "negative_prompt": prompts["negative_prompt"],
                "clip_l_content_tokens": tokenizer_lengths[prompt["id"]][0],
                "clip_g_content_tokens": tokenizer_lengths[prompt["id"]][1],
                "late_detail_token_positions": late_positions.get(prompt["id"], []),
                "inference_token_limit": MAX_CAPTION_TOKENS,
                "strength": item["strength"],
                "lora_strength": item["strength"],
                "seed": item["seed"],
                "sampler": "OneTrainer StableDiffusionXLSampler / Euler ancestral",
                "scheduler": "Euler ancestral",
                "steps": SAMPLE_STEPS,
                "cfg": CFG,
                "width": WIDTH,
                "height": HEIGHT,
                **timing,
            }
            records.append(record)
            append_metadata(metadata_path, record)
            print(f"{image_id}: {run}, step {item['step']}, strength {item['strength']}, {prompt['id']}, seed {item['seed']}")

    if args.stage == "screen":
        prompt_ids = {prompt["id"] for prompt in prompts["prompts"] if prompt.get("base_control")}
        for prompt_id in sorted(prompt_ids):
            prompt = prompt_map[prompt_id]
            positive = prompts["style_prefix"] + ", " + prompt["text"]
            for seed in SEEDS:
                if ("base", 3999, 0.0, prompt_id, seed) in completed_keys:
                    continue
                image, timing = sample_one(
                    sampler, model, positive, prompts["negative_prompt"], seed, 0.0
                )
                image_id = "img_" + uuid.uuid4().hex[:10]
                image_path = save_pair_image(image, output_dir, image_id)
                record = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "stage": args.stage,
                    "run": "base",
                    "training_caption_dataset": None,
                    "effective_training_token_limit": None,
                    "checkpoint_step": 3999,
                    "checkpoint_path": None,
                    "loaded_lora_tensor_count": 0,
                    "base_checkpoint": str(DEFAULT_CHECKPOINT),
                    "base_checkpoint_sha256": base_sha256,
                    "prompt_id": prompt_id,
                    "prompt_group": prompt["group"],
                    "prompt_text": prompt["text"],
                    "positive_prompt": positive,
                    "negative_prompt": prompts["negative_prompt"],
                    "clip_l_content_tokens": content_counts(tokenizer_pair, positive)[0],
                    "clip_g_content_tokens": content_counts(tokenizer_pair, positive)[1],
                    "late_detail_token_positions": late_positions.get(prompt_id, []),
                    "inference_token_limit": MAX_CAPTION_TOKENS,
                    "strength": 0.0,
                    "lora_strength": 0.0,
                    "seed": seed,
                    "sampler": "OneTrainer StableDiffusionXLSampler / Euler ancestral",
                    "scheduler": "Euler ancestral",
                    "steps": SAMPLE_STEPS,
                    "cfg": CFG,
                    "width": WIDTH,
                    "height": HEIGHT,
                    **timing,
                }
                records.append(record)
                append_metadata(metadata_path, record)

    generate_blind_assets(output_dir, records, prompt_map)
    (output_dir / "environment.json").write_text(json.dumps(environment_info(base_sha256), indent=2) + "\n", encoding="utf-8")
    summary = {
        "stage": args.stage,
        "planned_paired_images": len(planned_items),
        "images_generated": len(records),
        "base_controls": sum(record["run"] == "base" for record in records),
        "seeds": list(SEEDS if args.stage != "pilot" else [42]),
        "strengths": list(STRENGTHS if args.stage in {"screen", "progression"} else [1.0]),
        "checkpoint_steps": [3999] if args.stage in {"pilot", "screen"} else list(PROGRESSION_STEPS),
        "prompts": sorted({item["prompt_id"] for item in planned_items}),
        "prompt_token_counts": tokenizer_lengths,
        "late_detail_token_positions": late_positions,
        "blinded_pair_count": sum(1 for _ in json.loads((output_dir / "blind_key.json").read_text(encoding="utf-8"))),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved results in " + str(output_dir))


if __name__ == "__main__":
    main()
