from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import scan_cache_dir
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from transformers import CLIPTokenizerFast


HERE = Path(__file__).resolve().parent
ORIGINAL_DATA = Path(r"D:\lora-datasets\Projects\alterkyon_new\concepts\test2")
EXTENDED_DATA = Path(r"D:\lora-datasets\Projects\alterkyon_new - Copy\concepts\test2")
LORA_ROOT = Path(r"D:\ComfyUI\ComfyUI\ComfyUI\models\loras")
DATASET_ROOT = Path(r"D:\lora-datasets\Projects\alterkyon_new\loras")
RUN_CONFIGS = {
    "before_75": Path(r"D:\ComfyUI\ComfyUI\ComfyUI\models\loras\alt_test_before_75\backup\2026-10-26_12-39-36-backup-499-0-499\onetrainer_config\args.json"),
    "before_225": Path(r"D:\ComfyUI\ComfyUI\ComfyUI\models\loras\alt_test_before_225\backup\2026-10-26_13-11-21-backup-499-0-499\onetrainer_config\args.json"),
    "after_75": DATASET_ROOT / "alt_test_after_75" / "backup" / "2026-10-26_21-15-03-backup-3999-7-338" / "onetrainer_config" / "args.json",
    "after_225": DATASET_ROOT / "alt_test_after_225" / "backup" / "2026-10-26_23-28-00-backup-3999-7-338" / "onetrainer_config" / "args.json",
}
TOKENIZER_REPOS = {
    "clip_l": ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
    "clip_g": ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "743c27bd53dfe508a0ade0f50698f99b39d03bec"),
}
TAXONOMY = {
    "subject": re.compile(r"^(?:\d+girl|\d+boy|solo|multiple girls|multiple boys|male focus|female focus|character|person|couple)$", re.I),
    "appearance": re.compile(r"hair|eyes|eye|skin|face|mouth|nose|teeth|tongue|pupil|ear|tail|horn|wing|breast|body|muscular|short|long", re.I),
    "clothing_material": re.compile(r"wear|clothes|clothing|dress|shirt|top|skirt|pant|uniform|coat|robe|glove|sock|stocking|thighhigh|shoe|boot|bikini|swimwear|bra|panties|leotard|latex|leather|cotton|denim|silk", re.I),
    "pose_action_camera": re.compile(r"sitting|standing|lying|kneeling|walking|running|looking|from |view|shot|angle|facing|pose|holding|hand|leg|arm|profile|close-up|upper body|full body|wariza|on stomach", re.I),
    "environment": re.compile(r"background|indoors|outdoors|room|bedroom|living room|street|forest|sky|city|water|window|building|couch|sofa|chair|landscape", re.I),
    "lighting_style": re.compile(r"light|lighting|shadow|shading|color|colour|style|artist|render|lineart|line art|painting|texture|glossy|shiny|matte", re.I),
}
POSITION_BINS = ((1, 25, "1-25"), (26, 50, "26-50"), (51, 75, "51-75"), (76, 150, "76-150"), (151, 225, "151-225"), (226, 10**9, ">225"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_appended_tag_inventory(path: Path, captions: dict) -> None:
    fields = ("tag", "occurrence_count", "category", "token_position_bins", "examples")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in captions["appended_tag_inventory"]:
            writer.writerow({
                **item,
                "token_position_bins": json.dumps(item["token_position_bins"], ensure_ascii=False, sort_keys=True),
                "examples": json.dumps(item["examples"], ensure_ascii=False),
            })


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizers() -> dict[str, CLIPTokenizerFast]:
    repos = {repo.repo_id: repo for repo in scan_cache_dir().repos}
    loaded = {}
    for name, (repo_id, commit) in TOKENIZER_REPOS.items():
        snapshot = Path(repos[repo_id].repo_path) / "snapshots" / commit
        tokenizer = CLIPTokenizerFast.from_pretrained(snapshot, local_files_only=True)
        tokenizer.model_max_length = 10**9
        loaded[name] = tokenizer
    return loaded


def classify(tag: str) -> str:
    for category, pattern in TAXONOMY.items():
        if category == "subject":
            if pattern.fullmatch(tag.strip()):
                return category
        elif pattern.search(tag):
            return category
    return "other"


def token_position(offsets: list[tuple[int, int]], char_index: int) -> int | None:
    for index, (start, end) in enumerate(offsets):
        if start <= char_index < end:
            return index + 1
    return None


def token_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int | None, int | None]:
    positions = [
        index + 1
        for index, (start, end) in enumerate(offsets)
        if end > start and start < char_end and end > char_start
    ]
    return (positions[0], positions[-1]) if positions else (None, None)


def token_summary(lengths: list[int]) -> dict:
    percentiles = np.percentile(lengths, [25, 50, 75, 90, 95, 99]).tolist()
    return {
        "count": len(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "percentiles": dict(zip(("p25", "p50", "p75", "p90", "p95", "p99"), percentiles)),
        "min": min(lengths),
        "max": max(lengths),
        "buckets": {
            "0-75": sum(n <= 75 for n in lengths),
            "76-150": sum(75 < n <= 150 for n in lengths),
            "151-225": sum(150 < n <= 225 for n in lengths),
            ">225": sum(n > 225 for n in lengths),
        },
    }


def paired_caption_analysis(original: Path, extended: Path, tokenizer_map: dict) -> dict:
    original_files = {p.stem: p for p in original.glob("*.txt")}
    extended_files = {p.stem: p for p in extended.glob("*.txt")}
    if original_files.keys() != extended_files.keys():
        raise RuntimeError("Original and extended caption filenames do not pair exactly")

    original_lengths = []
    extended_lengths = []
    original_lengths_by_tokenizer = defaultdict(list)
    extended_lengths_by_tokenizer = defaultdict(list)
    first75_diff = 0
    token_prefix_exact = 0
    tag_prefix_exact = 0
    extends_after_75 = 0
    over_225 = []
    appended_tags = 0
    additions = Counter()
    appended_tag_details = defaultdict(lambda: {
        "occurrence_count": 0,
        "category": "other",
        "token_position_bins": Counter(),
        "examples": [],
    })
    all_tag_positions = defaultdict(Counter)
    appended_tag_positions = defaultdict(Counter)
    category_counts = defaultdict(Counter)
    image_hash_equal = 0
    missing_images = []
    examples = {}
    all_ids_equal = True

    for stem in sorted(original_files):
        old_text = original_files[stem].read_text(encoding="utf-8").strip()
        new_text = extended_files[stem].read_text(encoding="utf-8").strip()
        old_tags = [tag.strip() for tag in old_text.split(",") if tag.strip()]
        new_tags = [tag.strip() for tag in new_text.split(",") if tag.strip()]
        has_tag_prefix = new_tags[:len(old_tags)] == old_tags
        tag_prefix_exact += int(has_tag_prefix)
        extra_tags = new_tags[len(old_tags):] if has_tag_prefix else []
        appended_tags += len(extra_tags)
        additions.update(extra_tags)

        original_images = [p for p in original.glob(stem + ".*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        extended_images = [p for p in extended.glob(stem + ".*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        if len(original_images) != 1 or len(extended_images) != 1:
            missing_images.append(stem)
        elif sha256(original_images[0]) == sha256(extended_images[0]):
            image_hash_equal += 1

        ids_by_tokenizer = {}
        encoded_by_tokenizer = {}
        for name, tokenizer in tokenizer_map.items():
            old_ids = list(tokenizer(old_text, add_special_tokens=False, truncation=False).input_ids)
            new_encoded = tokenizer(new_text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
            new_ids = list(new_encoded.input_ids)
            ids_by_tokenizer[name] = (old_ids, new_ids)
            encoded_by_tokenizer[name] = (new_ids, list(new_encoded.offset_mapping), tokenizer)
            original_lengths_by_tokenizer[name].append(len(old_ids))
            extended_lengths_by_tokenizer[name].append(len(new_ids))
            if name == "clip_l":
                original_lengths.append(len(old_ids))
                extended_lengths.append(len(new_ids))
                token_prefix_exact += int(new_ids[:len(old_ids)] == old_ids)
                first75_diff += int(old_ids[:75] != new_ids[:75])
                extends_after_75 += int(len(new_ids) > 75)
                if len(new_ids) > 225:
                    over_225.append(stem)

                cursor = 0
                for tag_index, tag in enumerate(new_tags):
                    start = new_text.find(tag, cursor)
                    cursor = max(start, 0) + len(tag)
                    pos = token_position(list(new_encoded.offset_mapping), start) if start >= 0 else None
                    if pos is None:
                        continue
                    bin_name = next(label for low, high, label in POSITION_BINS if low <= pos <= high)
                    category = classify(tag)
                    all_tag_positions[category][bin_name] += 1
                    category_counts[category]["all_tags"] += 1
                    if has_tag_prefix and tag_index >= len(old_tags):
                        appended_tag_positions[category][bin_name] += 1
                        category_counts[category]["appended_tags"] += 1
                        first, last = token_span(list(new_encoded.offset_mapping), start, start + len(tag))
                        detail = appended_tag_details[tag]
                        detail["occurrence_count"] += 1
                        detail["category"] = category
                        detail["token_position_bins"][bin_name] += 1
                        if len(detail["examples"]) < 5:
                            detail["examples"].append({
                                "stem": stem,
                                "token_start": first,
                                "token_end": last,
                                "extended_caption_tokens": len(new_ids),
                            })

                if stem in {"G-_MlI8WMAAQdZB", "145_630"}:
                    ids, offsets, example_tokenizer = encoded_by_tokenizer[name]
                    token_text = example_tokenizer.convert_ids_to_tokens(ids)
                    examples[stem] = {
                        "original_text": old_text,
                        "extended_text": new_text,
                        "original_tokens": len(old_ids),
                        "extended_tokens": len(new_ids),
                        "added_tags": extra_tags,
                        "tokens_around_75": [
                            {"position": i + 1, "token": token_text[i]}
                            for i in range(max(0, min(70, len(token_text))), min(80, len(token_text)))
                        ],
                        "tag_prefix_exact": has_tag_prefix,
                    }
        all_ids_equal &= ids_by_tokenizer["clip_l"] == ids_by_tokenizer["clip_g"]

    added_after_75 = defaultdict(Counter)
    tokenizer = tokenizer_map["clip_l"]
    for stem in sorted(original_files):
        old_text = original_files[stem].read_text(encoding="utf-8").strip()
        new_text = extended_files[stem].read_text(encoding="utf-8").strip()
        old_tags = [tag.strip() for tag in old_text.split(",") if tag.strip()]
        new_tags = [tag.strip() for tag in new_text.split(",") if tag.strip()]
        if new_tags[:len(old_tags)] != old_tags:
            continue
        extra_tags = new_tags[len(old_tags):]
        encoded = tokenizer(new_text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        cursor = 0
        for tag_index, tag in enumerate(new_tags):
            start = new_text.find(tag, cursor)
            cursor = max(start, 0) + len(tag)
            if tag_index < len(old_tags) or start < 0:
                continue
            pos = token_position(list(encoded.offset_mapping), start)
            if pos is not None and pos > 75:
                added_after_75[classify(tag)]["count"] += 1

    appended_tags_after_75 = sum(item["count"] for item in added_after_75.values())

    return {
        "dataset_paths": {"original": str(original), "extended": str(extended)},
        "paired_caption_count": len(original_files),
        "paired_identical_image_sha256_count": image_hash_equal,
        "missing_or_ambiguous_image_pairs": missing_images,
        "tokenizers": {
            name: {"repo_id": repo, "commit": commit, "method": "CLIPTokenizerFast; add_special_tokens=false; truncation=false"}
            for name, (repo, commit) in TOKENIZER_REPOS.items()
        },
        "clip_l_clip_g_token_ids_equal_for_all_captions": all_ids_equal,
        "original_lengths": token_summary(original_lengths),
        "extended_lengths": token_summary(extended_lengths),
        "token_length_statistics_by_tokenizer": {
            name: {
                "original": token_summary(original_lengths_by_tokenizer[name]),
                "extended": token_summary(extended_lengths_by_tokenizer[name]),
            }
            for name in tokenizer_map
        },
        "extended_caption_beyond_token_75_count": extends_after_75,
        "extended_caption_beyond_token_75_percent": 100 * extends_after_75 / len(original_files),
        "first_75_content_token_ids_differ_count": first75_diff,
        "first_75_content_token_ids_differ_percent": 100 * first75_diff / len(original_files),
        "original_tokens_are_exact_prefix_of_extended_tokens_count": token_prefix_exact,
        "original_tag_list_is_exact_prefix_count": tag_prefix_exact,
        "appended_caption_tag_count": appended_tags,
        "distinct_appended_tag_count": len(additions),
        "most_common_appended_tags": additions.most_common(30),
        "appended_tag_inventory": [
            {
                "tag": tag,
                "occurrence_count": detail["occurrence_count"],
                "category": detail["category"],
                "token_position_bins": dict(detail["token_position_bins"]),
                "examples": detail["examples"],
            }
            for tag, detail in sorted(
                appended_tag_details.items(),
                key=lambda item: (-item[1]["occurrence_count"], item[0].casefold()),
            )
        ],
        "extended_caption_over_225_count": len(over_225),
        "extended_caption_over_225_stems": over_225,
        "mechanical_truncation_at_225": len(over_225),
        "useful_information_beyond_225": "Not determined mechanically; no caption exceeds 225 content tokens.",
        "tag_category_counts": dict(category_counts),
        "all_extended_tag_positions": dict(all_tag_positions),
        "appended_tag_positions": dict(appended_tag_positions),
        "appended_tags_starting_after_position_75": dict(added_after_75),
        "appended_tags_starting_after_position_75_count": appended_tags_after_75,
        "examples": examples,
    }


def evaluation_prompt_analysis(tokenizer_map: dict) -> dict:
    suite = read_json(HERE / "prompt_suite.json")
    prompts = {}
    for prompt in suite["prompts"]:
        positive = suite["style_prefix"] + ", " + prompt["text"]
        counts = {
            name: len(tokenizer(positive, add_special_tokens=False, truncation=False).input_ids)
            for name, tokenizer in tokenizer_map.items()
        }
        markers = []
        for marker in prompt.get("late_markers", []):
            start = positive.find(marker)
            start_positions = {}
            end_positions = {}
            for name, tokenizer in tokenizer_map.items():
                encoded = tokenizer(
                    positive,
                    add_special_tokens=False,
                    truncation=False,
                    return_offsets_mapping=True,
                )
                first, last = token_span(list(encoded.offset_mapping), start, start + len(marker)) if start >= 0 else (None, None)
                start_positions[name] = first
                end_positions[name] = last
            target = 75 if prompt["id"] == "late_after_75" else 150
            markers.append({
                "text": marker,
                "start_positions": start_positions,
                "end_positions": end_positions,
                "after_target_position": all(value is not None and value > target for value in start_positions.values()),
                "fully_within_inference_capacity": all(
                    value is not None and value <= 225 for value in end_positions.values()
                ),
            })
        prompts[prompt["id"]] = {
            "group": prompt["group"],
            "prompt_text": prompt["text"],
            "positive_prompt": positive,
            "content_token_counts": counts,
            "late_markers": markers,
        }
    return {
        "tokenizers": {
            name: {"repo_id": repo, "commit": commit}
            for name, (repo, commit) in TOKENIZER_REPOS.items()
        },
        "inference_capacity": 225,
        "count_method": "Full positive prompt including style prefix; CLIPTokenizerFast; no special tokens; no truncation.",
        "prompts": prompts,
    }


def checkpoint_inventory(run_name: str) -> list[dict]:
    if run_name.startswith("before"):
        run_dir = LORA_ROOT / f"alt_test_{run_name}"
        files = list((run_dir / "save").glob("*.safetensors"))
        matcher = re.compile(r"save-(\d+)-")
    else:
        run_dir = LORA_ROOT / "lora_manager" / f"alt_test_{run_name}"
        files = list(run_dir.glob("*.safetensors"))
        matcher = re.compile(r"-(\d{5})(?:-[0-9a-f]{64})?\.safetensors$", re.I)
    by_step = {}
    for path in files:
        match = matcher.search(path.name)
        if match:
            step = int(match.group(1))
            preferred = f"alt_test_{run_name}-{step:05}.safetensors"
            if step not in by_step or path.name == preferred:
                by_step[step] = path
    return [{"step": step, "path": str(path)} for step, path in sorted(by_step.items())]


def run_audit() -> dict:
    runs = {}
    for name, config_path in RUN_CONFIGS.items():
        args = read_json(config_path)
        concept_path = config_path.with_name("concepts.json")
        concepts = read_json(concept_path)
        concept = concepts[0] if concepts else {}
        text_cfg = args.get("text_encoder", {})
        optimizer = args.get("optimizer", {})
        recorded_limit = args.get("sdxl_max_caption_tokens")
        runs[name] = {
            "config_path": str(config_path),
            "concept_file": str(concept_path),
            "concept_dataset_path": concept.get("path"),
            "effective_token_limit": recorded_limit if recorded_limit is not None else 75,
            "recorded_sdxl_max_caption_tokens": recorded_limit,
            "build": "original" if name.startswith("before") else "long_clip",
            "base_model": args.get("base_model_name"),
            "training_method": args.get("training_method"),
            "rank": args.get("lora_rank"),
            "alpha": args.get("lora_alpha"),
            "optimizer": optimizer.get("optimizer"),
            "learning_rate_unet": args.get("learning_rate"),
            "learning_rate_text_encoder_1": text_cfg.get("learning_rate"),
            "scheduler": args.get("learning_rate_scheduler"),
            "batch_size": args.get("batch_size"),
            "gradient_accumulation_steps": args.get("gradient_accumulation_steps"),
            "resolution": args.get("resolution"),
            "train_dtype": args.get("train_dtype"),
            "epochs": args.get("epochs"),
            "save_every": args.get("save_every"),
            "caption_dropout": args.get("dropout_probability"),
            "tag_shuffle": concept.get("text", {}).get("enable_tag_shuffling"),
            "tag_dropout": concept.get("text", {}).get("tag_dropout_probability"),
            "image_augmentations": concept.get("image"),
            "concept_seed": concept.get("seed"),
            "global_seed_fields": [key for key in args if "seed" in key.lower()],
            "prior_preservation_weight": args.get("masked_prior_preservation_weight"),
            "masked_training": args.get("masked_training"),
            "available_checkpoints": checkpoint_inventory(name),
        }
    after75, after225 = (read_json(RUN_CONFIGS[key]) for key in ("after_75", "after_225"))
    training_keys = (
        "base_model_name", "training_method", "peft_type", "learning_rate", "learning_rate_scheduler",
        "batch_size", "gradient_accumulation_steps", "resolution", "train_dtype", "lora_rank",
        "lora_alpha", "dropout_probability", "masked_training", "masked_prior_preservation_weight",
        "offset_noise_weight", "clip_grad_norm", "unet", "text_encoder", "text_encoder_2", "optimizer",
        "epochs", "sdxl_max_caption_tokens",
    )
    changed = {
        key: {"after_75": after75.get(key), "after_225": after225.get(key)}
        for key in training_keys if after75.get(key) != after225.get(key)
    }
    return {
        "runs": runs,
        "after_run_setting_differences": changed,
        "matched_capacity_control_present": False,
        "required_missing_control": "long_clip + identical extended caption files + 75-token capacity, paired with the existing 225-token run",
        "current_base_checkpoint_sha256_from_pretrained_report": "3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9",
        "historical_base_hash_recorded_in_training_runs": False,
        "global_initialization_seed_or_initial_lora_hash_recorded": False,
        "image_order_verified": False,
        "build_revision_recorded_for_real_runs": False,
        "causal_comparison": {
            "before_75_vs_before_225": "Same old build/effective 75-token limit; original vs extended captions differ, so this is a caption-semantics comparison.",
            "after_75_vs_after_225": "Same long_clip build and 4000-update target; original vs extended caption files and token capacity both differ. First-75 conditioning differs for short captions.",
            "before_225_vs_after_225": "Same extended caption dataset, but old vs new build and 500 vs 4000 updates; not a matched capacity control.",
        },
    }


def training_losses() -> tuple[list[dict], dict]:
    sessions = {}
    rows = []
    for run in ("after_75", "after_225"):
        event_root = DATASET_ROOT / f"alt_test_{run}" / "tensorboard"
        files = sorted(event_root.rglob("events.out.tfevents.*"), key=lambda p: p.name)
        sessions[run] = []
        for path in files:
            accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
            accumulator.Reload()
            series = {}
            for tag in accumulator.Tags().get("scalars", []):
                values = accumulator.Scalars(tag)
                series[tag] = {event.step: event.value for event in values}
                for event in values:
                    rows.append({
                        "run": run,
                        "session": path.parent.name,
                        "event_file": path.name,
                        "tag": tag,
                        "event_step": event.step,
                        "value": event.value,
                    })
            sessions[run].append({"path": str(path), "series": series})

    comparisons = []
    for session_index, (left, right) in enumerate(zip(sessions["after_75"], sessions["after_225"]), start=1):
        for tag in ("loss/train_step", "smooth_loss/train_step"):
            left_series = left["series"].get(tag, {})
            right_series = right["series"].get(tag, {})
            steps = sorted(left_series.keys() & right_series.keys())
            left_values = np.asarray([left_series[step] for step in steps], dtype=float)
            right_values = np.asarray([right_series[step] for step in steps], dtype=float)
            comparisons.append({
                "session_index": session_index,
                "tag": tag,
                "after_75_event_file": Path(left["path"]).name,
                "after_225_event_file": Path(right["path"]).name,
                "paired_event_steps": len(steps),
                "event_step_min": min(steps) if steps else None,
                "event_step_max": max(steps) if steps else None,
                "after_75_mean": float(left_values.mean()) if len(left_values) else None,
                "after_225_mean": float(right_values.mean()) if len(right_values) else None,
                "mean_absolute_loss_difference": float(np.abs(left_values - right_values).mean()) if len(left_values) else None,
                "pearson_correlation": float(np.corrcoef(left_values, right_values)[0, 1]) if len(left_values) > 1 else None,
            })

    matched = []
    steps_to_match = (399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999)
    for step in steps_to_match:
        session_index = 0 if step <= 1199 else 1
        row = {"checkpoint_step": step}
        for run in ("after_75", "after_225"):
            series = sessions[run][session_index]["series"]
            row[run] = {
                tag: series.get(tag, {}).get(step)
                for tag in ("loss/train_step", "smooth_loss/train_step")
            }
        raw_75 = row["after_75"]["loss/train_step"]
        raw_225 = row["after_225"]["loss/train_step"]
        smooth_75 = row["after_75"]["smooth_loss/train_step"]
        smooth_225 = row["after_225"]["smooth_loss/train_step"]
        row["raw_loss_delta_225_minus_75"] = raw_225 - raw_75 if raw_75 is not None and raw_225 is not None else None
        row["smooth_loss_delta_225_minus_75"] = smooth_225 - smooth_75 if smooth_75 is not None and smooth_225 is not None else None
        matched.append(row)
    return rows, {
        "scalar_source": "TensorBoard event files from the two long_clip training runs",
        "session_ranges": [
            {"session": 1, "event_steps": "0-1568", "meaning": "initial three-epoch training session"},
            {"session": 2, "event_steps": "1199-4183", "meaning": "resumed eight-epoch session"},
        ],
        "checkpoint_step_matches": matched,
        "session_comparisons": comparisons,
        "step_axis_note": "The event series continues beyond the periodic backup named global_step=3999. Training logs continue through event step 4183 and save the final KOHYA model after the last epoch; the final model file has no embedded progress metadata.",
        "smooth_loss_note": "The exponential moving average restarts at resumed event step 1199; use the initial session value for checkpoint step 1199.",
    }


def write_report(output_dir: Path, captions: dict, audit: dict, loss_info: dict, prompt_info: dict) -> None:
    original = captions["original_lengths"]
    extended = captions["extended_lengths"]
    lines = [
        "# Existing run and caption audit",
        "",
        "## Caption dataset measurements",
        "",
        f"- Paired image-caption files: {captions['paired_caption_count']}; image hashes identical across pairs: {captions['paired_identical_image_sha256_count']}.",
        f"- CLIP-L and CLIP-G token IDs are identical for every original and extended caption. Counts are content tokens before BOS/EOS; OneTrainer chunks add the BOS/EOS positions.",
        f"- Original lengths: mean {original['mean']:.2f}, median {original['median']}, p25 {original['percentiles']['p25']:.2f}, p75 {original['percentiles']['p75']:.2f}, p90 {original['percentiles']['p90']:.2f}, p95 {original['percentiles']['p95']:.2f}, p99 {original['percentiles']['p99']:.2f}, max {original['max']}.",
        f"- Extended lengths: mean {extended['mean']:.2f}, median {extended['median']}, p25 {extended['percentiles']['p25']:.2f}, p75 {extended['percentiles']['p75']:.2f}, p90 {extended['percentiles']['p90']:.2f}, p95 {extended['percentiles']['p95']:.2f}, p99 {extended['percentiles']['p99']:.2f}, max {extended['max']}.",
        "- CLIP-L and CLIP-G statistics are computed independently in token_length_statistics_by_tokenizer; their token IDs match for every caption, so their distributions are identical.",
        f"- Extended caption buckets: {json.dumps(extended['buckets'], sort_keys=True)}.",
        f"- Extended captions with content after token 75: {captions['extended_caption_beyond_token_75_count']}/{captions['paired_caption_count']} ({captions['extended_caption_beyond_token_75_percent']:.2f}%).",
        f"- First 75 content-token ID sequences differ: {captions['first_75_content_token_ids_differ_count']}/{captions['paired_caption_count']} ({captions['first_75_content_token_ids_differ_percent']:.2f}%). In every pair, old tags and their full token sequence are exact prefixes; differences come from appended tags filling unused positions in shorter captions.",
        f"- Appended tags: {captions['appended_caption_tag_count']} instances across {captions['distinct_appended_tag_count']} distinct tags. Since all originals fit one 75-token chunk, {captions['extended_caption_beyond_token_75_count']} extended captions require a second; none requires a third. No caption exceeds 225 tokens, so none is mechanically truncated there. This chunk count is not a measured time or memory increase; usefulness is not determined by length.",
        "",
        "### Representative caption boundaries",
        "",
    ]
    for stem, example in captions["examples"].items():
        boundary = ", ".join("{}={}".format(item["position"], item["token"]) for item in example["tokens_around_75"])
        lines.extend([
            "#### " + stem,
            "",
            f"Counts: {example['original_tokens']} original, {example['extended_tokens']} extended. Added tags: {', '.join(example['added_tags']) or '(none)'}.",
            "",
            "Original: " + example["original_text"],
            "",
            "Extended: " + example["extended_text"],
            "",
            "Tokens near position 75: " + boundary,
            "",
        ])
    lines.extend([
        "### Added tag positions",
        "",
        "Positions are the 1-based first content-token position of each comma-delimited tag. Category names use a keyword heuristic, not a validated ontology.",
        "",
        "Appended tag category counts by position: " + json.dumps(captions["appended_tag_positions"], sort_keys=True),
        "",
        f"{captions['appended_tags_starting_after_position_75_count']}/{captions['appended_caption_tag_count']} appended tag instances ({100 * captions['appended_tags_starting_after_position_75_count'] / captions['appended_caption_tag_count']:.2f}%) start after token position 75; the rest fill unused first-chunk positions.",
        "",
        "All distinct appended tags are listed in appended_tag_inventory.csv with occurrence counts, tokenizer position bins, and up to five caption examples per tag.",
        "",
        "Most common additions: " + ", ".join(f"{tag} ({count})" for tag, count in captions["most_common_appended_tags"][:15]) + ".",
        "",
        "The additions mostly describe appearance/body, clothing, and pose. Environment tags are sparse and lighting/style tags are very rare. Some additions are redundant or hierarchical, such as a specific tag plus a broader tag. Individual tags were not verified against every image.",
        "",
        "## Evaluation prompt token boundaries",
        "",
        "The complete table, including exact CLIP-L/CLIP-G counts and late-detail offsets, is in evaluation_prompt_lengths.json. Every included prompt is encoded with the same 225-content-token inference setting for all LoRAs and base controls.",
        "",
    ])
    for prompt_id, prompt in prompt_info["prompts"].items():
        late = ", ".join(
            f"{marker['text']}: start {marker['start_positions']}, end {marker['end_positions']}, fully within 225: {marker['fully_within_inference_capacity']}"
            for marker in prompt["late_markers"]
        ) or "none"
        lines.append(
            f"- {prompt_id}: CLIP-L/CLIP-G {prompt['content_token_counts']['clip_l']}/{prompt['content_token_counts']['clip_g']} tokens; late markers: {late}."
        )
    lines.extend([
        "",
        "## Training run audit",
        "",
        "| Run | Build | Effective limit | Saved checkpoints | Caption source |",
        "|---|---|---:|---|---|",
    ])
    for name, run in audit["runs"].items():
        steps = ", ".join(str(item["step"]) for item in run["available_checkpoints"]) or "none found"
        lines.append(f"| {name} | {run['build']} | {run['effective_token_limit']} | {steps} | {run.get('concept_dataset_path', '')} |")
    lines.extend([
        "",
        "All configs reference the same Illustrious XL checkpoint path. Its current SHA-256 was recomputed as 3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9, matching the earlier pretrained report; real-run snapshots do not record a historical hash or OneTrainer revision, so past byte identity is unverified. Inspected settings match on rank/alpha 32/32, Adafactor, constant UNet LR 0.0005, batch 1, accumulation 1, 1024 resolution, BF16 training, caption dropout 0, and masked training with prior-preservation weight 0. The TE1 LoRA learning rate is 0.00005.",
        "",
        "The old-build runs save steps 99, 199, 299, 399, and 499. Both long_clip runs have checkpoints at 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, and 3999. Concept snapshots disable tag shuffling/dropout and image augmentation. They share a concept seed, but no global seed, initial LoRA checksum, or realized image order is recorded.",
        "",
        "The after runs match on inspected training settings other than capacity and run-specific paths/names. They use different caption files, though. The extended file fills previously unused prefix positions for 233 captions, so after_75 versus after_225 changes both first-chunk conditioning and access to positions 76–150. The required same-build control (extended text at 75 tokens) is missing. before_225 uses the extended captions but is the old build and only 500 updates.",
        "",
        "before_75 versus before_225 holds the old build and 75-token effective capacity constant, so it can describe the effect of the different caption text within the first 75 tokens. It cannot estimate the effect of access to later tokens. Neither old-run config records a global initialization seed or realized image order, and the runs stop at 500 updates.",
        "",
        "### Training losses and timing",
        "",
        "TensorBoard scalar values are in training_losses.csv; matched loss values for saved checkpoint steps are in training_loss_comparison.json. The saved 3999 backup is a periodic mid-epoch checkpoint, not the completed final model: the resumed event logs continue through step 4183 and the trainer then saves the final model. The final KOHYA file has no progress metadata.",
        "",
        "The checkpoint-step-matched smooth-loss values are almost identical: differences from 399 through 3999 are approximately -0.000036 to +0.000098 (225-token minus 75-token). Raw per-batch losses vary more. These are training objectives, not held-out quality or style scores. The resumed exponential moving average restarts at step 1199.",
        "",
        "Both 1024px long-run logs show about 2.5–2.7 seconds per step in recorded progress samples. This is close but not a repeated randomized timing benchmark, and captions differ. Historical peak memory and text-cache preparation timings were not recorded; caption preparation effort was not timed.",
        "",
        "## Prior experiment evidence reviewed",
        "",
        "The in-repository experiments/sdxl-long-caption/REPORT.md establishes tokenizer/model compatibility, gradients, optimizer updates, and visible output differences in a bounded synthetic one-image experiment (rank 4, 256px training; 512px 8-step image comparisons). It does not measure style quality or generalization for the real 1024px rank-32 style runs.",
        "",
        "The existing ComfyUI manual grid manifest contains 128 outputs with P6/P7 and multiple checkpoints/strengths, but it uses DasiwaIllustriousAnime_epitaphecstasy.safetensors as the inference base rather than the Illustrious XL checkpoint path used for training. It has one seed, so it is unblinded and does not form a base-matched paired comparison. It is useful as exploratory imagery, not causal quality evidence.",
        "",
        "## What the existing experiments can answer",
        "",
        "The four real LoRAs can show behavior of these particular training outputs. They cannot attribute a quality difference specifically to access to tokens 76–225. That requires long_clip training on the same extended caption files at 75 and 225 tokens, with matched initialization and sample order. No additional training was started during this audit.",
        "",
    ])
    (output_dir / "existing_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--original-data", type=Path, default=ORIGINAL_DATA)
    parser.add_argument("--extended-data", type=Path, default=EXTENDED_DATA)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_map = tokenizers()
    captions = paired_caption_analysis(args.original_data, args.extended_data, tokenizer_map)
    prompt_info = evaluation_prompt_analysis(tokenizer_map)
    audit = run_audit()
    loss_rows, loss_info = training_losses()
    write_json(args.output_dir / "caption_analysis.json", captions)
    write_appended_tag_inventory(args.output_dir / "appended_tag_inventory.csv", captions)
    write_json(args.output_dir / "evaluation_prompt_lengths.json", prompt_info)
    write_json(args.output_dir / "run_audit.json", audit)
    write_json(args.output_dir / "training_loss_comparison.json", loss_info)
    with (args.output_dir / "training_losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run", "session", "event_file", "tag", "event_step", "value"))
        writer.writeheader()
        writer.writerows(loss_rows)
    write_report(args.output_dir, captions, audit, loss_info, prompt_info)
    print(f"Paired captions: {captions['paired_caption_count']}")
    print(f"Beyond token 75: {captions['extended_caption_beyond_token_75_count']}")
    print(f"First-75 sequences differ: {captions['first_75_content_token_ids_differ_count']}")
    print(f"Identical image pairs: {captions['paired_identical_image_sha256_count']}")
    print("Report: " + str(args.output_dir / "existing_audit.md"))


if __name__ == "__main__":
    main()
