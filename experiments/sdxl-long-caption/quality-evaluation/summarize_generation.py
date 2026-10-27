from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def describe(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    metadata_path = args.result_dir / "metadata.jsonl"
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line]
    by_run = defaultdict(list)
    paired = defaultdict(dict)
    for index, record in enumerate(records):
        by_run[record["run"]].append(record)
        if record["run"] in {"after_75", "after_225"}:
            key = (
                record["prompt_id"], record["checkpoint_step"],
                record["strength"], record["seed"],
            )
            paired[key][record["run"]] = (index, record)

    complete_pairs = [pair for pair in paired.values() if set(pair) == {"after_75", "after_225"}]
    pair_time_deltas = []
    pair_memory_deltas = []
    run_summary = {}
    for run, run_records in by_run.items():
        run_summary[run] = {
            "image_count": len(run_records),
            "generation_seconds": describe([r["generation_seconds"] for r in run_records]),
            "peak_cuda_allocated_mib": describe([r["peak_cuda_allocated_bytes"] / 1024**2 for r in run_records]),
            "peak_cuda_reserved_mib": describe([r["peak_cuda_reserved_bytes"] / 1024**2 for r in run_records]),
        }

    paired_summaries = defaultdict(list)
    pair_order = []
    for pair in complete_pairs:
        left_index, left = pair["after_75"]
        right_index, right = pair["after_225"]
        delta_time = right["generation_seconds"] - left["generation_seconds"]
        delta_memory = (right["peak_cuda_allocated_bytes"] - left["peak_cuda_allocated_bytes"]) / 1024**2
        pair_time_deltas.append(delta_time)
        pair_memory_deltas.append(delta_memory)
        paired_summaries[str(left["strength"])].append({
            "prompt_id": left["prompt_id"],
            "checkpoint_step": left["checkpoint_step"],
            "seed": left["seed"],
            "after_225_minus_after_75_seconds": delta_time,
            "after_225_minus_after_75_peak_allocated_mib": delta_memory,
        })
        pair_order.append("after_75_first" if left_index < right_index else "after_225_first")

    checkpoint_paths = sorted({
        Path(record["checkpoint_path"])
        for record in records
        if record.get("checkpoint_path")
    }, key=lambda path: str(path).casefold())
    checkpoint_inventory = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        }
        for path in checkpoint_paths
    ]

    result = {
        "stage": records[0]["stage"] if records else None,
        "record_count": len(records),
        "paired_lora_comparison_count": len(complete_pairs),
        "unpaired_treatment_records": sum(
            len(pair) != 2 for pair in paired.values()
        ),
        "base_model_control_count": len(by_run.get("base", [])),
        "base_checkpoint": records[0].get("base_checkpoint") if records else None,
        "base_checkpoint_sha256": records[0].get("base_checkpoint_sha256") if records else None,
        "checkpoint_file_inventory": checkpoint_inventory,
        "per_run_resource_summary": run_summary,
        "paired_resource_deltas": {
            "after_225_minus_after_75_generation_seconds": describe(pair_time_deltas),
            "after_225_minus_after_75_peak_allocated_mib": describe(pair_memory_deltas),
            "by_strength": {key: values for key, values in paired_summaries.items()},
            "generation_order_counts": {
                "after_75_first": pair_order.count("after_75_first"),
                "after_225_first": pair_order.count("after_225_first"),
            },
        },
        "interpretation": (
            "Resource measurements describe this inference harness at fixed 225-token capacity. "
            "They do not measure training throughput, training peak memory, or caption-cache preparation. "
            "Generation order is not randomized, so small timing differences should not be treated as causal."
        ),
    }
    out_path = args.result_dir / "resource_summary.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
