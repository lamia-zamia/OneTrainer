from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SCORE_FIELDS = {
    "prompt_compliance": "prompt_compliance_1to5",
    "style_fidelity": "style_fidelity_1to5",
    "anatomy_artifact": "anatomy_artifact_0to3",
    "unwanted_prior": "unwanted_prior_0to3",
}


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean_delta_225_minus_75": None, "median_delta_225_minus_75": None}
    return {
        "n": len(values),
        "mean_delta_225_minus_75": statistics.mean(values),
        "median_delta_225_minus_75": statistics.median(values),
        "p10_delta_225_minus_75": sorted(values)[round((len(values) - 1) * 0.10)],
        "p90_delta_225_minus_75": sorted(values)[round((len(values) - 1) * 0.90)],
        "direction": "higher favors 225" if values and len(values) else "unscored",
    }


def paired_delta(row: dict, conditions: dict, suffix: str) -> float | None:
    a_value = row.get("A_" + suffix, "").strip()
    b_value = row.get("B_" + suffix, "").strip()
    if not a_value or not b_value:
        return None
    scores = {"A": float(a_value), "B": float(b_value)}
    treatment_values = {
        conditions["A_condition"]: scores["A"],
        conditions["B_condition"]: scores["B"],
    }
    if "after_75" not in treatment_values or "after_225" not in treatment_values:
        return None
    return treatment_values["after_225"] - treatment_values["after_75"]


def cluster_bootstrap_ci(values_by_prompt: dict[str, list[float]], seed: int, replicates: int = 2000) -> list[float] | None:
    nonempty = {key: values for key, values in values_by_prompt.items() if values}
    if len(nonempty) < 2:
        return None
    rng = random.Random(seed)
    prompts = list(nonempty)
    means = []
    for _ in range(replicates):
        sample = []
        for prompt in rng.choices(prompts, k=len(prompts)):
            sample.extend(rng.choices(nonempty[prompt], k=len(nonempty[prompt])))
        means.append(statistics.mean(sample))
    means.sort()
    return [means[round((replicates - 1) * 0.025)], means[round((replicates - 1) * 0.975)]]


def score_direction_counts(values: list[float], metric: str) -> dict:
    higher_better = metric in {"prompt_compliance", "style_fidelity"}
    wins_225 = sum(value > 0 if higher_better else value < 0 for value in values)
    wins_75 = sum(value < 0 if higher_better else value > 0 for value in values)
    ties = sum(value == 0 for value in values)
    return {
        "after_225_better": wins_225,
        "after_75_better": wins_75,
        "tied": ties,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed blind scores after revealing the key.")
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    with (args.result_dir / "human_scoring.csv").open(newline="", encoding="utf-8") as stream:
        score_rows = list(csv.DictReader(stream))
    key_rows = json.loads((args.result_dir / "blind_key.json").read_text(encoding="utf-8"))
    keys = {row["pair_id"]: row for row in key_rows}

    deltas_by_strength = defaultdict(lambda: defaultdict(list))
    deltas_by_prompt = defaultdict(lambda: defaultdict(list))
    preferences_by_strength = defaultdict(Counter)
    preferences_by_prompt_strength = defaultdict(Counter)
    scored_pair_count = 0
    scored_pair_rows = set()
    for row in score_rows:
        key = keys[row["pair_id"]]
        conditions = {
            "A_condition": key["A_condition"],
            "B_condition": key["B_condition"],
        }
        strength = str(row["strength"])
        for metric, suffix in SCORE_FIELDS.items():
            delta = paired_delta(row, conditions, suffix)
            if delta is not None:
                deltas_by_strength[strength][metric].append(delta)
                deltas_by_prompt[(row["prompt_id"], strength)][metric].append(delta)
                scored_pair_count += 1
                scored_pair_rows.add(row["pair_id"])
        preference = row.get("pair_style_preference_A_B_tie", "").strip().upper()
        if preference in {"A", "B", "TIE"}:
            chosen_condition = "tie" if preference == "TIE" else conditions[preference + "_condition"]
            preferences_by_strength[strength][chosen_condition] += 1
            preferences_by_prompt_strength[(row["prompt_id"], strength)][chosen_condition] += 1

    score_summary = {}
    prompt_score_summary = {}
    for strength, metrics in deltas_by_strength.items():
        score_summary[strength] = {}
        for metric, values in metrics.items():
            # Cluster by prompt while resampling seed-level paired scores within each prompt.
            per_prompt = {
                prompt_id: prompt_metrics.get(metric, [])
                for (prompt_id, prompt_strength), prompt_metrics in deltas_by_prompt.items()
                if prompt_strength == strength
            }
            metric_result = describe(values)
            metric_result["direction"] = (
                "higher favors 225" if metric in {"prompt_compliance", "style_fidelity"}
                else "lower favors 225"
            )
            metric_result["pair_direction_counts"] = score_direction_counts(values, metric)
            metric_result["prompt_cluster_bootstrap_95pct_ci_for_mean_delta"] = cluster_bootstrap_ci(
                per_prompt, seed=20261027 + int(float(strength) * 100) + len(metric)
            )
            score_summary[strength][metric] = metric_result
        score_summary[strength]["pair_style_preference"] = dict(preferences_by_strength[strength])

    for (prompt_id, strength), metrics in deltas_by_prompt.items():
        prompt_key = f"{prompt_id}|strength={strength}"
        prompt_score_summary[prompt_key] = {}
        for metric, values in metrics.items():
            metric_result = describe(values)
            metric_result["direction"] = (
                "higher favors 225" if metric in {"prompt_compliance", "style_fidelity"}
                else "lower favors 225"
            )
            metric_result["pair_direction_counts"] = score_direction_counts(values, metric)
            prompt_score_summary[prompt_key][metric] = metric_result
        prompt_score_summary[prompt_key]["pair_style_preference"] = dict(
            preferences_by_prompt_strength[(prompt_id, strength)]
        )

    extended_summary = None
    extended_csv = args.result_dir / "human_extended_scoring.csv"
    extended_key_path = args.result_dir / "extended_blind_key.json"
    if extended_csv.exists() and extended_key_path.exists():
        with extended_csv.open(newline="", encoding="utf-8") as stream:
            extended_rows = list(csv.DictReader(stream))
        extended_keys = {
            row["comparison_id"]: row
            for row in json.loads(extended_key_path.read_text(encoding="utf-8"))
        }
        detail_values = defaultdict(list)
        detail_values_by_prompt = defaultdict(list)
        for row in extended_rows:
            key = extended_keys[row["comparison_id"]]
            for label in ("A", "B", "C"):
                value = row.get(label + "_late_detail_compliance_1to5", "").strip()
                if value:
                    condition = key[label + "_condition"]
                    score = float(value)
                    detail_values[condition].append(score)
                    detail_values_by_prompt[(row["prompt_id"], condition)].append(score)
        extended_summary = {
            "overall": {
                condition: {
                    "n": len(values),
                    "mean_late_detail_compliance_1to5": statistics.mean(values) if values else None,
                    "median_late_detail_compliance_1to5": statistics.median(values) if values else None,
                }
                for condition, values in detail_values.items()
            },
            "by_prompt": {
                f"{prompt_id}|{condition}": {
                    "n": len(values),
                    "mean_late_detail_compliance_1to5": statistics.mean(values),
                    "median_late_detail_compliance_1to5": statistics.median(values),
                }
                for (prompt_id, condition), values in detail_values_by_prompt.items()
            },
        }

    result = {
        "result_dir": str(args.result_dir),
        "scored_pair_rows": len(scored_pair_rows),
        "scored_metric_comparison_count": scored_pair_count,
        "expected_pair_rows": len(score_rows),
        "scores_by_strength": score_summary,
        "scores_by_prompt_and_strength": prompt_score_summary,
        "extended_prompt_late_detail_compliance": extended_summary,
        "limitations": (
            "Report each prompt and strength alongside pooled estimates. The prompt-cluster bootstrap interval "
            "resamples this fixed prompt suite and paired seeds; it does not establish generalization to all prompts. "
            "A wide interval or a small score difference is not evidence of equivalence."
        ),
    }
    output_path = args.result_dir / "human_score_summary.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
