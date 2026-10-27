# SDXL long-caption quality evaluation

This directory audits the four existing training runs and provides a reproducible, blinded comparison of the two `long_clip` LoRAs. The comparison answers whether these particular 75/225-caption runs differ. They do not isolate the causal effect of tokens 76–225 because the training caption text also differs and 233/523 extended captions fill unused positions in the first 75 tokens.

## Files

- `existing_audit.md` — run audit, caption statistics, representative tokenizer boundaries, loss comparison, and known confounders.
- `caption_analysis.json`, `appended_tag_inventory.csv`, `run_audit.json`, `training_loss_comparison.json`, `training_losses.csv` — machine-readable audit evidence. The inventory lists each appended tag, occurrence count, category heuristic, token-position buckets, and representative caption filenames.
- `evaluation_prompt_lengths.json` — exact CLIP-L/CLIP-G prompt lengths and token spans of late details.
- `prompt_suite.json` — generation prompts. P6 and P7 match the supplied prompts exactly before the common style prefix.
- `analyze_existing.py` — regenerates the audit reports from the located configs, checkpoints, TensorBoard logs, caption files, and cached tokenizers.
- `generate_quality_evaluation.py` — staged, seeded generation with per-image metadata, blinded A/B and A/B/C sheets, and scoring templates.
- `summarize_generation.py` — summarizes per-image inference time and peak CUDA memory and records hashes for the exact LoRA checkpoint files used, without turning resource differences into quality claims.
- `summarize_human_scores.py` — summarizes a completed blind scoring CSV after the scorer has saved it and is ready to reveal the key.
- `reference_style_sheet.png` — eight repeatable style references selected from captions that include clothing tags and exclude explicit, sexual, underwear, and age-risk tags. `reference_style_manifest.json` records their source files and hashes.
- `results/` — generated comparison images, metadata, contact sheets, and blind scoring keys.

## Reproduce the audit

From the repository root, with the existing Windows virtual environment and locally available training files:

```powershell
.\venv\Scripts\python.exe experiments\sdxl-long-caption\quality-evaluation\analyze_existing.py
.\venv\Scripts\python.exe experiments\sdxl-long-caption\quality-evaluation\make_reference_sheet.py
```

The tokenizer snapshots are pinned by commit in the report and loaded from the local Hugging Face cache. The analyzer does not alter the source datasets, LoRAs, or training runs.

## Generate comparisons

Run one stage at a time. The pilot checks one P6 pair at 1024px. The screen renders 360 LoRA images (15 prompts, four paired seeds, three strengths, both runs) plus eight base-model long-prompt controls. The progression stage renders 192 LoRA images at steps 1199, 2399, 3199, and 3999 for P6/P7 at strengths 1, 1.5, and 2 with four paired seeds.

```powershell
.\venv\Scripts\python.exe experiments\sdxl-long-caption\quality-evaluation\generate_quality_evaluation.py --stage pilot
.\venv\Scripts\python.exe experiments\sdxl-long-caption\quality-evaluation\generate_quality_evaluation.py --stage screen
.\venv\Scripts\python.exe experiments\sdxl-long-caption\quality-evaluation\generate_quality_evaluation.py --stage progression
```

Each stage writes a new timestamped subdirectory under `results/`. Use `--output-dir` to select another empty directory. Generation computes the local Illustrious XL checkpoint hash at startup and stops if it differs from the hash verified against the prior pretrained report. All outputs use 1024×1024 resolution, OneTrainer Euler ancestral sampler, 28 steps, CFG 3, the same negative prompt, and 225-token inference capacity for all LoRAs and base controls. Random seeds are paired across treatments. The late-detail prompts put their full marked details at tokens 142–150 and 205–223 in both tokenizers.

If generation is interrupted, rerun the same stage with the same output directory and `--resume`. Resume validates the stage, planned image assignments, saved PNGs, base checkpoint hash, and LoRA checkpoint file metadata before skipping completed images and continuing. It preserves any existing human scoring CSVs.

Raw 1024px PNGs stay in the local `results/*/images/` folders and are git-ignored; contact sheets, metadata, scoring CSVs, blind keys, and summaries are retained as review evidence.

The GPU records per-image generation time and peak allocated/reserved CUDA memory. The run-level timing includes prompt encoding and sampling, but excludes process startup and model loading. Existing training logs provide step-time estimates; historical peak training memory, text-cache preparation time, and caption-preparation labor were not recorded.

After generation finishes, summarize its inference resources with:

```powershell
.\venv\Scripts\python.exe experiments\sdxl-long-caption\quality-evaluation\summarize_generation.py experiments\sdxl-long-caption\quality-evaluation\results\<stage_directory>
```

## Blind scoring

Score `human_scoring.csv` while looking only at the corresponding `contact_sheets` and `reference_style_sheet.png`. Do not open `blind_key.json` or `extended_blind_key.json` until scoring is saved. A/B orientation is randomized independently for each paired seed/checkpoint/strength comparison. The image IDs in the CSV and contact sheets are opaque and can be used to join scores to `metadata.jsonl`.

Use the following rubric consistently:

- **Prompt compliance (1–5):** Score requested attributes only. For P6, consider hair, glasses, camisole, sofa, indoor living room, seated/crossed-leg pose, viewer gaze, and soft light. For P7 and other vague prompts, do not penalize unspecified clothes, colors, poses, or backgrounds.
- **Style fidelity (1–5):** Compare linework, shading, highlights, proportions, color treatment, and overall rendering with the reference sheet. Do not score composition similarity as style fidelity.
- **Anatomy artifact (0–3):** 0 none noticeable; 1 minor local defect; 2 clear, repeated, or distracting distortion; 3 severe defects or image collapse.
- **Unwanted prior (0–3):** 0 none; 1 one minor contradiction; 2 clear recurring contradiction; 3 strong/repeated contradiction. Only score content the prompt specifies (for example requested material, clothing coverage, lighting, subject count, or body type). Do not count an unrequested color or pose as leakage.
- **Pair preference:** Choose A, B, or tie for style quality, considering prompt compliance and artifacts. Leave blank if the evidence is genuinely unclear.
- **Extended prompt CSV:** For each blinded A/B/C long-prompt set, score whether the specified late clothing/object detail is visibly present. The base-model column estimates how much of that behavior is already present without either LoRA.

Fill numeric fields with the indicated scale and add concise notes for ambiguous cases. After saving, reveal the key and map A/B to `after_75`/`after_225` (or A/B/C to base/75/225). The keys also include prompt, seed, checkpoint, and strength.

Once a scoring CSV is complete and the key has been revealed, summarize it with:

```powershell
.\venv\Scripts\python.exe experiments\sdxl-long-caption\quality-evaluation\summarize_human_scores.py experiments\sdxl-long-caption\quality-evaluation\results\<stage_directory>
```

The script writes paired score deltas, preference counts, per-strength prompt-cluster bootstrap intervals, and extended-detail compliance summaries. It does not treat non-significance as equivalence.

Do not infer an overall quality winner from image differences, pixel distances, or embeddings alone. Summarize paired score differences by prompt, seed, and strength; report medians and the share of pairs favoring each treatment. With four seeds per prompt, uncertainty remains large. A lack of clear score separation is “no detectable advantage at this screen's sensitivity,” not proof of equivalence.

## Interpretation limits

- The existing 75-vs-225 runs confound caption semantics and capacity. A clean causal test needs a new-build matched pair using identical extended caption files, with only capacity (75 vs 225) changed and matched initialization, image order, and random seed. That control is not present, and no training was started during the audit.
- Configs point to the same Illustrious XL path, but historical checkpoint hashes and exact application revisions were not recorded. Current checkpoint bytes are checked during evaluation; past byte identity cannot be reconstructed from run metadata alone.
- Training configs share a concept seed but record no global initialization seed, adapter checksum, realized sample order, or full stochastic state. Deterministic equivalence is unproven.
- Caption additions mostly describe appearance/body, clothing, and pose, with little scene, lighting, or style information. Some are redundant or inaccurate; tags were not manually verified image by image.
- Human scoring is required for style quality, prompt adherence, anatomy, leakage, and “melting.” Empty scoring CSVs are unscored evidence, not a result.
- The 225-token method costs more GPU memory and text-encoding work in principle; this evaluation records inference resource use only. It does not measure paired training throughput or peak training memory. Existing run logs show roughly 2.5–2.7 seconds per 1024px training step, without a repeated randomized benchmark.
