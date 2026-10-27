# SDXL long-caption quality evaluation: findings

**Status:** the existing-run and caption audit, 1024px screen, and checkpoint progression generation are complete. Human scoring remains pending. This report separates measured facts from conclusions that require visual review.

## Executive finding so far

The existing experiments show both caption workflows can learn the style, but they do not establish a consistent quality advantage for 225-token training. The checkpoint-matched training losses are nearly the same. The comparison is confounded because the 225-token runs use extended captions whose content also changes the first 75 token positions for 44.6% of images. The required same-build, same-caption 75-token control does not exist.

Therefore, no observed difference can currently be attributed specifically to access to tokens 76–225. The blinded screen contains 360 LoRA images (180 complete A/B pairs) and eight base-model controls. The checkpoint progression contains 192 LoRA images (96 complete A/B pairs). The human scoring CSVs are intentionally unfilled; empty scoring fields are not quality measurements.

## Measured facts

### Captions

- There are 523 paired caption/image records; all paired image SHA-256 hashes match.
- CLIP-L and CLIP-G were tokenized independently and produce the same content token IDs for these captions. Counts exclude BOS/EOS.
- Original captions: mean 69.01, median 73, p25 69, p75/p90/p95/p99 75, maximum 75 tokens.
- Extended captions: mean 76.65, median 76, p25 71, p75 83, p90 95, p95 104, p99 127.34, maximum 150 tokens.
- Token buckets are 523 original / 249 extended captions at 0–75 tokens, 274 extended at 76–150, and zero in both sets at 151–225 or over 225.
- 274/523 (52.39%) extended captions contain tokens after position 75; 1,131/1,319 appended tag instances (85.75%) begin after position 75. None exceeds 225, so there is no mechanically truncated caption at the 225-token limit.
- No caption text is truncated by a 225-token limit because the longest caption has 150 tokens. Whether any omitted/added tag would have been useful is a semantic judgment; tag accuracy and usefulness were not manually verified image by image.
- Because every original caption fits within one 75-token chunk, those 274 extended captions require a second chunk; none requires a third. This is a workload count (about 0.52 extra chunks per caption on average), not a measured time or memory increase.
- All 523 original tag lists are exact prefixes of the extended tag lists. The extensions add 1,319 tag instances across 454 distinct labels. About 44.6% of caption pairs have different first-75-token sequences because appended tags fill positions that were unused in shorter originals.
- Added labels mostly concern appearance/body, clothing, and pose. Scene/environment and lighting/style additions are rare. Frequent additions include `symbol-shaped pupils` (48), `animal ears` (25), `heart` (21), `long hair` (21), and `tail` (20); the set also adds sexual-content labels such as `cum on body` (17), `object insertion` (13), and `sex toy` (10). These are semantic content tags, not style descriptions. The appended-tag inventory shows every label's frequency, position bucket, and example captions. Some labels are redundant or hierarchical; for example, `twintails` follows `short twintails`, and `simple background` follows `white background`. The tagger output has not been verified against every image and may include inaccurate, ambiguous, or conflicting labels.

### Existing runs

| Run | Build | Effective capacity | Available saved steps |
|---|---|---:|---|
| `before_75` | original | 75 | 99, 199, 299, 399, 499 |
| `before_225` | original | 75 | 99, 199, 299, 399, 499 |
| `after_75` | `long_clip` | 75 | 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999 |
| `after_225` | `long_clip` | 225 | 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999 |

All configs point to the same Illustrious XL checkpoint path. Its current SHA-256 was recomputed as `3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9`, matching the earlier pretrained report. The historical training runs did not record a checkpoint hash. Inspected configs match on rank/alpha (32/32), optimizer (Adafactor), UNet learning rate (0.0005), TE1 learning rate (0.00005), constant schedule, batch/accumulation (1/1), resolution (1024), BF16 training, caption dropout (0), and prior-preservation weight (0). Tag shuffle/dropout and listed image augmentations are disabled. They share a concept seed, but no global seed, initial adapter hash, realized image order, or recorded OneTrainer revision is available. Deterministic equivalence is unproven.

The long-clip logs continue through event step 4183 after the periodic checkpoint named 3999; that backup is not the final training state. Training loss at matched saved steps is nearly identical: smoothed 225-minus-75 differences range from approximately −0.000036 to +0.000098. Loss is a training objective, not a held-out quality score. The resumed EMA restarts at step 1199.

`before_75` versus `before_225` holds the old build and its 75-token capacity constant, so it can describe the effect of the different caption text only within the first 75 positions. It cannot test access to later tokens; it also has only 500 updates and no recorded global initialization seed or realized image order.

The recorded training speed is about 2.5–2.7 seconds per 1024px step for both long runs, but this was not a randomized repeated benchmark and the caption contents differ. Historical peak training memory, caption-cache preparation time, and caption preparation labor were not recorded.

### Earlier quality evidence

The in-repository technical report under `experiments/sdxl-long-caption/REPORT.md` shows compatibility, gradients, optimizer updates, and visibly different outputs in a bounded synthetic run (rank 4, 256px training; 512px 8-step output comparisons). It does not measure style quality or generalization for the real rank-32 1024px style runs. The old ComfyUI manual grid has 128 outputs including P6/P7, but uses a different inference base checkpoint, one seed, and no blinding; it is exploratory imagery rather than a valid matched quality comparison.

## Quality comparison and progression

See `results/screen_full/` for the 1024px paired outputs, per-image metadata, randomized contact sheets, and blank scoring CSVs. The screen contains 360 LoRA images (15 prompts × 4 seeds × 3 strengths × 2 treatments), eight base-model long-prompt controls, 180 complete A/B pairs, and zero unpaired treatment records. Both LoRAs use the same base checkpoint, sampler, CFG, resolution, negative prompt, 225-token inference capacity, and paired seeds at strengths 1.0, 1.5, and 2.0. The prompt suite includes explicit, vague, gender, material, lighting, subject-count, body-type, and late-detail cases. The progression comparison in `results/progression_full/` contains 192 LoRA images at matched checkpoints 1199, 2399, 3199, and 3999 on P6/P7 at strengths 1, 1.5, and 2. It has 96 complete A/B pairs, zero unpaired records, 24 randomized contact sheets, and a blank scoring CSV.

The late-inference marker starts at tokens 142–150 and 205–223 for the two prompts, matching across CLIP-L/CLIP-G; both complete details fall within the 225-token inference limit. Base-model controls use the identical inference prompts and seeds.

At 1024px and 28 steps, screen inference took a median 30.57 seconds per image for `after_75` and 30.78 seconds for `after_225`. The median paired time difference was +0.21 seconds, with a 10th–90th percentile range from −2.18 to +9.47 seconds. All `after_75` samples were generated first, so order, warm-up, and system variation confound treatment timing; the 225-minus-75 latency difference is not causal evidence. Peak CUDA allocation averaged about 5,933 MiB in both groups (reserved memory about 6,445 MiB in both); paired median allocated-memory difference was 0 MiB. These are inference-harness measurements at identical 225-token inference capacity, not training throughput, training peak memory, or cache-preparation costs. One `after_225` sample took 115.6 seconds, showing a long tail; the screen median is more representative of the typical observed sample.

The blind sheets should be scored before opening the key. Subjective quality claims remain pending until that scoring is completed. See `README.md` for the rubric and interpretation instructions. `resource_summary.json` records per-image inference time and peak CUDA memory; it does not measure training costs.

For the progression images, the recorded generation medians were 30.45 seconds/image for `after_75` and 35.65 seconds/image for `after_225`; the paired median delta was +5.20 seconds (10th–90th percentile −0.07 to +6.77 seconds). Peak allocated memory medians were both 5,962 MiB and reserved memory medians were both 6,502 MiB. Every `after_75` image was generated before every `after_225` image, so timing is confounded by order, warm-up, and checkpoint-loading effects; these are descriptive inference-harness values, not evidence that 225-token training changes runtime. The screen timing and inference-memory results above have the same treatment-order limitation. Neither stage measures training peak memory, caption caching, or captioning labor.

## Answers to the evaluation questions

These are evidence-bounded answers. They are not a claim of equivalence.

1. **Does 225-token captioning produce measurably better style LoRAs?** Not established. Existing visual review found no obvious consistent improvement; standardized blinded scoring is pending.
2. **Does it improve disentanglement or prompt adherence?** Not established by existing runs. The paired screen includes counterfactual prompts, but human scores are needed.
3. **Does it reduce unwanted content leakage?** Not established. Added tags include content-specific appearance/clothing/action labels and may change learned priors; the current runs confound text changes with capacity.
4. **Does it affect style-learning speed or onset of melting?** Matched loss curves are very close, but loss cannot identify either outcome. Matched-checkpoint progression images now exist; blinded human review is still needed to judge style development, prompt adherence, anatomy degradation, and melting.
5. **Are differences attributable specifically to tokens beyond 75?** No. The matched long-clip/extended-caption/75-token control is missing. The `after_75` versus `after_225` comparison also changes first-chunk text on 233 images.
6. **Does it meaningfully increase training time, memory, or captioning effort?** 52.4% of extended examples require a second text chunk, but the actual time, peak training memory, cache preparation, and manual caption effort were not measured. Existing logs show similar approximate step times, without isolating capacity. Inference resource summaries are not a substitute for training benchmarks.
7. **Should 225-token captions become the default?** Current evidence supports keeping them optional, not making them the default. This recommendation can change if blind results show a repeatable advantage and a matched causal training pair confirms that tokens 76–225 produced it.

## Strongest and weakest supported conclusions

- **Strongest:** the extended dataset is append-only at the string/tag level, but it changes first-chunk token content for 233/523 images and gives later tokens to 274/523 images. The current real-run comparison therefore does not isolate the capacity effect.
- **Moderate:** the particular 75- and 225-caption long runs have nearly identical recorded training loss and similar approximate logged step speeds.
- **Weakest / unresolved:** which LoRA has better subjective style, content disentanglement, prompt compliance, anatomy, or checkpoint stability. These require blinded human scoring of the generated outputs.
