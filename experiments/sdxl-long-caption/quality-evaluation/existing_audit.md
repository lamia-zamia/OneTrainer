# Existing run and caption audit

## Caption dataset measurements

- Paired image-caption files: 523; image hashes identical across pairs: 523.
- CLIP-L and CLIP-G token IDs are identical for every original and extended caption. Counts are content tokens before BOS/EOS; OneTrainer chunks add the BOS/EOS positions.
- Original lengths: mean 69.01, median 73, p25 69.00, p75 75.00, p90 75.00, p95 75.00, p99 75.00, max 75.
- Extended lengths: mean 76.65, median 76, p25 71.00, p75 83.00, p90 95.00, p95 104.00, p99 127.34, max 150.
- CLIP-L and CLIP-G statistics are computed independently in token_length_statistics_by_tokenizer; their token IDs match for every caption, so their distributions are identical.
- Extended caption buckets: {"0-75": 249, "151-225": 0, "76-150": 274, ">225": 0}.
- Extended captions with content after token 75: 274/523 (52.39%).
- First 75 content-token ID sequences differ: 233/523 (44.55%). In every pair, old tags and their full token sequence are exact prefixes; differences come from appended tags filling unused positions in shorter captions.
- Appended tags: 1319 instances across 454 distinct tags. Since all originals fit one 75-token chunk, 274 extended captions require a second; none requires a third. No caption exceeds 225 tokens, so none is mechanically truncated there. This chunk count is not a measured time or memory increase; usefulness is not determined by length.

### Representative caption boundaries

#### 145_630

Counts: 72 original, 75 extended. Added tags: twintails.

Original: yoonalter, 1girl, red eyes, breasts, animal ears, hair over one eye, china dress, chinese clothes, solo, wet, rain, fox ears, dress, clothing cutout, cleavage cutout, gloves, fox girl, cleavage, short twintails, animal ear fluff, black dress, black gloves, looking at viewer, wet clothes, black hair

Extended: yoonalter, 1girl, red eyes, breasts, animal ears, hair over one eye, china dress, chinese clothes, solo, wet, rain, fox ears, dress, clothing cutout, cleavage cutout, gloves, fox girl, cleavage, short twintails, animal ear fluff, black dress, black gloves, looking at viewer, wet clothes, black hair, twintails

Tokens near position 75: 71=black</w>, 72=hair</w>, 73=,</w>, 74=twin, 75=tails</w>

#### G-_MlI8WMAAQdZB

Counts: 75 original, 78 extended. Added tags: simple background.

Original: yoonalter, 1boy, male focus, animal ears, gloves, dog boy, dog ears, multicolored hair, blush, tail, solo, dog tail, black gloves, fingerless gloves, blue eyes, white hair, white background, crop top, streaked hair, hair ornament, navel, hair over one eye, brown hair, sweat, upper body

Extended: yoonalter, 1boy, male focus, animal ears, gloves, dog boy, dog ears, multicolored hair, blush, tail, solo, dog tail, black gloves, fingerless gloves, blue eyes, white hair, white background, crop top, streaked hair, hair ornament, navel, hair over one eye, brown hair, sweat, upper body, simple background

Tokens near position 75: 71=,</w>, 72=sweat</w>, 73=,</w>, 74=upper</w>, 75=body</w>, 76=,</w>, 77=simple</w>, 78=background</w>

### Added tag positions

Positions are the 1-based first content-token position of each comma-delimited tag. Category names use a keyword heuristic, not a validated ontology.

Appended tag category counts by position: {"appearance": {"26-50": 3, "51-75": 72, "76-150": 389}, "clothing_material": {"51-75": 17, "76-150": 189}, "environment": {"51-75": 5, "76-150": 17}, "other": {"1-25": 1, "26-50": 2, "51-75": 72, "76-150": 446}, "pose_action_camera": {"51-75": 4, "76-150": 62}, "subject": {"26-50": 1, "51-75": 11, "76-150": 28}}

1131/1319 appended tag instances (85.75%) start after token position 75; the rest fill unused first-chunk positions.

All distinct appended tags are listed in appended_tag_inventory.csv with occurrence counts, tokenizer position bins, and up to five caption examples per tag.

Most common additions: symbol-shaped pupils (48), animal ears (25), heart (21), long hair (21), tail (20), underwear (19), cum on body (17), multicolored hair (17), navel (16), clothes (16), blush (16), open mouth (14), object insertion (13), gloves (13), multiple boys (13).

The additions mostly describe appearance/body, clothing, and pose. Environment tags are sparse and lighting/style tags are very rare. Some additions are redundant or hierarchical, such as a specific tag plus a broader tag. Individual tags were not verified against every image.

## Evaluation prompt token boundaries

The complete table, including exact CLIP-L/CLIP-G counts and late-detail offsets, is in evaluation_prompt_lengths.json. Every included prompt is encoded with the same 225-content-token inference setting for all LoRAs and base controls.

- P6: CLIP-L/CLIP-G 50/50 tokens; late markers: none.
- P7: CLIP-L/CLIP-G 22/22 tokens; late markers: none.
- specified_cafe: CLIP-L/CLIP-G 73/73 tokens; late markers: none.
- adult_man: CLIP-L/CLIP-G 68/68 tokens; late markers: none.
- adult_woman: CLIP-L/CLIP-G 60/60 tokens; late markers: none.
- material_matte_cotton: CLIP-L/CLIP-G 72/72 tokens; late markers: none.
- material_shiny_latex: CLIP-L/CLIP-G 72/72 tokens; late markers: none.
- lighting_flat_day: CLIP-L/CLIP-G 65/65 tokens; late markers: none.
- lighting_dramatic_night: CLIP-L/CLIP-G 68/68 tokens; late markers: none.
- solo_character: CLIP-L/CLIP-G 60/60 tokens; late markers: none.
- two_characters: CLIP-L/CLIP-G 73/73 tokens; late markers: none.
- bodytype_plus_size: CLIP-L/CLIP-G 66/66 tokens; late markers: none.
- vague_outdoor: CLIP-L/CLIP-G 21/21 tokens; late markers: none.
- late_after_75: CLIP-L/CLIP-G 159/159 tokens; late markers: wearing a bright yellow raincoat and purple scarf: start {'clip_l': 142, 'clip_g': 142}, end {'clip_l': 150, 'clip_g': 150}, fully within 225: True.
- late_after_150: CLIP-L/CLIP-G 223/223 tokens; late markers: wearing a deep green velvet coat with gold buttons and carrying a folded paper fan in her left hand: start {'clip_l': 205, 'clip_g': 205}, end {'clip_l': 223, 'clip_g': 223}, fully within 225: True.

## Training run audit

| Run | Build | Effective limit | Saved checkpoints | Caption source |
|---|---|---:|---|---|
| before_75 | original | 75 | 99, 199, 299, 399, 499 | D:/lora-datasets/Projects/alterkyon_new/concepts/test2 |
| before_225 | original | 75 | 99, 199, 299, 399, 499 | D:/lora-datasets/Projects/alterkyon_new - Copy/concepts/test2 |
| after_75 | long_clip | 75 | 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999 | D:/lora-datasets/Projects/alterkyon_new/concepts/test2 |
| after_225 | long_clip | 225 | 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999 | D:/lora-datasets/Projects/alterkyon_new - Copy/concepts/test2 |

All configs reference the same Illustrious XL checkpoint path. Its current SHA-256 was recomputed as 3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9, matching the earlier pretrained report; real-run snapshots do not record a historical hash or OneTrainer revision, so past byte identity is unverified. Inspected settings match on rank/alpha 32/32, Adafactor, constant UNet LR 0.0005, batch 1, accumulation 1, 1024 resolution, BF16 training, caption dropout 0, and masked training with prior-preservation weight 0. The TE1 LoRA learning rate is 0.00005.

The old-build runs save steps 99, 199, 299, 399, and 499. Both long_clip runs have checkpoints at 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, and 3999. Concept snapshots disable tag shuffling/dropout and image augmentation. They share a concept seed, but no global seed, initial LoRA checksum, or realized image order is recorded.

The after runs match on inspected training settings other than capacity and run-specific paths/names. They use different caption files, though. The extended file fills previously unused prefix positions for 233 captions, so after_75 versus after_225 changes both first-chunk conditioning and access to positions 76–150. The required same-build control (extended text at 75 tokens) is missing. before_225 uses the extended captions but is the old build and only 500 updates.

before_75 versus before_225 holds the old build and 75-token effective capacity constant, so it can describe the effect of the different caption text within the first 75 tokens. It cannot estimate the effect of access to later tokens. Neither old-run config records a global initialization seed or realized image order, and the runs stop at 500 updates.

### Training losses and timing

TensorBoard scalar values are in training_losses.csv; matched loss values for saved checkpoint steps are in training_loss_comparison.json. The saved 3999 backup is a periodic mid-epoch checkpoint, not the completed final model: the resumed event logs continue through step 4183 and the trainer then saves the final model. The final KOHYA file has no progress metadata.

The checkpoint-step-matched smooth-loss values are almost identical: differences from 399 through 3999 are approximately -0.000036 to +0.000098 (225-token minus 75-token). Raw per-batch losses vary more. These are training objectives, not held-out quality or style scores. The resumed exponential moving average restarts at step 1199.

Both 1024px long-run logs show about 2.5–2.7 seconds per step in recorded progress samples. This is close but not a repeated randomized timing benchmark, and captions differ. Historical peak memory and text-cache preparation timings were not recorded; caption preparation effort was not timed.

## Prior experiment evidence reviewed

The in-repository experiments/sdxl-long-caption/REPORT.md establishes tokenizer/model compatibility, gradients, optimizer updates, and visible output differences in a bounded synthetic one-image experiment (rank 4, 256px training; 512px 8-step image comparisons). It does not measure style quality or generalization for the real 1024px rank-32 style runs.

The existing ComfyUI manual grid manifest contains 128 outputs with P6/P7 and multiple checkpoints/strengths, but it uses DasiwaIllustriousAnime_epitaphecstasy.safetensors as the inference base rather than the Illustrious XL checkpoint path used for training. It has one seed, so it is unblinded and does not form a base-matched paired comparison. It is useful as exploratory imagery, not causal quality evidence.

## What the existing experiments can answer

The four real LoRAs can show behavior of these particular training outputs. They cannot attribute a quality difference specifically to access to tokens 76–225. That requires long_clip training on the same extended caption files at 75 and 225 tokens, with matched initialization and sample order. No additional training was started during this audit.
