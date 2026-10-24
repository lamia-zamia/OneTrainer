# SDXL long captions

`sdxl_max_caption_tokens` sets the SDXL training and preview capacity to 75, 150, or 225 content tokens. Each 75-token chunk is encoded as a separate 77-position CLIP input with its own BOS and EOS tokens. CLIP-L and CLIP-G tokenize independently; their sequence outputs are concatenated on the feature dimension. SDXL pooled conditioning uses the first CLIP-G chunk, matching ComfyUI.

The default is 75, which keeps the existing tokenization and text-encoder path. Extended capacities pad every prompt to the selected number of chunks so training batches and classifier-free guidance prompts have consistent shapes. The generated padding mask is retained by tokenization, while CLIP attention remains unmasked as in OneTrainer's current SDXL path and ComfyUI's default.

OneTrainer passes prompt strings directly to each tokenizer and does not parse weighted-prompt syntax. Existing textual and output embedding handling remains in place; prompt weights are not part of this feature.

The setting is available in the SDXL training tab and training configuration. Headless training can set it with `--config-value sdxl_max_caption_tokens=150`. Standalone sampling accepts `--sdxl-max-caption-tokens 225`; training previews inherit the training setting. Extended text-embedding caches use a capacity-specific directory, while the existing 75-token cache remains compatible.

The SDXL UNet cross-attention sequence grows from 77 to 154 or 231 positions. Text-encoder work and cached sequence embeddings grow roughly with chunk count; UNet cross-attention activation and compute also grow with the longer context.
