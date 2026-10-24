import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from huggingface_hub import scan_cache_dir
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from modules.dataLoader.mixin.DataLoaderText2ImageMixin import sdxl_text_cache_dir
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.model.util.clip_chunk import (
    CLIP_CHUNK_TOKENS,
    CLIP_CONTENT_TOKENS,
    encode_clip_chunks,
    sdxl_chunk_attention_mask,
    sdxl_chunk_aware_unet_prediction,
    tokenize_clip_chunks,
    validate_sdxl_caption_token_limit,
)
from modules.model.util.clip_util import encode_clip
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType


def _cached_sdxl_tokenizers() -> tuple[CLIPTokenizer, CLIPTokenizer] | None:
    repo = next(
        (item for item in scan_cache_dir().repos if item.repo_id == "stabilityai/stable-diffusion-xl-base-1.0"),
        None,
    )
    if repo is None or not repo.revisions:
        return None

    revision = next(iter(repo.revisions))
    snapshot = Path(repo.repo_path) / "snapshots" / revision.commit_hash
    tokenizer_paths = (snapshot / "tokenizer", snapshot / "tokenizer_2")
    if not all((path / "vocab.json").is_file() and (path / "merges.txt").is_file() for path in tokenizer_paths):
        return None

    return tuple(
        CLIPTokenizer.from_pretrained(path, local_files_only=True)
        for path in tokenizer_paths
    )


def _caption_with_content_count(tokenizer: CLIPTokenizer, count: int, word: str = "photo") -> str:
    text = (word + " ") * count
    actual_count = len(tokenizer(text, add_special_tokens=False, truncation=True, max_length=count).input_ids)
    if actual_count != count:
        raise AssertionError(f"test token '{word}' must produce one token per word; got {actual_count} for {count}")
    return text


def _small_clip_encoders() -> tuple[CLIPTextModel, CLIPTextModelWithProjection]:
    encoder_1_config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=12,
        max_position_embeddings=77,
        bos_token_id=49406,
        eos_token_id=49407,
        pad_token_id=49407,
        attention_dropout=0.0,
    )
    encoder_2_config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1280,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=16,
        max_position_embeddings=77,
        bos_token_id=49406,
        eos_token_id=49407,
        pad_token_id=0,
        projection_dim=1280,
        attention_dropout=0.0,
    )
    return CLIPTextModel(encoder_1_config), CLIPTextModelWithProjection(encoder_2_config)


def _make_xl_model(tokenizers: tuple[CLIPTokenizer, CLIPTokenizer]) -> StableDiffusionXLModel:
    torch.manual_seed(321)
    model = StableDiffusionXLModel(ModelType.STABLE_DIFFUSION_XL_10_BASE)
    model.tokenizer_1, model.tokenizer_2 = tokenizers
    model.text_encoder_1, model.text_encoder_2 = _small_clip_encoders()
    model.text_encoder_1.eval()
    model.text_encoder_2.eval()
    return model


class SDXLLongCaptionUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizers = _cached_sdxl_tokenizers()

    def test_capacity_validation_and_cache_namespaces(self):
        for token_limit in (75, 150, 225):
            self.assertEqual(validate_sdxl_caption_token_limit(token_limit), token_limit)
        with self.assertRaisesRegex(ValueError, "75, 150, or 225"):
            validate_sdxl_caption_token_limit(100)
        with self.assertRaisesRegex(ValueError, "75, 150, or 225"):
            validate_sdxl_caption_token_limit(150.0)

        config = TrainConfig.default_values()
        config.model_type = ModelType.STABLE_DIFFUSION_XL_10_BASE
        config.cache_dir = "cache"
        config.sdxl_max_caption_tokens = 75
        self.assertEqual(sdxl_text_cache_dir(config), os.path.join("cache", "text"))
        config.sdxl_max_caption_tokens = 150
        self.assertEqual(sdxl_text_cache_dir(config), os.path.join("cache", "text", "sdxl-150"))
        config.sdxl_max_caption_tokens = 225
        self.assertEqual(sdxl_text_cache_dir(config), os.path.join("cache", "text", "sdxl-225"))

    def test_config_roundtrip_and_old_defaults(self):
        config = TrainConfig.default_values()
        config.model_type = ModelType.STABLE_DIFFUSION_XL_10_BASE
        config.sdxl_max_caption_tokens = 225
        restored = TrainConfig.default_values().from_dict(config.to_dict())
        self.assertEqual(restored.sdxl_max_caption_tokens, 225)

        old_data = config.to_dict()
        old_data.pop("sdxl_max_caption_tokens")
        restored_old = TrainConfig.default_values().from_dict(old_data)
        self.assertEqual(restored_old.sdxl_max_caption_tokens, 75)

        sample = SampleConfig.default_values(ModelType.STABLE_DIFFUSION_XL_10_BASE)
        sample.from_train_config(config)
        self.assertEqual(sample.sdxl_max_caption_tokens, 225)

        non_sdxl = TrainConfig.default_values()
        non_sdxl.model_type = ModelType.STABLE_DIFFUSION_15
        non_sdxl.cache_dir = "cache"
        non_sdxl.sdxl_max_caption_tokens = 225
        self.assertEqual(sdxl_text_cache_dir(non_sdxl), os.path.join("cache", "text"))

    def test_tokenization_boundaries_for_both_sdxl_tokenizers(self):
        if self.tokenizers is None:
            self.skipTest("cached SDXL CLIP-L/CLIP-G tokenizer assets are unavailable")

        token_counts = (1, 74, 75, 76, 150, 151, 225, 226)
        for tokenizer in self.tokenizers:
            for token_limit in (75, 150, 225):
                chunk_count = token_limit // CLIP_CONTENT_TOKENS
                for content_count in token_counts:
                    text = _caption_with_content_count(tokenizer, content_count)
                    expected_content = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=token_limit,
                    ).input_ids
                    tokens, mask = tokenize_clip_chunks(tokenizer, text, token_limit)

                    self.assertEqual(tokens.shape, (chunk_count * CLIP_CHUNK_TOKENS,))
                    self.assertEqual(mask.shape, tokens.shape)
                    chunks = tokens.reshape(chunk_count, CLIP_CHUNK_TOKENS)
                    masks = mask.reshape(chunk_count, CLIP_CHUNK_TOKENS)
                    reconstructed_content = []
                    for chunk_index, (chunk, chunk_mask) in enumerate(zip(chunks, masks)):
                        consumed = min(max(content_count - chunk_index * CLIP_CONTENT_TOKENS, 0), CLIP_CONTENT_TOKENS)
                        self.assertEqual(chunk[0].item(), tokenizer.bos_token_id)
                        self.assertEqual(chunk[consumed + 1].item(), tokenizer.eos_token_id)
                        self.assertTrue(torch.all(chunk_mask[:consumed + 2] == 1))
                        self.assertTrue(torch.all(chunk_mask[consumed + 2:] == 0))
                        self.assertTrue(torch.all(chunk[consumed + 2:] == tokenizer.pad_token_id))
                        reconstructed_content.extend(chunk[1:consumed + 1].tolist())

                    self.assertEqual(reconstructed_content, expected_content[:token_limit])

                    if token_limit == 75:
                        legacy = tokenizer(
                            text,
                            padding="max_length",
                            truncation=True,
                            max_length=77,
                            return_tensors="pt",
                        )
                        self.assertTrue(torch.equal(tokens, legacy.input_ids[0]))
                        self.assertTrue(torch.equal(mask, legacy.attention_mask[0]))

    def test_fixed_capacity_batch_and_225_prompt_encoding(self):
        if self.tokenizers is None:
            self.skipTest("cached SDXL CLIP-L/CLIP-G tokenizer assets are unavailable")

        tokenizer_1, tokenizer_2 = self.tokenizers
        prompts = [
            _caption_with_content_count(tokenizer_1, 1),
            _caption_with_content_count(tokenizer_1, 226),
        ]
        batch = torch.stack([tokenize_clip_chunks(tokenizer_1, prompt, 225)[0] for prompt in prompts])
        self.assertEqual(batch.shape, (2, 231))

        model = _make_xl_model(self.tokenizers)
        positive = _caption_with_content_count(tokenizer_1, 225)
        negative = _caption_with_content_count(tokenizer_1, 225, "dog")
        medium_conditioning = model.combine_text_encoder_output(
            *model.encode_text(text=positive, train_device=torch.device("cpu"), max_caption_tokens=150)
        )
        positive_conditioning = model.combine_text_encoder_output(
            *model.encode_text(text=positive, train_device=torch.device("cpu"), max_caption_tokens=225)
        )
        negative_conditioning = model.combine_text_encoder_output(
            *model.encode_text(text=negative, train_device=torch.device("cpu"), max_caption_tokens=225)
        )
        self.assertEqual(medium_conditioning[0].shape, (1, 154, 2048))
        self.assertEqual(medium_conditioning[1].shape, (1, 1280))
        self.assertEqual(positive_conditioning[0].shape, (1, 231, 2048))
        self.assertEqual(positive_conditioning[1].shape, (1, 1280))
        self.assertEqual(negative_conditioning[0].shape, (1, 231, 2048))
        self.assertEqual(negative_conditioning[1].shape, (1, 1280))
        self.assertEqual(torch.cat([negative_conditioning[0], positive_conditioning[0]]).shape, (2, 231, 2048))
        self.assertEqual(torch.cat([negative_conditioning[1], positive_conditioning[1]]).shape, (2, 1280))

    def test_chunk_attention_mask_preserves_short_and_aligned_conditioning(self):
        if self.tokenizers is None:
            self.skipTest("cached SDXL CLIP-L/CLIP-G tokenizer assets are unavailable")

        tokenizer_1, tokenizer_2 = self.tokenizers
        captions = (
            ("", 1),
            (_caption_with_content_count(tokenizer_1, 40), 1),
            (_caption_with_content_count(tokenizer_1, 75), 1),
            (_caption_with_content_count(tokenizer_1, 76), 2),
            (_caption_with_content_count(tokenizer_1, 150), 2),
            (_caption_with_content_count(tokenizer_1, 151), 3),
            (_caption_with_content_count(tokenizer_1, 225), 3),
        )
        for caption, expected_chunks in captions:
            tokens_1, _ = tokenize_clip_chunks(tokenizer_1, caption, 225)
            tokens_2, _ = tokenize_clip_chunks(tokenizer_2, caption, 225)
            mask = sdxl_chunk_attention_mask(
                tokenizer_1, tokens_1, tokenizer_2, tokens_2, 225
            )
            self.assertEqual(mask.shape, (1, 231))
            self.assertTrue(torch.all(mask[0, :expected_chunks * 77] == 1))
            self.assertTrue(torch.all(mask[0, expected_chunks * 77:] == 0))

        short = _caption_with_content_count(tokenizer_1, 40)
        short_1, _ = tokenize_clip_chunks(tokenizer_1, short, 225)
        short_2, _ = tokenize_clip_chunks(tokenizer_2, short, 225)
        legacy_1, _ = tokenize_clip_chunks(tokenizer_1, short, 75)
        legacy_2, _ = tokenize_clip_chunks(tokenizer_2, short, 75)
        self.assertTrue(torch.equal(short_1[:77], legacy_1))
        self.assertTrue(torch.equal(short_2[:77], legacy_2))

        aligned_l, _ = tokenize_clip_chunks(
            tokenizer_1, _caption_with_content_count(tokenizer_1, 151), 225
        )
        aligned_g, _ = tokenize_clip_chunks(
            tokenizer_2, _caption_with_content_count(tokenizer_2, 151), 225
        )
        aligned_g[2 * 77 + 1:] = tokenizer_2.pad_token_id
        aligned_g[2 * 77] = tokenizer_2.bos_token_id
        aligned_g[2 * 77 + 1] = tokenizer_2.eos_token_id
        aligned_mask = sdxl_chunk_attention_mask(
            tokenizer_1, aligned_l, tokenizer_2, aligned_g, 225
        )
        self.assertTrue(torch.all(aligned_mask[0, :2 * 77] == 1))
        self.assertTrue(torch.all(aligned_mask[0, 2 * 77:] == 0))

        short = _caption_with_content_count(tokenizer_1, 40)
        long = _caption_with_content_count(tokenizer_1, 151)
        batch_l = torch.stack([tokenize_clip_chunks(tokenizer_1, prompt, 225)[0] for prompt in (short, long)])
        batch_g = torch.stack([tokenize_clip_chunks(tokenizer_2, prompt, 225)[0] for prompt in (short, long)])
        batch_mask = sdxl_chunk_attention_mask(tokenizer_1, batch_l, tokenizer_2, batch_g, 225)
        self.assertEqual(batch_mask.shape, (2, 231))
        self.assertTrue(torch.all(batch_mask[0, :77] == 1))
        self.assertTrue(torch.all(batch_mask[0, 77:] == 0))
        self.assertTrue(torch.all(batch_mask[1, :3 * 77] == 1))
        self.assertTrue(torch.all(batch_mask[1, 3 * 77:] == 0))

        # Classifier-free guidance concatenates negative then positive conditioning and masks.
        cfg_mask = torch.cat([batch_mask[:1], batch_mask[1:]], dim=0)
        self.assertEqual(cfg_mask.shape, (2, 231))
        self.assertTrue(torch.equal(cfg_mask[0], batch_mask[0]))
        self.assertTrue(torch.equal(cfg_mask[1], batch_mask[1]))

    def test_chunk_aware_unet_groups_mixed_lengths_and_restores_order(self):
        class RecordingUNet:
            def __init__(self):
                self.sequence_lengths = []

            def __call__(self, *, sample, timestep, encoder_hidden_states, added_cond_kwargs):
                self.sequence_lengths.append(encoder_hidden_states.shape[1])
                pooled = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
                time_condition = added_cond_kwargs["text_embeds"].mean(dim=1).view(-1, 1, 1, 1)
                return type("Output", (), {"sample": sample + pooled + time_condition})()

        batch_size = 3
        hidden = torch.randn(batch_size, 231, 4, requires_grad=True)
        sample = torch.zeros(batch_size, 2, 2, 2)
        timestep = torch.tensor([5, 7, 11])
        pooled = torch.randn(batch_size, 6)
        mask = torch.cat([
            torch.cat([torch.ones(1, 77), torch.zeros(1, 154)], dim=1),
            torch.ones(1, 231),
            torch.cat([torch.ones(1, 154), torch.zeros(1, 77)], dim=1),
        ]).long()
        unet = RecordingUNet()
        result = sdxl_chunk_aware_unet_prediction(
            unet,
            sample,
            timestep,
            hidden,
            {"text_embeds": pooled},
            mask,
        )
        self.assertEqual(sorted(unet.sequence_lengths), [77, 154, 231])
        self.assertEqual(result.shape, sample.shape)
        result.sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertGreater(torch.linalg.vector_norm(hidden.grad).item(), 0)

        short_hidden = torch.randn(2, 231, 4)
        short_sample = torch.randn(2, 2, 2, 2)
        short_pooled = torch.randn(2, 6)
        short_mask = torch.cat([
            torch.cat([torch.ones(1, 77), torch.zeros(1, 154)], dim=1),
            torch.cat([torch.ones(1, 77), torch.zeros(1, 154)], dim=1),
        ]).long()
        extended_unet = RecordingUNet()
        extended_output = sdxl_chunk_aware_unet_prediction(
            extended_unet,
            short_sample,
            torch.tensor(5),
            short_hidden,
            {"text_embeds": short_pooled},
            short_mask,
        )
        legacy_unet = RecordingUNet()
        legacy_output = sdxl_chunk_aware_unet_prediction(
            legacy_unet,
            short_sample,
            torch.tensor(5),
            short_hidden[:, :77].contiguous(),
            {"text_embeds": short_pooled},
            None,
        )
        self.assertEqual(extended_unet.sequence_lengths, [77])
        torch.testing.assert_close(extended_output, legacy_output, rtol=0, atol=0)

    def test_short_caption_hidden_states_and_empty_prompt_mask(self):
        if self.tokenizers is None:
            self.skipTest("cached SDXL CLIP-L/CLIP-G tokenizer assets are unavailable")

        model = _make_xl_model(self.tokenizers)
        short = _caption_with_content_count(self.tokenizers[0], 40)
        legacy = model.encode_text(text=short, train_device=torch.device("cpu"))
        extended = model.encode_text(
            text=short,
            train_device=torch.device("cpu"),
            max_caption_tokens=225,
            return_chunk_attention_mask=True,
        )
        for legacy_output, extended_output in zip(legacy, extended[:3]):
            torch.testing.assert_close(legacy_output, extended_output[..., :legacy_output.shape[-2], :]
                                       if legacy_output.ndim == 3 else extended_output)
        self.assertEqual(extended[3].shape, (1, 231))
        self.assertTrue(torch.all(extended[3][:, :77] == 1))
        self.assertTrue(torch.all(extended[3][:, 77:] == 0))

        empty = model.encode_text(
            text="",
            train_device=torch.device("cpu"),
            max_caption_tokens=225,
            return_chunk_attention_mask=True,
        )
        self.assertTrue(torch.all(empty[3][:, :77] == 1))
        self.assertTrue(torch.all(empty[3][:, 77:] == 0))

    def test_75_token_conditioning_matches_legacy_encoder_calls(self):
        if self.tokenizers is None:
            self.skipTest("cached SDXL CLIP-L/CLIP-G tokenizer assets are unavailable")

        torch.manual_seed(123)
        encoder_1, encoder_2 = _small_clip_encoders()
        prompt = _caption_with_content_count(self.tokenizers[0], 74)
        tokens_1, _ = tokenize_clip_chunks(self.tokenizers[0], prompt, 75)
        tokens_2, _ = tokenize_clip_chunks(self.tokenizers[1], prompt, 75)

        legacy_1 = encode_clip(
            encoder_1, tokens_1.unsqueeze(0), default_layer=-2, add_pooled_output=False,
            use_attention_mask=False, add_layer_norm=False,
        )[0]
        chunked_1, _ = encode_clip_chunks(encoder_1, tokens_1, hidden_state_output_index=-2)
        legacy_2 = encode_clip(
            encoder_2, tokens_2.unsqueeze(0), default_layer=-2, add_pooled_output=True,
            use_attention_mask=False, add_layer_norm=False,
        )
        chunked_2, chunked_pool = encode_clip_chunks(
            encoder_2, tokens_2, hidden_state_output_index=-2, add_pooled_output=True,
        )
        torch.testing.assert_close(chunked_1, legacy_1, rtol=0, atol=0)
        torch.testing.assert_close(chunked_2, legacy_2[0], rtol=0, atol=0)
        torch.testing.assert_close(chunked_pool, legacy_2[1], rtol=0, atol=0)

    def test_trainable_text_encoder_retains_gradient_flow(self):
        if self.tokenizers is None:
            self.skipTest("cached SDXL CLIP-L/CLIP-G tokenizer assets are unavailable")

        model = _make_xl_model(self.tokenizers)
        tokens_1, _ = tokenize_clip_chunks(self.tokenizers[0], _caption_with_content_count(self.tokenizers[0], 151), 225)
        tokens_2, _ = tokenize_clip_chunks(self.tokenizers[1], _caption_with_content_count(self.tokenizers[1], 151), 225)
        output = model.encode_text(
            train_device=torch.device("cpu"),
            tokens_1=tokens_1.unsqueeze(0),
            tokens_2=tokens_2.unsqueeze(0),
            max_caption_tokens=225,
            return_chunk_attention_mask=True,
        )
        combined, pooled = model.combine_text_encoder_output(*output[:3])
        self.assertTrue(torch.all(output[3] == 1))
        (combined.square().mean() + pooled.square().mean()).backward()
        gradient = model.text_encoder_1.text_model.encoder.layers[0].self_attn.q_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(torch.linalg.vector_norm(gradient).item(), 0)

        with self.assertRaisesRegex(ValueError, "expected 77"):
            model.encode_text(
                train_device=torch.device("cpu"),
                tokens_1=tokens_1.unsqueeze(0),
                tokens_2=tokens_2.unsqueeze(0),
                max_caption_tokens=75,
            )

    def test_third_chunk_changes_real_sdxl_unet_lora_gradients(self):
        if self.tokenizers is None:
            self.skipTest("cached SDXL CLIP-L/CLIP-G tokenizer assets are unavailable")

        torch.manual_seed(987)
        model = _make_xl_model(self.tokenizers)
        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        tokenizer_1, tokenizer_2 = self.tokenizers
        caption_a = _caption_with_content_count(tokenizer_1, 225)
        caption_b = _caption_with_content_count(tokenizer_1, 150) + ("dog " * 75)

        def encode(prompt: str, token_limit: int):
            values = model.encode_text(
                text=prompt,
                train_device=torch.device("cpu"),
                max_caption_tokens=token_limit,
            )
            return model.combine_text_encoder_output(*values)

        tokens_a_l, _ = tokenize_clip_chunks(tokenizer_1, caption_a, 225)
        tokens_b_l, _ = tokenize_clip_chunks(tokenizer_1, caption_b, 225)
        tokens_a_g, _ = tokenize_clip_chunks(tokenizer_2, caption_a, 225)
        tokens_b_g, _ = tokenize_clip_chunks(tokenizer_2, caption_b, 225)
        self.assertTrue(torch.equal(tokens_a_l[:154], tokens_b_l[:154]))
        self.assertTrue(torch.equal(tokens_a_g[:154], tokens_b_g[:154]))
        self.assertFalse(torch.equal(tokens_a_l[154:], tokens_b_l[154:]))
        self.assertFalse(torch.equal(tokens_a_g[154:], tokens_b_g[154:]))

        conditioning_a, pooled_a = encode(caption_a, 225)
        conditioning_b, pooled_b = encode(caption_b, 225)
        self.assertTrue(torch.equal(conditioning_a[:, :154], conditioning_b[:, :154]))
        self.assertFalse(torch.equal(conditioning_a[:, 154:], conditioning_b[:, 154:]))
        self.assertTrue(torch.equal(pooled_a, pooled_b))

        legacy_a, legacy_pooled_a = encode(caption_a, 75)
        legacy_b, legacy_pooled_b = encode(caption_b, 75)
        self.assertTrue(torch.equal(legacy_a, legacy_b))
        self.assertTrue(torch.equal(legacy_pooled_a, legacy_pooled_b))

        unet = UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(32, 64),
            layers_per_block=1,
            cross_attention_dim=2048,
            attention_head_dim=8,
            norm_num_groups=8,
            addition_embed_type="text_time",
            addition_time_embed_dim=256,
            projection_class_embeddings_input_dim=2816,
        ).eval()
        for parameter in unet.parameters():
            parameter.requires_grad_(False)

        config = TrainConfig.default_values()
        config.lora_rank = 2
        config.lora_alpha = 2.0
        config.train_device = "cpu"
        target_layer = "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k"
        lora = LoRAModuleWrapper(unet, "unet", config, module_filter=[target_layer])
        lora.set_dropout(0.0)
        lora.requires_grad_(True)
        lora.hook_to_module()
        adapter = lora.lora_modules[target_layer]
        initial_down = adapter.lora_down.weight.detach().clone()
        initial_up = adapter.lora_up.weight.detach().clone()
        latent = torch.randn(1, 4, 8, 8)
        target = torch.randn_like(latent)
        timestep = torch.tensor([123])
        time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=conditioning_a.dtype)

        def lora_gradient(conditioning, pooled):
            for parameter in lora.parameters():
                parameter.grad = None
            prediction = unet(
                sample=latent,
                timestep=timestep,
                encoder_hidden_states=conditioning,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
            ).sample
            F.mse_loss(prediction, target).backward()
            return adapter.lora_down.weight.grad.detach().clone(), adapter.lora_up.weight.grad.detach().clone()

        gradient_a = lora_gradient(conditioning_a, pooled_a)
        identical_gradient = lora_gradient(conditioning_a, pooled_a)
        gradient_b = lora_gradient(conditioning_b, pooled_b)
        legacy_gradient_a = lora_gradient(legacy_a, legacy_pooled_a)
        legacy_gradient_b = lora_gradient(legacy_b, legacy_pooled_b)

        torch.testing.assert_close(gradient_a[1], identical_gradient[1], rtol=0, atol=0)
        torch.testing.assert_close(legacy_gradient_a[1], legacy_gradient_b[1], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(gradient_a[1]).all())
        self.assertTrue(torch.isfinite(gradient_b[1]).all())
        self.assertGreater(torch.linalg.vector_norm(gradient_a[1] - gradient_b[1]).item(), 1e-8)
        self.assertTrue(torch.equal(initial_down, adapter.lora_down.weight.detach()))
        self.assertTrue(torch.equal(initial_up, adapter.lora_up.weight.detach()))
        lora.remove_hook_from_module()


if __name__ == "__main__":
    unittest.main()
