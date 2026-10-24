import threading

import torch
from torch import Tensor
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


CLIP_CHUNK_TOKENS = 77
CLIP_CONTENT_TOKENS = 75
SDXL_CAPTION_TOKEN_LIMITS = (75, 150, 225)
_TOKENIZER_LOCK_INIT = threading.Lock()


def validate_sdxl_caption_token_limit(token_limit: int) -> int:
    if type(token_limit) is not int or token_limit not in SDXL_CAPTION_TOKEN_LIMITS:
        raise ValueError(
            f"SDXL caption token limit must be 75, 150, or 225, got {token_limit}"
        )
    return token_limit


def _tokenizer_lock(tokenizer: CLIPTokenizer) -> threading.Lock:
    lock = getattr(tokenizer, "_mgds_tokenizer_lock", None)
    if lock is None:
        with _TOKENIZER_LOCK_INIT:
            lock = getattr(tokenizer, "_mgds_tokenizer_lock", None)
            if lock is None:
                lock = threading.Lock()
                tokenizer._mgds_tokenizer_lock = lock
    return lock


def tokenize_clip_chunks(
        tokenizer: CLIPTokenizer,
        text: str,
        token_limit: int,
) -> tuple[Tensor, Tensor]:
    token_limit = validate_sdxl_caption_token_limit(token_limit)

    with _tokenizer_lock(tokenizer):
        if token_limit == 75:
            tokenizer_output = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=CLIP_CHUNK_TOKENS,
                return_tensors="pt",
            )
            return tokenizer_output.input_ids[0], tokenizer_output.attention_mask[0]

        content_ids = tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=token_limit,
            return_tensors="pt",
        ).input_ids[0].tolist()

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if bos_token_id is None or eos_token_id is None or pad_token_id is None:
        raise ValueError("SDXL CLIP tokenizers must define BOS, EOS, and PAD token IDs")

    token_ids = []
    attention_mask = []
    for chunk_index in range(token_limit // CLIP_CONTENT_TOKENS):
        chunk_content = content_ids[
            chunk_index * CLIP_CONTENT_TOKENS:(chunk_index + 1) * CLIP_CONTENT_TOKENS
        ]
        chunk = [bos_token_id, *chunk_content, eos_token_id]
        chunk_mask = [1] * len(chunk)
        padding_length = CLIP_CHUNK_TOKENS - len(chunk)
        chunk.extend([pad_token_id] * padding_length)
        chunk_mask.extend([0] * padding_length)
        token_ids.extend(chunk)
        attention_mask.extend(chunk_mask)

    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


def encode_clip_chunks(
        text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
        tokens: Tensor,
        hidden_state_output_index: int,
        add_pooled_output: bool = False,
) -> tuple[Tensor, Tensor | None]:
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2 or tokens.shape[-1] == 0 or tokens.shape[-1] % CLIP_CHUNK_TOKENS != 0:
        raise ValueError(f"Chunked SDXL tokens must have shape [batch, N * {CLIP_CHUNK_TOKENS}]")

    batch_size, sequence_length = tokens.shape
    chunk_count = sequence_length // CLIP_CHUNK_TOKENS
    chunk_tokens = tokens.reshape(batch_size * chunk_count, CLIP_CHUNK_TOKENS)
    encoder_output = text_encoder(
        chunk_tokens,
        attention_mask=None,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden_state = encoder_output.hidden_states[hidden_state_output_index]
    hidden_state = hidden_state.reshape(batch_size, sequence_length, hidden_state.shape[-1])

    pooled_state = None
    if add_pooled_output:
        pooled_state = getattr(encoder_output, "text_embeds", None)
        if pooled_state is None:
            pooled_state = getattr(encoder_output, "pooler_output", None)
        if pooled_state is None:
            raise ValueError("The SDXL pooled text encoder output is missing")
        pooled_state = pooled_state.reshape(batch_size, chunk_count, -1)[:, 0]

    return hidden_state, pooled_state
