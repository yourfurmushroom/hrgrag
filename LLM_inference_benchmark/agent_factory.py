# agent_factory.py
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM
from LLM_stratiges import HarmonyHFStrategy, GenericHFStrategy


def needs_harmony(model_id: str) -> bool:
    # ✅ 你可以依你的命名規則調整
    # 只要是 openai/gpt-oss 類就走 Harmony
    model_id_lower = model_id.lower()
    return model_id_lower.startswith("openai/") or "gpt-oss" in model_id_lower


def build_llm_strategy(
    model_id: str,
    max_new_tokens: int = 1024,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    if needs_harmony(model_id):
        return HarmonyHFStrategy(model=model, max_new_tokens=max_new_tokens)

    return GenericHFStrategy(model=model, tokenizer_name_or_path=model_id, max_new_tokens=max_new_tokens)
