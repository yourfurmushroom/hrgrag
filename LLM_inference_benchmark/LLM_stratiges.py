# llm_strategies.py
from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer


class LLMStrategy(ABC):
    def __init__(self, model):
        self.model = model
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def _input_device(self):
        """Use the embedding/input device for dispatched HF models."""
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            pass
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @abstractmethod
    async def inference(self, developer_instruction: str, user_content: str) -> str:
        """Return raw model text (may include special tokens)."""
        raise NotImplementedError


class HarmonyHFStrategy(LLMStrategy):
    """
    適用：openai/gpt-oss-* 這種需要 Harmony prompt 的 HF 模型
    """
    def __init__(self, model, max_new_tokens: int = 2048):
        super().__init__(model)
        self.max_new_tokens = max_new_tokens

        # ⚠️ 只有需要 Harmony 的情況才 import，避免其他模型也被迫依賴
        from openai_harmony import (
            load_harmony_encoding,
            HarmonyEncodingName,
            Role,
            Message,
            Conversation,
            DeveloperContent,
            SystemContent,
        )

        self.Role = Role
        self.Message = Message
        self.Conversation = Conversation
        self.DeveloperContent = DeveloperContent
        self.SystemContent = SystemContent
        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    async def inference(self, developer_instruction: str, user_content: str) -> str:
        convo = self.Conversation.from_messages([
            self.Message.from_role_and_content(self.Role.SYSTEM, self.SystemContent.new()),
            self.Message.from_role_and_content(
                self.Role.DEVELOPER,
                self.DeveloperContent.new().with_instructions(developer_instruction),
            ),
            self.Message.from_role_and_content(self.Role.USER, user_content),
        ])

        tokens = self.enc.render_conversation_for_completion(convo, self.Role.ASSISTANT)

        input_device = self._input_device()
        input_tensor = torch.tensor([tokens], device=input_device)
        attention_mask = torch.ones_like(input_tensor, device=input_device)

        with torch.no_grad():
            out = self.model.generate(
                input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.model.config.pad_token_id,
            )

        new_tokens = out[0][len(tokens):].tolist()

        if hasattr(self.enc, "decode_string"):
            return self.enc.decode_string(new_tokens)
        if hasattr(self.enc, "decode"):
            return self.enc.decode(new_tokens)
        raise RuntimeError("Harmony encoding has no decode method.")


class GenericHFStrategy(LLMStrategy):
    """
    適用：一般 HF 模型（不需要 Harmony）
    """
    def __init__(self, model, tokenizer_name_or_path: str, max_new_tokens: int = 2048):
        super().__init__(model)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)

        # 有些 tokenizer 沒有 pad_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    async def inference(self, developer_instruction: str, user_content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant.\n{developer_instruction}",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback for tokenizers that do not define a chat template.
            prompt = (
                "System: You are a helpful assistant.\n"
                f"Developer: {developer_instruction}\n"
                f"User: {user_content}\n"
                "Assistant:"
            )

        input_device = self._input_device()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(input_device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.model.config.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        new_ids = out[0][prompt_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)
