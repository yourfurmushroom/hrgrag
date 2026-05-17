# llm_strategies.py
from __future__ import annotations

from abc import ABC, abstractmethod
import os
import torch
from transformers import AutoTokenizer


class LLMStrategy(ABC):
    def __init__(self, model):
        self.model = model
        self._normalize_generation_token_ids()

    def _first_token_id(self, value):
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def _config_token_id(self, name):
        return self._first_token_id(getattr(self.model.config, name, None))

    def _normalize_generation_token_ids(self):
        eos_id = self._config_token_id("eos_token_id")
        if isinstance(getattr(self.model.config, "eos_token_id", None), list):
            self.model.config.eos_token_id = eos_id
        pad_id = self._config_token_id("pad_token_id")
        if pad_id is None:
            setattr(self.model.config, "pad_token_id", eos_id)
        else:
            self.model.config.pad_token_id = pad_id

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
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
        self.qwen_enable_thinking = os.getenv("QWEN_ENABLE_THINKING", "").strip().lower() in {"1", "true", "yes"}
        model_name = (tokenizer_name_or_path or "").lower()
        self.is_qwen_family = "qwen" in model_name
        self.is_gemma4_family = "gemma-4" in model_name or "gemma4" in model_name
        print(
            "[llm strategy] "
            f"tokenizer={tokenizer_name_or_path} "
            f"chat_template={bool(getattr(self.tokenizer, 'chat_template', None))} "
            f"gemma4_enable_thinking={False if self.is_gemma4_family else 'n/a'}",
            flush=True,
        )

        # 有些 tokenizer 沒有 pad_token
        tokenizer_eos_id = self._first_token_id(self.tokenizer.eos_token_id)
        if isinstance(self.tokenizer.eos_token_id, list):
            self.tokenizer.eos_token_id = tokenizer_eos_id
        if self.tokenizer.pad_token_id is None and tokenizer_eos_id is not None:
            self.tokenizer.pad_token_id = tokenizer_eos_id
        elif self.tokenizer.pad_token_id is not None:
            self.tokenizer.pad_token_id = self._first_token_id(self.tokenizer.pad_token_id)

    def _build_messages(self, developer_instruction: str, user_content: str):
        system_content = f"You are a helpful assistant.\n{developer_instruction}"
        user_content_final = user_content

        # Qwen3.x supports explicit non-thinking mode. Prefer a model-native switch,
        # and fall back to /no_think prompt control if the current template path
        # does not accept chat_template_kwargs.
        if self.is_qwen_family and not self.qwen_enable_thinking:
            system_content += "\n/no_think"
            if "/no_think" not in user_content_final:
                user_content_final = f"/no_think\n{user_content_final}"

        return [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": user_content_final,
            },
        ]

    async def inference(self, developer_instruction: str, user_content: str) -> str:
        messages = self._build_messages(developer_instruction, user_content)

        if getattr(self.tokenizer, "chat_template", None):
            apply_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.is_qwen_family:
                apply_kwargs["chat_template_kwargs"] = {
                    "enable_thinking": self.qwen_enable_thinking,
                }
            if self.is_gemma4_family:
                apply_kwargs["enable_thinking"] = False
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    **apply_kwargs,
                )
            except TypeError as exc:
                if self.is_gemma4_family:
                    raise TypeError(
                        f"{self.tokenizer_name_or_path} chat template does not accept "
                        "enable_thinking=False, so Gemma4 thinking mode cannot be controlled."
                    ) from exc
                # Older transformers/tokenizers may not accept chat_template_kwargs.
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            if self.is_gemma4_family:
                raise ValueError(
                    f"{self.tokenizer_name_or_path} does not provide a chat template, "
                    "so Gemma4 thinking mode cannot be controlled. Use the instruction-tuned "
                    "model google/gemma-4-E4B-it for benchmarking."
                )
            # Fallback for tokenizers that do not define a chat template.
            prompt = (
                "System: You are a helpful assistant.\n"
                f"Developer: {messages[0]['content']}\n"
                f"User: {messages[1]['content']}\n"
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
