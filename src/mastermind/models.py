import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ChatHistory = List[Dict[str, str]]


@dataclass
class GenerationArgs:
    max_tokens: int = 2048
    temperature: float = 0.9

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return self.__annotations__.keys()

    def hf_format(self):
        return {self._to_hf_format(k): v for k, v in self.__dict__.items() if v is not None}

    def _to_hf_format(self, arg):
        if arg == "max_tokens":
            return "max_new_tokens"
        return arg


class LanguageModel(ABC):
    @abstractmethod
    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        pass

    @abstractmethod
    def get_model_info(self) -> str:
        pass


class HFModel(LanguageModel):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        generation_args: Optional[GenerationArgs] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if "torch_dtype" not in kwargs and device.startswith("cuda") and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        if "device_map" not in kwargs:
            self.model = self.model.to(device)
        self.model.eval()
        self.generation_args = generation_args or GenerationArgs().hf_format()

    def __call__(self, chat_history: ChatHistory, **kwargs) -> ChatHistory:
        model_inputs = self._prepare_inputs(chat_history)
        generation_args = {**self.generation_args, **kwargs}
        if "do_sample" not in generation_args and "temperature" in generation_args:
            generation_args["do_sample"] = generation_args["temperature"] > 0
        if self.tokenizer.eos_token_id is not None:
            generation_args.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        with torch.inference_mode():
            outputs = self.model.generate(**model_inputs, **generation_args)

        input_length = model_inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        chat_history.append({"role": "assistant", "content": generated_text})
        return chat_history

    def _prepare_inputs(self, chat_history: ChatHistory) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            template_kwargs = {}
            if self.enable_thinking is not None:
                template_kwargs["enable_thinking"] = self.enable_thinking
            model_inputs = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **template_kwargs,
            )
        else:
            rendered_history = self._render_chat_history(chat_history)
            model_inputs = self.tokenizer(rendered_history, return_tensors="pt")

        return {key: value.to(self.model.device) for key, value in model_inputs.items()}

    def _render_chat_history(self, chat_history: ChatHistory) -> str:
        rendered_turns = []
        for turn in chat_history:
            role = turn["role"].capitalize()
            rendered_turns.append(f"{role}: {turn['content']}")
        rendered_turns.append("Assistant:")
        return "\n".join(rendered_turns)

    def get_model_info(self) -> str:
        return f"HF Model: {self.model_name}"


class OpenAIModel(LanguageModel):
    def __init__(self, model_name: str = "gpt-4-turbo-preview", generation_args: Optional[GenerationArgs] = None):
        """Initialize OpenAI model with API key from environment variables."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install the openai package with 'pip install openai'")

        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_history,
        )

        chat_history.append({"content": response.choices[0].message.content, "role": "assistant"})

        return chat_history

    def get_model_info(self) -> str:
        return f"OpenAI Model: {self.model_name}"


class VLLMModel(LanguageModel):
    def __init__(
        self,
        model_name: str,
        generation_args: Optional[GenerationArgs] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install the openai package with 'pip install openai'")

        self.model_name = model_name
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.client = OpenAI(
            api_key=api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
            base_url=self.base_url,
        )
        self.generation_args = generation_args or GenerationArgs()

    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_history,
            max_tokens=self.generation_args.max_tokens,
            temperature=self.generation_args.temperature,
        )

        chat_history.append({"content": response.choices[0].message.content, "role": "assistant"})
        return chat_history

    def get_model_info(self) -> str:
        return f"vLLM Model: {self.model_name} @ {self.base_url}"


class AnthropicModel(LanguageModel):
    def __init__(self, model_name: str = "claude-3-opus-20240229", generation_args: Optional[GenerationArgs] = None):
        """Initialize Anthropic model with API key from environment variables."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Please install the anthropic package with 'pip install anthropic'")

        self.model_name = model_name
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.generation_args = generation_args or GenerationArgs()

    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        """Convert chat history to Anthropic format and make API call."""

        # NOTE: Anthropic model requires the system message to be explicitly set,
        # so we have to fill the first message with a user message, as empty messages are not allowed
        response = self.client.messages.create(
            model=self.model_name,
            system="You are the codebreaker in a game of Mastermind. Find out the secret code based on the instructions given by the user.",
            messages=chat_history,
            max_tokens=self.generation_args.max_tokens,
            temperature=self.generation_args.temperature,
        )

        chat_history.append({"content": response.content[0].text, "role": "assistant"})

        return chat_history

    def get_model_info(self) -> str:
        return f"Anthropic Model: {self.model_name}"
