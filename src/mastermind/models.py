import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, pipeline

ChatHistory = List[Dict[str, str]]


@dataclass
class GenerationArgs:
    max_tokens: int = 1024
    temperature: float = 0.7

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
        regex_constrained=True,
        **kwargs,
    ):
        self.model_name = model_name
        self.regex_constrained = regex_constrained
        self.game_regex = r"Guess: \[(red|blue|green|yellow|orange|purple|pink|brown|black|white)(,\s*(red|blue|green|yellow|orange|purple|pink|brown|black|white))*\]"

        self.pipe = pipeline("text-generation", model=model_name, device=device, **kwargs)
        if self.regex_constrained:
            parser = RegexParser(self.game_regex)
            self.prefix_function = build_transformers_prefix_allowed_tokens_fn(self.pipe.tokenizer, parser)
        self.generation_args = generation_args or GenerationArgs()

    def __call__(self, chat_history: ChatHistory, **kwargs) -> ChatHistory:
        return self.pipe(
            chat_history,
            prefix_allowed_tokens_fn=None if not self.regex_constrained else self.prefix_function,
            **self.generation_args.hf_format(),
            **kwargs,
        )[0]["generated_text"]

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
        self.generation_args = generation_args or GenerationArgs()

    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_history,
            temperature=self.generation_args.temperature,
            max_tokens=self.generation_args.max_tokens,
        )

        chat_history.append({"content": response.choices[0].message.content, "role": "assistant"})

        return chat_history

    def get_model_info(self) -> str:
        return f"OpenAI Model: {self.model_name}"


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
            system="You are a logic gamer playing a game of Mastermind. You are the codebreaker.",
            messages=chat_history,
            max_tokens=self.generation_args.max_tokens,
            temperature=self.generation_args.temperature,
        )

        chat_history.append({"content": response.content[0].text, "role": "assistant"})

        return chat_history

    def get_model_info(self) -> str:
        return f"Anthropic Model: {self.model_name}"
