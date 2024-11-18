from abc import ABC, abstractmethod
from typing import Dict, List

from transformers import pipeline

CHAT_HISTORY = List[Dict[str, str]]


class LanguageModel(ABC):

    @abstractmethod
    def __call__(self, chat_history: CHAT_HISTORY) -> CHAT_HISTORY:
        pass

    @abstractmethod
    def get_model_info(self) -> str:
        pass


class HFModel(LanguageModel):

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.pipe = pipeline("text-generation", model=model_name, device="mps", **kwargs)

    def __call__(self, chat_history: CHAT_HISTORY, **kwargs) -> CHAT_HISTORY:
        return self.pipe(chat_history, max_new_tokens=128, **kwargs)[0]['generated_text']

    def get_model_info(self) -> str:
        return self.model_name


class OpenAIModel(LanguageModel):

    def __init__(self):
        pass

    def __call__(self, chat_history: CHAT_HISTORY) -> CHAT_HISTORY:
        pass

    def get_model_info(self) -> str:
        pass


class AnthropicModel(LanguageModel):

    def __init__(self):
        pass

    def __call__(self, chat_history: CHAT_HISTORY) -> CHAT_HISTORY:
        pass

    def get_model_info(self) -> str:
        pass
