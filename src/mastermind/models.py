from abc import ABC, abstractmethod

from transformers import pipeline


class LanguageModel(ABC):

    @abstractmethod
    def __call__(self):
        pass


class HFModel(LanguageModel):

    def __init__(self, model_name: str, **kwargs):
        self.pipe = pipeline("text-generation", model=model_name, **kwargs)

    def __call__(self, prompt: str, **kwargs):
        return self.pipe(prompt, **kwargs)


class OpenAIModel(LanguageModel):

    def __init__(self):
        pass

    def __call__(self, prompt: str):
        pass


class AnthropicModel(LanguageModel):

    def __init__(self):
        pass

    def __call__(self, prompt: str):
        pass
