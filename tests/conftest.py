import pytest

from mastermind.models import CHAT_HISTORY, LanguageModel


class DummyModel(LanguageModel):
    def __init__(self, model_name: str = "dummy"):
        self.model_name = model_name

    def __call__(self, chat_history: CHAT_HISTORY) -> CHAT_HISTORY:
        # Simulate a response by appending a fixed message
        response = {"role": "assistant", "content": "This is a dummy response."}
        chat_history.append(response)
        return chat_history

    def get_model_info(self) -> str:
        return self.model_name


@pytest.fixture(scope="module")
def dummy_model():
    return DummyModel(model_name="dummy")
