
from abc import ABC, abstractmethod
from typing import Any, List
from pydantic import BaseModel

class LocalLLM(ABC, BaseModel):
    """
    Abstract class for local language model
    """
    model_name_or_path: str
    device: str = "cuda"

    @abstractmethod
    def generate(
            self,
            prompt: str,
            temperature: float,
            top_p: float,
            max_tokens: int,
            stop_list: List[str],
        ) -> str:
        pass

    @abstractmethod
    async def async_generate(
            self,
            prompt: str,
            temperature: float,
            top_p: float,
            max_tokens: int,
            stop_list: List[str],
        ) -> str:
        pass

    @abstractmethod
    def list_models(self) -> List[Any]:
        pass

    @abstractmethod
    def set_model(self, model_name: str):
        pass
