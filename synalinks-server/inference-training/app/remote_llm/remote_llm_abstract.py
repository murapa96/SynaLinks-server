from abc import ABC, abstractmethod
from typing import List, Any
from pydantic import BaseModel

class RemoteLLM(ABC, BaseModel):
    """
    Abstract class for remote language model
    """
    host: str
    port: str
    model: str
    api_key: str

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
    def async_generate(
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


