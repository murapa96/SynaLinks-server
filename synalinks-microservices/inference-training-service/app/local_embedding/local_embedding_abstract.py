
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Union, List

class LocalEmbedding(ABC, BaseModel):
    """
    Abstract class for local embedding model
    """
    name_or_path: str
    device: str = "cuda"

    @abstractmethod
    def embed(self, sentences: Union[str, List[str]]):
        pass

    @abstractmethod
    async def async_embed(self, sentences: Union[str, List[str]]):
        pass

    @abstractmethod
    def list_models(self):
        pass

    @abstractmethod
    def set_model(self, model_name: str):
        pass
