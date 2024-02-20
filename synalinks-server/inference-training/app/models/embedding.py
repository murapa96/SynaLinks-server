from typing import Union, List

from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]