from typing import Union, Optional, List

from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    random_seed: Optional[int] = None
    stop: List[str] = []
