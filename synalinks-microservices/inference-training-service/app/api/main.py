
from decouple import config
from fastapi import FastAPI

from .models import EmbeddingRequest, ChatCompletionRequest

app = FastAPI()

EMBEDDING_MODEL_NAME = config("EMBEDDING_MODEL_NAME")
LLM_MODEL_NAME = config("LLM_MODEL_NAME")


@app.get("/v1/embeddings")
async def embed(request: EmbeddingRequest):
    pass #TODO

@app.get("/v1/chat/completions")
async def generate(request: ChatCompletionRequest):
    pass #TODO