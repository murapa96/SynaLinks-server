
from decouple import config
from fastapi import FastAPI

from .models import EmbeddingRequest, ChatCompletionRequest
from .api.remote_llm.remote_llm_abstract import RemoteLLMFactory

app = FastAPI()

EMBEDDING_MODEL_NAME = config("EMBEDDING_MODEL_NAME")
LLM_MODEL_NAME = config("LLM_MODEL_NAME")

REMOTE_LLM_HOST = config("REMOTE_LLM_HOST")
REMOTE_LLM_PORT = config("REMOTE_LLM_PORT")
REMOTE_LLM_API_KEY = config("REMOTE_LLM_API_KEY")

REMOTE_LLM_INSTANCE = RemoteLLMFactory.create_remote_llm(
    host=REMOTE_LLM_HOST,
    port=REMOTE_LLM_PORT,
    model=LLM_MODEL_NAME,
    api_key=REMOTE_LLM_API_KEY,
)
@app.get("/v1/embeddings")
async def embed(request: EmbeddingRequest):
    pass #TODO

@app.get("/v1/chat/completions")
async def generate(request: ChatCompletionRequest):
    pass #TODO
