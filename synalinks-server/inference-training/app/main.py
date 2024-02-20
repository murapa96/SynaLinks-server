
from decouple import config
from fastapi import FastAPI

from .models.embedding import EmbeddingRequest
from .models.chat_completion import ChatCompletionRequest

from .local_embedding.local_embedding_sentence_transformers import LocalEmbeddingSentenceTransformer
from .local_llm.local_llm_transformers import LocalLLMTransformers

app = FastAPI()

EMBEDDING_MODEL_NAME = config("EMBEDDING_MODEL_NAME", default="sentence-transformers/all-MiniLM-l6-v2")
LLM_MODEL_NAME = config("LLM_MODEL_NAME", default="microsoft/phi-2")

embedding = LocalEmbeddingSentenceTransformer(EMBEDDING_MODEL_NAME)
llm = LocalLLMTransformers(LLM_MODEL_NAME)

@app.get("/v1/embeddings")
def embed(request: EmbeddingRequest):
    return {}

@app.get("/v1/chat/completions")
def generate(request: ChatCompletionRequest):
    return {}