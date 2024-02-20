
from decouple import config
from fastapi import FastAPI

from .models.embedding import EmbeddingRequest
from .models.chat_completion import ChatCompletionRequest

from .local_embedding.local_embedding_sentence_transformers import LocalEmbeddingSentenceTransformer
from .local_llm.local_llm_transformers import LocalLLMTransformers
from .remote_llm.remote_llm_abstract import RemoteLLMFactory

app = FastAPI()

EMBEDDING_MODEL_NAME = config(
    "EMBEDDING_MODEL_NAME", default="sentence-transformers/all-MiniLM-l6-v2")
LLM_MODEL_NAME = config("LLM_MODEL_NAME", default="microsoft/phi-2")

REMOTE_LLM_HOST = config("REMOTE_LLM_HOST")
REMOTE_LLM_PORT = config("REMOTE_LLM_PORT")
REMOTE_LLM_API_KEY = config("REMOTE_LLM_API_KEY")

REMOTE_LLM_INSTANCE = RemoteLLMFactory.create_remote_llm(
    host=REMOTE_LLM_HOST,
    port=REMOTE_LLM_PORT,
    model=LLM_MODEL_NAME,
    api_key=REMOTE_LLM_API_KEY,
)


embedding = LocalEmbeddingSentenceTransformer(EMBEDDING_MODEL_NAME)
llm = LocalLLMTransformers(LLM_MODEL_NAME)


@app.get("/v1/embeddings")
def embed(request: EmbeddingRequest):
    return {}


@app.get("/v1/chat/completions")
async def generate(request: ChatCompletionRequest):
    return REMOTE_LLM_INSTANCE.generate(
        prompt=request.messages[0].content,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop_list=request.stop,
    )
