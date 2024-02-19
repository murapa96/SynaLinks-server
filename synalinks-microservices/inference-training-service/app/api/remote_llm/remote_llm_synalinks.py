from .remote_llm_abstract import RemoteLLM
from typing import List


class RemoteLLMSynalinks(RemoteLLM):

    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_list: List[str],
    ) -> str:
        pass

    async def async_generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_list: List[str],
    ) -> str:
        pass

    def list_models(self):
        pass

    def set_model(self, model_name):
        pass
