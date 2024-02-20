from .remote_llm_abstract import RemoteLLM
import openai
from typing import List


class RemoteLLMOpenAI(RemoteLLM):
    model: str = "davinci-codex"

    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_list: List[str],
    ) -> str:
        openai.api_key = self.api_key
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].text.strip()

    async def async_generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_list: List[str],
    ) -> str:
        openai.api_key = self.api_key
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def list_models(self):
        return openai.Models.list()

    def set_model(self, model_name):
        self.model = model_name
