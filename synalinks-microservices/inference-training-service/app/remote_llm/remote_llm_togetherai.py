from typing import List, Any
from .remote_llm_abstract import RemoteLLM
import together


class RemoteLLMTogetherAI(RemoteLLM):
    model: str = "togethercomputer/RedPajama-INCITE-7B-Instruct"

    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_list: List[str],
    ) -> str:
        together.api_key = self.api_key
        output = together.Complete.create(
            prompt=self.prompt,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=60,
            top_p=top_p,
            repetition_penalty=1.1,
            stop=stop_list,
        )
        return output.choices[0].text.strip()

    async def async_generate(
        self,
        prompt: str,
        max_tokens: int,
        stop_list: List[str],
    ) -> str:
        together.api_key = self.api_key
        output = together.Complete.create(
            prompt=self.prompt,
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.7,
            top_k=60,
            top_p=0.9,
            repetition_penalty=1.1,
            stop=stop_list,
        )
        return output.choices[0].text.strip()

    def list_models(self) -> List[Any]:
        return together.Models.list()

    def set_model(self, model_name: str):
        self.model = model_name
