from .remote_llm_abstract import RemoteLLM
from typing import List
from synalinks.client import SynaLinksClient,SynaLinksAsyncClient
from synalinks.models.chat_completion import ChatMessage


class RemoteLLMSynalinks(RemoteLLM):

    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_list: List[str],
    ) -> str:
        client = SynaLinksClient(api_key=self.api_key)
        chat_response = client.chat(
            model=self.model,
            messages=[ChatMessage(prompt, role="user")],

        )
        return chat_response.choices[0].message.content

    async def async_generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_list: List[str],
    ) -> str:
        client = SynaLinksAsyncClient(api_key=self.api_key)
        chat_response = await client.chat(
            model=self.model,
            messages=[ChatMessage(
                role="user", content=prompt)],
        )
        await client.close()
        return chat_response.choices[0].message.content

    def list_models(self):
        client = SynaLinksClient(api_key=self.api_key)
        list_models_response =  client.list_models()
        return list_models_response.models

    def set_model(self, model_name):
        self.model = model_name
        return self.model
