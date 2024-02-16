from .remote_llm_abstract import RemoteLLM
import openai


class RemoteLLMOpenAi(RemoteLLM):
    def __init__(self, host, port, api_key):
        self.host = host
        self.port = port
        self.api_key = api_key

    def generate(self, prompt):
        openai.api_key = self.api_key
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    def list_models(self):
        return openai.Models.list()

    def set_model(self, model_name):
        self.model = model_name