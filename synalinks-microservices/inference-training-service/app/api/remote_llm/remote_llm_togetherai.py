from .remote_llm_abstract import RemoteLLM
import together


class RemoteLLMTogetherAI(RemoteLLM):
    def __init__(self, host, port, api_key):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.model = ""

    def generate(self, prompt):
        output = together.Complete.create(
            prompt="<human>: What are Isaac Asimov's Three Laws of Robotics?\n<bot>:",
            model="togethercomputer/RedPajama-INCITE-7B-Instruct",
            max_tokens=256,
            temperature=0.8,
            top_k=60,
            top_p=0.6,
            repetition_penalty=1.1,
            stop=['<human>', '\n\n']
        )
        return output.choices[0].text.strip()

    def list_models(self):
        return together.Models.list()

    def set_model(self, model_name):
        self.model = model_name