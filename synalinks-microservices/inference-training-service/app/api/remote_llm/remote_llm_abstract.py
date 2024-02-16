from abc import ABC, abstractmethod

class RemoteLLM(ABC):
    """
    Abstract class for remote language model
    """

    @abstractmethod
    def __init__(self, host, port, api_key):
        self.host = host
        self.port = port
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt):
        pass

    @abstractmethod
    def list_models(self):
        pass

    @abstractmethod
    def set_model(self, model_name):
        pass


