from typing import Union, List, Any

from .local_embedding_abstract import LocalEmbedding

from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig,
)

import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LocalEmbedding(LocalEmbedding):
    model_name_or_path: str
    model: Any
    tokenizer: PreTrainedTokenizerFast

    def __init__(
            self,
            model_name_or_path: str,
        ):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True)
        super().__init__(
            model_name_or_path=model_name_or_path,
            model=model,
            tokenizer=tokenizer,
        )

    def embed(
            self,
            sentences: Union[str, List[str]]
        ) -> Union[List[float], List[List[float]]]:
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    async def async_embed(
            self,
            sentences: Union[str, List[str]]
        ) -> Union[List[float], List[List[float]]]:
        pass #TODO
