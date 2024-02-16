import os
import torch
from datasets import Dataset
from peft import LoraConfig
from pydantic.v1 import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, PreTrainedTokenizerFast
from transformers import StoppingCriteria
from .training_config import DPOTrainingConfig
from transformers import BitsAndBytesConfig

from trl import DPOTrainer
from typing import Any, List

torch.set_default_device("cuda")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequences: List[List[int]]):
        self.eos_sequences = eos_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        for eos_sequence in self.eos_sequences:
            if self.eos_sequence in last_ids:
                return True
        return False

class LocalLLMModel(BaseModel):
    model_name_or_path: str
    model: Any
    tokenizer: PreTrainedTokenizerFast

    class Config():
        arbitrary_types_allowed = True

    def __init__(
        self,
        model_name_or_path:str,
        ):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True)
        super().__init__(
            model_name_or_path=model_name_or_path,
            model=model,
            tokenizer=tokenizer,
        )

    def dpo_train(
            self,
            output_dir: str,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            training_config: DPOTrainingConfig,
            max_length: int = 1024,
            max_prompt_length: int = 512,
            ):
        """"""

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        model.config.use_cache = False

        model_ref = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        model_ref.config.use_cache = False
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # initialize training arguments

        training_args = TrainingArguments(
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            max_steps=training_config.max_steps,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            gradient_checkpointing=training_config.gradient_checkpointing,
            learning_rate=training_config.learning_rate,
            evaluation_strategy="steps",
            eval_steps=training_config.eval_steps,
            output_dir=output_dir,
            report_to=training_config.report_to,
            lr_scheduler_type=training_config.lr_scheduler_type,
            warmup_steps=training_config.warmup_steps,
            optim=training_config.optimizer_type,
            bf16=True,
            remove_unused_columns=False,
            run_name="dpo_llama2",
        )

        peft_config = LoraConfig(
            r=training_config.lora_r,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc_in",
                "fc_out",
                "wte",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

        dpo_trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            beta=training_config.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            max_prompt_length=max_prompt_length,
            max_length=max_length,
        )

        dpo_trainer.train()

        # train
        dpo_trainer.train()
        dpo_trainer.save_model(output_dir)

        # save
        output_dir = os.path.join(output_dir, "final_checkpoint")
        dpo_trainer.model.save_pretrained(output_dir)

    def predict(self, prompt, max_output_tokens=512, stop="```") -> str:
        """Generate a prediction"""
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        stop_tokens = self.tokenizer.encode(stop)
        outputs = self.model.generate(**inputs, max_length=max_output_tokens, stopping_criteria = [EosListStoppingCriteria(stop_tokens)])
        text = self.tokenizer.batch_decode(outputs)[0]
        return text