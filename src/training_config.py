from pydantic.v1 import BaseModel, Field
from typing import Optional

class DPOTrainingConfig(BaseModel):
    """
    The arguments for the DPO training.
    """
    # data parameters
    beta: Optional[float] = Field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = Field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = Field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = Field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = Field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = Field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = Field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = Field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = Field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = Field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = Field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = Field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = Field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = Field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = Field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = Field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = Field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = Field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = Field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = Field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = Field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = Field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = Field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = Field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = Field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
