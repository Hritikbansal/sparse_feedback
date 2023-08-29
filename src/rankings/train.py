from dataclasses import dataclass, field
from typing import Optional


from transformers import (
    HfArgumentParser,
    TrainingArguments,
    LlamaForSequenceClassification,
    LlamaTokenizerFast,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from utils import RewardDataset, RewardTrainer, custom_collate
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training

import os
import torch

os.environ['TOKENIZERS_PARALLELISM']="false"
# torch.autograd.set_detect_anomaly(True)

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "If you want to resume training where it left off."})
    just_evaluate: Optional[bool] = field(default=False, metadata={"help": "If you want to just evaluate."})
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(default="2", metadata={"help": "The number of training epochs for the reward model. OpenAI used 5."})
    lora_r: Optional[int] = field(default="32", metadata={"help": "Lora r value"})
    output_dir: Optional[str] = field(default = "")
    run_name: Optional[str] = field(default = "test")
    save_file: Optional[str] = field(default = None)
    train_input: Optional[str] = field(default = "data_files/train_score_dense_alpaca_user_orient.csv")
    test_input: Optional[str] = field(default = "data_files/train_score_dense_alpaca_user_orient.csv")


def main():

    # Parse the arguments.
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    num_labels = 1 
    tokenizer = LlamaTokenizerFast.from_pretrained(script_args.model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Allow batched inference
    model  = LlamaForSequenceClassification.from_pretrained(script_args.model_name, num_labels = num_labels, load_in_8bit=True, torch_dtype=torch.float16, device_map = "auto")
    model  = prepare_model_for_int8_training(model)
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=script_args.lora_r, lora_alpha=16, lora_dropout=0.05, target_modules = ["q_proj", "v_proj"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = RewardDataset(script_args.train_input, tokenizer)
    print(len(train_dataset))
    test_dataset = RewardDataset(script_args.test_input, tokenizer)
    print(len(test_dataset))

    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        # save_strategy="steps",
        # save_steps=200,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        remove_unused_columns=False,
        label_names=[],
        ddp_find_unused_parameters=False,
        report_to="none" if script_args.just_evaluate else "wandb",
        logging_steps=50,
        do_eval=True,
        fp16=True,
        run_name=script_args.run_name,
        optim="adamw_torch",
        warmup_steps=100,
    )

    trainer = RewardTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=custom_collate
            )

    if not script_args.just_evaluate:  
        trainer.train(script_args.resume_from_checkpoint)

    else:
        if script_args.output_dir != "":
            ckpt_path = os.path.join(script_args.output_dir, 'pytorch_model.bin')
            with open(ckpt_path, 'rb') as f:
                ckpt = torch.load(f, map_location = torch.device(f"cuda:0"))
            model.load_state_dict(ckpt)
            print('Model Loaded')
        trainer.save_file = script_args.save_file
        print(trainer.evaluate(test_dataset))

if __name__ == "__main__":
    main()

