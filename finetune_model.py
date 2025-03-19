import sys
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python finetune_model.py <jsonl_file> <model_dir>")
        sys.exit(1)
    jsonl_file = sys.argv[1]
    model_dir = sys.argv[2]

    # Load dataset
    dataset = load_dataset("json", data_files=jsonl_file, split="train")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=10,  # Adjust for actual training
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            output_dir="outputs",
            seed=3407,
        ),
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(model_dir)