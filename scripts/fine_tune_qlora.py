from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

from ai_tutor.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline QLoRA fine-tuning script.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Config.lora_adapter_path),
        help="Directory to save LoRA adapter weights.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train/train.jsonl",
        help="Path to training JSONL file.",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/val/val.jsonl",
        help="Path to validation JSONL file.",
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max training steps (if > 0, overrides num-epochs).",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit QLoRA (requires GPU + bitsandbytes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, print config and exit without training.",
    )
    return parser.parse_args()


def format_example(example: Dict[str, Any]) -> str:
    question = example.get("question", "").strip()
    context = example.get("context", "").strip()
    answer = example.get("answer", "").strip()

    if context:
        prompt = (
            "You are an AI programming tutor.\n\n"
            f"Context:\n{context}\n\n"
            f"Student question:\n{question}\n\n"
            "Tutor answer:\n"
        )
    else:
        prompt = (
            "You are an AI programming tutor.\n\n"
            f"Student question:\n{question}\n\n"
            "Tutor answer:\n"
        )

    return prompt + answer


def main() -> None:
    args = parse_args()

    cfg = Config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== QLoRA Fine-Tuning ===")
    print(f"Base model id:      {cfg.base_model_id}")
    print(f"Embedding model id: {getattr(cfg, 'embedding_model_id', 'N/A')}")
    print(f"Train file:         {args.train_file}")
    print(f"Val file:           {args.val_file}")
    print(f"Output dir:         {output_dir}")
    print(f"Num epochs:         {args.num_epochs}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Learning rate:      {args.learning_rate}")
    print(f"Max steps:          {args.max_steps}")
    print(f"Use 4-bit:          {args.use_4bit}")
    print(f"Dry run:            {args.dry_run}")
    print()

    if args.dry_run:
        print("Dry run enabled. No training will be performed.")
        return

    # Load dataset
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")

    train_ds = train_ds.map(lambda ex: {"text": format_example(ex)})
    val_ds = val_ds.map(lambda ex: {"text": format_example(ex)})

    # Load model + tokenizer
    model_name = cfg.base_model_id

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize for Trainer
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    tokenized_train = train_ds.map(
        tokenize_function, batched=True, remove_columns=train_ds.column_names
    )
    tokenized_val = val_ds.map(
        tokenize_function, batched=True, remove_columns=val_ds.column_names
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # TrainingArguments (compatible with your Transformers version)
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        output_dir=str(output_dir),
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("QLoRA fine-tuning complete.")
    print(f"LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
