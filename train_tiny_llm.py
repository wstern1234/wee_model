from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 1. Load tokenizer and model
model_name = "gpt2"  # small enough for GTX 1660 Ti
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # avoid padding issues

model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Load dataset (example: wikitext-2)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 3. Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. For language modeling, labels = input_ids
def set_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

tokenized_datasets = tokenized_datasets.map(set_labels, batched=True)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    save_steps=1000,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # use mixed precision if CUDA
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 7. Train
trainer.train()