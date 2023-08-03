import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import Dataset
import os
import sys
from typing import List
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training
)
import fire
import torch
from datasets import load_dataset
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Init
model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

tokenizer.pad_token_id = (0)
tokenizer.padding_side = "left"

# Data
'''
Data should be in the form of 
{
  "instruction": string, 
  "input": string, 
  "output": string
}
'''


def sentiment(score):
    if score == 0:
        return "Bearish"
    elif score == 1:
        return "Bullish"
    return "Neutral"


dataset = load_dataset(
    "zeroshot/twitter-financial-news-sentiment",
    revision="main",
    split="train"  # tag name, or branch name, or commit hash
)

data = []

for i in range(len(dataset)):
    data.append({
        "instruction": "Detect the sentiment of the tweet.",
        "input": dataset["text"][i],
        "output": sentiment(dataset["label"][i])
    })


def generate_prompt(data):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
    ### Instruction:
    {data["instruction"]}
    ### Input:
    {data["input"]}
    ### Response:
    {data["output"]}"""


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        padding=False,
        return_tensors=None
    )

    result["labels"] = result["input_ids"].copy()

    return result


def generate(data):
    prompt = generate_prompt(data)
    token = tokenize(prompt)
    return token


data_set = Dataset.from_list(data)

train_val = data_set["train"].train_test_split(
    test_size=800, shuffle=True, seed=42
)

train_data = (
    train_val["train"].map(generate)
)

val_data = (
    train_val["test"].map(generate)
)

# Training
model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,
    warmup_steps=100,
    max_steps=300,
    learning_rate=0.0003,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir="exp",
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)

model.config.use_cache = False

old_state_dict = model.state_dict

model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

model = torch.compile(model)

trainer.train()

model.save_pretrained("exp")
