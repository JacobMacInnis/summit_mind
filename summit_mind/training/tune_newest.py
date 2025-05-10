import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import platform
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig
)

# Safe monkey patch for macOS < 14 (MPS + torch.isin)
if torch.backends.mps.is_available() and int(platform.mac_ver()[0].split(".")[0]) < 14:
    def isin_mps_friendly(elements, test_elements):
        return torch.tensor(
            np.isin(elements.cpu().numpy(), test_elements.cpu().numpy()),
            dtype=torch.bool,
            device=elements.device
        )
    torch.isin = isin_mps_friendly

# Load SAMSum and shrink for dev testing
raw_datasets = load_dataset("samsum")
train_dataset = raw_datasets["train"]
val_dataset = raw_datasets["validation"]

# Load tokenizer + model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Ensure proper config
model.config.pad_token_id = tokenizer.pad_token_id or 0
model.config.eos_token_id = tokenizer.eos_token_id or 1
model.config.decoder_start_token_id = model.config.decoder_start_token_id or model.config.pad_token_id

# Preprocessing
def preprocess(batch):
    inputs = ["summarize: " + d for d in batch["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    targets = tokenizer(text_target=batch["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = [
        [(tid if tid != tokenizer.pad_token_id else -100) for tid in target]
        for target in targets["input_ids"]
    ]
    return model_inputs

tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=["dialogue", "summary", "id"])
tokenized_val = val_dataset.map(preprocess, batched=True, remove_columns=["dialogue", "summary", "id"])

# Collator and metric
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
import evaluate
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.array(predictions)
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
    decoded_preds = tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    labels = np.clip(labels, 0, tokenizer.vocab_size - 1)
    decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v * 100, 4) for k, v in result.items()}

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/summit-mind-t5-small-optimized",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    generation_max_length=128,
    generation_num_beams=4,
    logging_steps=10,
    overwrite_output_dir=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

def main():
    trainer.train()
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_length = 128
    gen_cfg.num_beams = 4

    sample_input = val_dataset[0]["dialogue"]
    input_ids = tokenizer("summarize: " + sample_input, return_tensors="pt").input_ids
    if torch.backends.mps.is_available():
        input_ids = input_ids.to("mps")
        model.to("mps")
    output_ids = model.generate(input_ids, generation_config=gen_cfg)
    print("\nGenerated summary:\n", tokenizer.decode(output_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
