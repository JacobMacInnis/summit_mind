# summit_mind/training/finetune_samsum.py
import os

# ✅ Disable TensorFlow and force PyTorch
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import transformers
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np

# ✅ Monkey-patch generation method to prevent MPS LongTensor crash in `generate()`
from transformers.generation.utils import GenerationMixin

def patched_prepare_special_tokens(self, generation_config, has_attention_mask, device):
    def safe_tensor(value):
        tensor = torch.tensor(value, device=device)
        if device.type == "mps" and tensor.dtype == torch.long:
            return tensor.to(dtype=torch.float32)
        return tensor

    eos_token_tensor = safe_tensor(generation_config.eos_token_id) if generation_config.eos_token_id is not None else None
    pad_token_tensor = safe_tensor(generation_config.pad_token_id) if generation_config.pad_token_id is not None else None

    if eos_token_tensor is not None and pad_token_tensor is not None:
        if torch.isin(eos_token_tensor, pad_token_tensor).any():
            generation_config.eos_token_id = generation_config.pad_token_id
    return generation_config

GenerationMixin._prepare_special_tokens = patched_prepare_special_tokens

def main():
    # ✅ Force CPU or MPS consistently
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Loading SAMSum dataset...")
    dataset = load_dataset("samsum", trust_remote_code=True)

    print("Loading model and tokenizer...")
    model_checkpoint = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # ❌ Do NOT manually set decoder_start_token_id — let model handle it
    # model.generation_config.decoder_start_token_id = tokenizer.pad_token_id

    def preprocess_data(batch):
        inputs = ["summarize: " + dialogue for dialogue in batch["dialogue"]]
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            pad_to_multiple_of=8
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"],
                max_length=152,
                padding="max_length",
                truncation=True,
                pad_to_multiple_of=8
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing datasets...")
    tokenized_datasets = dataset.map(preprocess_data, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/summit-mind-t5-small-optimized",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        learning_rate=2e-4,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=1,  # ⏱️ Short epoch for debug
        weight_decay=0.01,
        save_total_limit=2,
        fp16=False,
        logging_dir="./models/logs",
        logging_steps=10,
        report_to="none",
        predict_with_generate=True,
        label_smoothing_factor=0.1,
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {key: value.mid.fmeasure * 100 for key, value in result.items()}

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    print("Initializing Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"].select(range(100)),  # ⏱️ Tiny subset for debug
        eval_dataset=tokenized_datasets["test"].select(range(50)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=data_collator,
    )

    # ✅ Ensure model is on the correct device and generation uses same device
    trainer.model = trainer.model.to(device)
    trainer.gen_kwargs = {
        "max_length": 150,
        "num_beams": 4,
        # "device": device,
    }

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    model.save_pretrained("./models/summit-mind-t5-small")
    tokenizer.save_pretrained("./models/summit-mind-t5-small")

if __name__ == "__main__":
    main()
