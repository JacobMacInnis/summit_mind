from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from evaluate import load as load_metric

import numpy as np

# Load model and tokenizer
model_path = "./models/summit-mind-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model = model.to("cpu")

# Load dataset
dataset = load_dataset("samsum", trust_remote_code=True)

def preprocess_data(batch):
    inputs = ["summarize: " + dialogue for dialogue in batch["dialogue"]]
    model_inputs = tokenizer(
        inputs, max_length=512, padding="max_length", truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["summary"], max_length=150, padding="max_length", truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_data, batched=True)

rouge = load_metric("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # No .mid.fmeasure anymore
    result = {key: value * 100 for key, value in result.items()}  # ✅

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result




# Setup training args (dummy for evaluation)
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/summit-mind-t5-small",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    do_train=False,
    no_cuda=True,
)

# Setup Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics, 
)

# Run Evaluation
metrics = trainer.evaluate()
print(metrics)
