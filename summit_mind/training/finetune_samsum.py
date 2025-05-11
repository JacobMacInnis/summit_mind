# summit_mind/training/finetune_samsum.py
import transformers
print(f"Transformers version inside script: {transformers.__version__}")
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments


# import torch

def main():
    # Load the SAMSum dataset
    # outdated format => dataset = load_dataset("samsum")
    print("Loading SAMSum dataset...")
    dataset = load_dataset("samsum", trust_remote_code=True)
    print("Dataset loaded!")

    # Load the T5-small model and tokenizer
    print("Loading model and tokenizer...")
    model_checkpoint = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    print("Model and tokenizer loaded!")

    # Tokenization function
    def preprocess_data(batch):
        inputs = ["summarize: " + dialogue for dialogue in batch["dialogue"]]
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"],
                max_length=150,
                padding="max_length",
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    # Preprocess the train and validation datasets
    print("Preprocessing datasets...")
    tokenized_datasets = dataset.map(preprocess_data, batched=True)
    print("Datasets preprocessed!")

    # Set training arguments
    print("Setting training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/summit-mind-t5-small",
        eval_strategy="no",
        learning_rate=2e-4,
        per_device_train_batch_size=4, #8,
        per_device_eval_batch_size=4, #8,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=False, #torch.cuda.is_available(),
        logging_dir="./models/logs",
        logging_steps=50,
        report_to="none",
        predict_with_generate=True,
        
    )
    print("Training arguments set!")

    # Create Trainer instance
    print("Initializing Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )
    print("Trainer initialized!")

    print("Starting training...")
    
    # Fine-tune the model
    trainer.train()
    print("Training complete!")

    # Save the final model
    model.save_pretrained("./models/summit-mind-t5-small")
    tokenizer.save_pretrained("./models/summit-mind-t5-small")

if __name__ == "__main__":
    main()
