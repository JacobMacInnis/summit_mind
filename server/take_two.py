import os
from transformers import T5Tokenizer
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# Paths
MODEL_DIR = "./models/summit-mind-t5-small"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load tokenizer and ONNX-optimized model
print("Loading tokenizer and ONNX model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR, use_auth_token=False)
print("Model loaded!")

# Load dataset
print("Loading SAMSum test dataset...")
dataset = load_dataset("samsum", trust_remote_code=True)
test_data = dataset["test"]
print("Dataset loaded!")

# Inference function
def summarize(dialogue: str, max_length=150):
    input_text = "summarize: " + dialogue
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True
    )

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        do_sample=False,
        num_beams=1,  # Greedy decoding
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Run on 5 samples
print("Running inference on 5 test examples...")
for i in range(5):
    dialogue = test_data[i]["dialogue"]
    target = test_data[i]["summary"]
    pred = summarize(dialogue)
    
    print(f"\n=== Example {i+1} ===")
    print(f"Dialogue:\n{dialogue}")
    print(f"\nTarget Summary:\n{target}")
    print(f"\nPredicted Summary:\n{pred}")
