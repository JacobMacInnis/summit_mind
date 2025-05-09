# generate_summary_onnx.py

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./models/summit-mind-t5-base-final-pytorch")

# Load ONNX model
session = ort.InferenceSession("./models/summit-mind-t5-base-onnx/model.onnx")

def generate_summary(dialogue: str, max_new_tokens=60):
    input_text = "summarize: " + dialogue
    inputs = tokenizer(input_text, return_tensors="np", max_length=512, truncation=True, padding="max_length")
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Start with BOS token for decoder
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)  # start decoding

    generated_ids = []
    
    for _ in range(max_new_tokens):
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": np.ones_like(decoder_input_ids),
        }
        outputs = session.run(None, ort_inputs)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        next_token_id = np.argmax(next_token_logits, axis=-1).reshape(-1, 1)

        # If end-of-sequence token is generated, break
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Append next token
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token_id], axis=-1)
        generated_ids.append(next_token_id.item())

    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return summary.strip()

# Example usage
if __name__ == "__main__":
    dialogue = """Speaker 1: Hey, are we meeting today? Speaker 2: Yeah, at 2 PM."""
    summary = generate_summary(dialogue)
    print("Generated Summary:", summary)
