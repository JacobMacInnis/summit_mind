# save_final_model.py
from transformers import T5ForConditionalGeneration, T5Tokenizer

checkpoint_path = "./models/summit-mind-t5-base/checkpoint-11049"

# Just load normally
model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)

final_save_path = "./models/summit-mind-t5-base-final-pytorch"

# Force classic PyTorch save (safe_serialization=False)
model.save_pretrained(final_save_path, safe_serialization=False)  # ✅ <-- only here
tokenizer.save_pretrained(final_save_path)

print(f"✅ Model saved to {final_save_path} (classic PyTorch format)")
