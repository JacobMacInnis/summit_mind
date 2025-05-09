from transformers import AutoTokenizer

# Path to your fine-tuned PyTorch model
model_path = "./models/summit-mind-t5-base"

# Export to ONNX
onnx_model = export_and_get_onnx_model(model_path)

# Save ONNX model for future loading
onnx_model.save_pretrained("./models/summit-mind-t5-base-onnx")

# Save the tokenizer (optional, usually you already have it)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained("./models/summit-mind-t5-base-onnx")

print("✅ Export complete. ONNX model saved to ./models/summit-mind-t5-base-onnx/")
