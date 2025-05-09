# server.py

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev: allow all. Replace with domain in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preload both models
from fastapi.middleware.cors import CORSMiddleware

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
tokenizers = {}

# # Load T5-small
# models["small"] = T5ForConditionalGeneration.from_pretrained("./models/summit-mind-t5-small").to("cpu")
# tokenizers["small"] = T5Tokenizer.from_pretrained("./models/summit-mind-t5-small")

# # Load T5-base
# models["base"] = T5ForConditionalGeneration.from_pretrained("./models/summit-mind-t5-base-final-pytorch").to("cpu")
# tokenizers["base"] = T5Tokenizer.from_pretrained("./models/summit-mind-t5-base-final-pytorch")

# Lazy-load models and tokenizers only when first requested
models = {"small": None, "base": None}
tokenizers = {"small": None, "base": None}

model_paths = {
    "small": "./models/summit-mind-t5-small",
    "base": "./models/summit-mind-t5-base-final-pytorch"
}


class SummarizeRequest(BaseModel):
    dialogue: str
    t5_model: Optional[str] = "base"  # default to "base"

def extract_action_items(summary: str):
    action_items = []
    sentences = re.split(r'(?<=[\.\!\?])\s+', summary)
    for sentence in sentences:
        if re.search(r'\b(will|should|needs to|plans to|agrees to|must)\b', sentence, re.IGNORECASE):
            action_items.append(sentence.strip())
    return action_items

def generate_summary_and_actions(dialogue: str, model_name: str):
    if models[model_name] is None or tokenizers[model_name] is None:
        print(f"🔄 Loading {model_name} model from disk...")
        tokenizers[model_name] = T5Tokenizer.from_pretrained(model_paths[model_name])
        models[model_name] = T5ForConditionalGeneration.from_pretrained(model_paths[model_name]).to("cpu")

    model = models[model_name]
    tokenizer = tokenizers[model_name]

    print(f"🛠 Running inference with model: {model_name}, device: {model.device}, model id: {id(model)}")
    input_text = "summarize: " + dialogue
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Make sure inputs are moved to the correct model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate summary
    with torch.no_grad():  # ✅ important for safe inference, no gradients
        summary_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    actions = extract_action_items(summary)

    return summary, actions


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    model_choice = request.t5_model.lower() if request.t5_model else "base"

    if model_choice not in models:
        return {"error": f"Invalid model '{model_choice}'. Available options: base, small"}

    summary, actions = generate_summary_and_actions(request.dialogue, model_choice)
    return {
        "summary": summary,
        "action_items": actions,
        "model_used": model_choice
    }
