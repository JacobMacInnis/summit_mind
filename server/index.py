import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

torch.set_num_threads(1)

# Read model name from environment variable ("small" or "base")
MODEL_NAME = os.environ.get("MODEL_NAME", "base").lower()
assert MODEL_NAME in ["small", "base"], "MODEL_NAME must be 'small' or 'base'"

MODEL_PATHS = {
    "small": "./models/summit-mind-t5-small",
    "base": "./models/summit-mind-t5-base-final-pytorch"
}

print(f"🔄 Loading model '{MODEL_NAME}' from {MODEL_PATHS[MODEL_NAME]}")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATHS[MODEL_NAME])
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATHS[MODEL_NAME]).to("cpu")

# Warm model to reduce first-request latency
dummy = tokenizer("summarize: hello", return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    _ = model.generate(**dummy, max_new_tokens=10)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://summit-mind-55f22.web.app",
        "https://summit-mind-55f22.firebaseapp.com"
    ],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    dialogue: str

def extract_action_items(summary: str):
    action_items = []
    sentences = re.split(r'(?<=[\.!?])\s+', summary)
    for sentence in sentences:
        if re.search(r'\b(will|should|needs to|plans to|agrees to|must)\b', sentence, re.IGNORECASE):
            action_items.append(sentence.strip())
    return action_items


def generate_summary_and_actions(dialogue: str):
    input_text = "summarize: " + dialogue
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=1,
            early_stopping=True
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    actions = extract_action_items(summary)
    return summary, actions

@app.options("/summarize")
async def options_summarize():
    return JSONResponse(content=None)

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    summary, actions = generate_summary_and_actions(request.dialogue)
    return {
        "summary": summary,
        "action_items": actions,
        "model_used": MODEL_NAME
    }
