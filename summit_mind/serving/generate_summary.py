from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load your fine-tuned model
model_path = "./models/summit-mind-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model = model.to("cpu")  # or "cuda" if you have a GPU

def summarize(dialogue: str):
    # Prefix "summarize:" just like during training
    input_text = "summarize: " + dialogue
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate the summary
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=60,  # You can adjust how long summaries are
        num_beams=4,        # Beam search improves quality (you can tune later)
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


import re

def extract_action_items(summary: str):
    """
    Extracts action items from a model-generated summary based on common future-oriented patterns.
    """
    action_items = []
    # Split summary into sentences first
    sentences = re.split(r'(?<=[\.\!\?])\s+', summary)
    
    # Look for sentences that likely indicate an action
    for sentence in sentences:
        if re.search(r'\b(will|should|needs to|plans to|agrees to|must)\b', sentence, re.IGNORECASE):
            action_items.append(sentence.strip())
    
    return action_items


def generate_summary_and_actions(dialogue: str):
    """
    Generates a summary and extracts action items from a given dialogue.
    """
    input_text = "summarize: " + dialogue
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    summary_ids = model.generate(
        **inputs,
        max_new_tokens=60,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    actions = extract_action_items(summary)

    return summary, actions


# Example dialogue to test
dialogue1 = """Speaker 1: Hey, are you coming to the meeting today? 
Speaker 2: I might be late, stuck in traffic. 
Speaker 1: No worries, I'll let them know. 
Speaker 2: Thanks!"""


dialogue2 = """Speaker 1: Morning! Are we still meeting today for the sprint planning?
Speaker 2: Hey! Umm, I think so, but did the PM confirm?
Speaker 3: I believe they sent an email yesterday, let me check.
Speaker 1: Oh, right, I might've missed it. 
Speaker 2: If it's not today, we might have to push to tomorrow.
Speaker 3: Yep, they confirmed — 2 PM today.
Speaker 1: Perfect, thanks for checking.
Speaker 2: Cool, see you then."""

dialogue3 = """Speaker 1: Can we move the client meeting to Thursday? 
Speaker 2: I have another call Thursday afternoon.
Speaker 3: What about Friday morning?
Speaker 2: Friday works for me.
Speaker 1: Ok, let's do Friday morning then.
"""

dialogue4 = """Speaker 1: Hello, I'm having trouble logging into my account.
Speaker 2: I'm sorry to hear that. Are you seeing an error?
Speaker 1: Yes, it says password incorrect, but I'm sure it's right.
Speaker 2: I'll escalate this to our support engineers. They'll follow up via email.
Speaker 1: Thanks.
"""

dialogue5 = """Speaker 1: Should we add dark mode support in the next sprint?
Speaker 2: Good idea, customers have asked for it.
Speaker 3: Agreed, but let's check bandwidth.
Speaker 1: Ok, I'll draft a proposal and share it.
Speaker 2: Great, thanks.
"""
summary, actionItems = generate_summary_and_actions(dialogue4)
print("\nGenerated Summary:\n", summary)

if len(actionItems):
    print("\nExtracted Action Items:")
    for item in actionItems:
        print("-", item)
# The above code is a simple example of how to use the T5 model for summarization and action item extraction.