"""
sentiment.py — Financial sentiment analysis using FinBERT.

WHAT THIS DOES:
Takes a news headline like "Apple reports record quarterly revenue"
and returns: {label: "positive", positive: 0.92, negative: 0.03, neutral: 0.05}

HOW IT WORKS:
1. Tokenizer converts text into numbers (tokens) the model understands
2. Model processes tokens through 12 layers of "attention" — each layer
   learns which words in the sentence relate to each other
3. Final layer outputs 3 raw scores (logits)
4. Softmax converts logits into probabilities that sum to 1.0

WHY FinBERT OVER REGULAR BERT:
Regular BERT was trained on Wikipedia and books.
FinBERT was FURTHER trained on 50,000+ financial news articles.
So it knows "guidance raised" = positive, "margin compression" = negative.
A general model might miss these domain-specific signals.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd


# Load model and tokenizer ONCE when this module is imported
# This avoids reloading the 438MB model for every headline
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()  # Set to evaluation mode (disables dropout — we're not training)
print("✅ FinBERT loaded!")

# These are the three output classes, in order
LABELS = ["positive", "negative", "neutral"]


def get_sentiment(text: str) -> dict:
    """
    Analyze sentiment of a single text string.
    
    Parameters:
    - text: a news headline or short text
    
    Returns: dict with 'label', 'positive', 'negative', 'neutral' scores
    
    STEP BY STEP:
    1. tokenizer() converts "Apple beats earnings" into token IDs like [101, 8347, 14428, ...]
       - max_length=512 truncates if text is too long (headlines are short, so rarely needed)
       - return_tensors="pt" returns PyTorch tensors (the format the model expects)
    
    2. model(**inputs) runs the forward pass through all 12 transformer layers
       - torch.no_grad() tells PyTorch not to track gradients (saves memory, we're not training)
    
    3. softmax converts raw scores into probabilities
       - e.g., logits [2.1, -0.5, 0.3] → probabilities [0.85, 0.06, 0.09]
    """
    # Step 1: Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Step 2: Run through model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Step 3: Convert to probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores = probs.numpy()[0]
    
    return {
        "label": LABELS[np.argmax(scores)],
        "positive": float(scores[0]),
        "negative": float(scores[1]),
        "neutral": float(scores[2]),
    }


def analyze_headlines(headlines: list, show_progress: bool = True) -> list:
    """
    Analyze sentiment for a list of headlines.
    
    Uses tqdm to show a progress bar since this can take a few minutes
    for thousands of headlines.
    """
    results = []
    iterator = tqdm(headlines, desc="Analyzing sentiment") if show_progress else headlines
    
    for headline in iterator:
        try:
            result = get_sentiment(headline)
            results.append(result)
        except Exception as e:
            # If a headline fails, give it neutral scores rather than crashing
            results.append({
                "label": "neutral",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
            })
    
    return results