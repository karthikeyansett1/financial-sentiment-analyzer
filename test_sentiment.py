"""Quick test: see FinBERT in action on sample headlines."""
from src.sentiment import get_sentiment

test_headlines = [
    "Apple reports record quarterly revenue, beating analyst expectations",
    "Tesla recalls 500,000 vehicles due to safety concerns",
    "Federal Reserve keeps interest rates unchanged",
    "Amazon stock plunges 8% after weak earnings guidance",
    "Microsoft announces $10 billion AI investment partnership",
    "JPMorgan faces regulatory probe over trading practices",
]

print("🧪 Testing FinBERT on sample headlines:\n")
for headline in test_headlines:
    result = get_sentiment(headline)
    
    # Color code: green for positive, red for negative, yellow for neutral
    emoji = "🟢" if result["label"] == "positive" else "🔴" if result["label"] == "negative" else "🟡"
    
    print(f"{emoji} [{result['label'].upper():>8}] (pos:{result['positive']:.2f} neg:{result['negative']:.2f} neu:{result['neutral']:.2f})")
    print(f"   {headline}\n")