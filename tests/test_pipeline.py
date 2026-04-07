"""
tests/test_pipeline.py — Basic tests for the sentiment analysis pipeline.

WHY WE WRITE TESTS:
In production ML systems, you need to know if something breaks.
If someone changes the model, updates a dependency, or refactors code,
tests catch regressions automatically. Running `pytest` gives you
instant confidence that the core logic still works.

HOW PYTEST WORKS:
- Any function starting with `test_` is automatically picked up and run
- If the function raises no error, the test passes (✅)
- If an `assert` fails or an exception is raised, the test fails (❌)
- Run with: pytest tests/ -v
"""

import pytest
from src.sentiment import get_sentiment


def test_sentiment_returns_valid_label():
    result = get_sentiment("Apple reports record quarterly revenue")
    assert result["label"] in ["positive", "negative", "neutral"]


def test_sentiment_returns_all_scores():
    result = get_sentiment("Markets closed mixed ahead of Fed decision")
    assert "positive" in result
    assert "negative" in result
    assert "neutral" in result


def test_sentiment_scores_are_probabilities():
    result = get_sentiment("Earnings beat expectations across all segments")
    assert 0.0 <= result["positive"] <= 1.0
    assert 0.0 <= result["negative"] <= 1.0
    assert 0.0 <= result["neutral"] <= 1.0
    total = result["positive"] + result["negative"] + result["neutral"]
    assert abs(total - 1.0) < 0.01


def test_clearly_positive_headline():
    result = get_sentiment("Company profits surge 50% beating all analyst expectations")
    assert result["positive"] > result["negative"]


def test_clearly_negative_headline():
    result = get_sentiment("Stock crashes after massive accounting fraud scandal revealed")
    assert result["negative"] > result["positive"]


def test_neutral_headline():
    result = get_sentiment("Federal Reserve meets Tuesday to discuss interest rate policy")
    assert result["neutral"] >= result["positive"] or result["neutral"] >= result["negative"]


def test_very_short_headline():
    result = get_sentiment("Bankruptcy")
    assert result["label"] in ["positive", "negative", "neutral"]


def test_long_headline():
    long_text = "Apple Inc. " + "reported strong earnings " * 50
    result = get_sentiment(long_text)
    assert result["label"] in ["positive", "negative", "neutral"]


def test_headline_with_numbers():
    result = get_sentiment("Revenue up 23.5% to $4.2B, EPS of $1.87 beats by $0.12")
    assert result["label"] in ["positive", "negative", "neutral"]
    assert result["positive"] + result["negative"] + result["neutral"] > 0.99