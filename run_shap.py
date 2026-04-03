"""
SHAP Analysis — Explain what drives the model's predictions.

WHAT IS SHAP?
SHAP (SHapley Additive exPlanations) tells you HOW MUCH each feature
contributed to each prediction. It comes from game theory — it's like
figuring out how much each player on a team contributed to winning.

WHY IT MATTERS:
- "My model is 62% accurate" is okay
- "My model uses sentiment_spread and volatility as top predictors, 
   and here's how they influence each prediction" is GREAT
- Interviewers LOVE seeing SHAP plots — it shows you don't just 
   build black boxes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from src.database import load_from_db
from src.model import FEATURE_COLS, prepare_data

# Prepare data (same split as training)
X_train, X_test, y_train, y_test, scaler = prepare_data()

# Load the saved model
model = joblib.load("models/best_model.pkl")
print(f"Loaded model: {type(model).__name__}")

# For Logistic Regression, we use LinearExplainer
# For tree models, we'd use TreeExplainer
if "Logistic" in type(model).__name__:
    explainer = shap.LinearExplainer(model, X_train)
else:
    explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

# Plot 1: Summary plot — which features matter most overall
print("\n📊 Generating SHAP summary plot...")
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, show=False)
plt.title("SHAP Feature Importance", fontweight="bold")
plt.tight_layout()
plt.savefig("docs/06_shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved docs/06_shap_summary.png")

# Plot 2: Bar plot — mean absolute SHAP values (simpler view)
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, 
                  plot_type="bar", show=False)
plt.title("Mean Feature Importance (SHAP)", fontweight="bold")
plt.tight_layout()
plt.savefig("docs/07_shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved docs/07_shap_bar.png")

# Print feature importance ranking
print("\n=== FEATURE IMPORTANCE RANKING ===")
mean_shap = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mean_shap": mean_shap
}).sort_values("mean_shap", ascending=False)
print(importance.to_string(index=False))