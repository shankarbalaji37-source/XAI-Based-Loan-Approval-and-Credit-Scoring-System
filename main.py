import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime import lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Loan Dataset for Instant Execution
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame(
    {
        "Income_k": np.random.normal(65, 25, n_samples).clip(15, 200),
        "Credit_Score": np.random.normal(680, 70, n_samples).clip(300, 850),
        "Loan_Amount_k": np.random.normal(150, 60, n_samples).clip(20, 500),
        "Debt_to_Income_Ratio": np.random.uniform(0.1, 0.6, n_samples),
        "Employment_Years": np.random.exponential(5, n_samples).clip(0, 30),
    }
)

# Rule-based target generation (Approval logic)
score_formula = (
    (data["Credit_Score"] - 600) * 0.4
    + (data["Income_k"] - 30) * 0.3
    - (data["Debt_to_Income_Ratio"] * 100) * 0.2
    + (data["Employment_Years"] * 3)
)
data["Loan_Approved"] = (score_formula > 15).astype(int)

# 2. Train-Test Split
X = data.drop(columns=["Loan_Approved"])
y = data["Loan_Approved"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation Output
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 4. Global & Local Interpretability with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Save Global Feature Importance Plot
plt.figure()
shap.summary_plot(
    shap_values[:, :, 1], X_test, plot_type="bar", show=False
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.close()

# Save Local Waterfall Explanation for Applicant #0
plt.figure()
shap.plots.waterfall(shap_values[0, :, 1], show=False)
plt.tight_layout()
plt.savefig("shap_waterfall_applicant_0.png", dpi=300)
plt.close()

# 5. Local Explanation with LIME
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=["Denied", "Approved"],
    mode="classification",
    random_state=42,
)

# Generate and save HTML explanation report for Applicant #0
exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[0], predict_fn=model.predict_proba, num_features=5
)
exp.save_to_file("lime_applicant_0_explanation.html")

print("Outputs saved: 'shap_summary.png', 'shap_waterfall_applicant_0.png', 'lime_applicant_0_explanation.html'")
