# Credit Default Early Warning System

## Project Overview
This project develops an Early Warning System that predicts loan defaults on an imbalanced credit dataset of 15,000 borrowers (7.7% default rate). The system applies a leakage-controlled machine learning pipeline and produces explainable, per-customer risk decisions for use in credit risk assessment.

## Key Features
- **Leakage-Controlled Pipeline:** Strict train/test separation with SMOTE applied to training data only, preserving real-world prevalence in evaluation.
- **Model Comparison:** Benchmarks Logistic Regression, Random Forest, and XGBoost; Random Forest selected as Champion (F1 = 0.737, Precision = 0.857).
- **Explainable AI (SHAP):** Global and local SHAP attributions make every prediction auditable for regulatory review.
- **Cost-Sensitive Decisions:** Threshold tuning under asymmetric costs (FN = 5, FP = 1) as a first step toward Reinforcement Learning.
- **Independent Cross-Check:** Workflow replicated in Orange Data Mining (no-code), confirming the supervised result and adding an unsupervised t-SNE validation.

## Technologies Used
- Python 3.11 (Pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn)
- SHAP for explainability (TreeExplainer)
- Matplotlib / Seaborn for visualisations
- Orange Data Mining (no-code cross-check)
- Google Colab (development environment)

## Course Context
Individual project for **IT9201 — Machine Learning and Data Mining**, MSc Artificial Intelligence, Bahrain Polytechnic.
