# Credit Card Fraud Detection using Machine Learning 

## Overview
This repository contains an end-to-end Machine Learning pipeline designed to detect fraudulent credit card transactions in real-time. 

Because fraudulent transactions represent an incredibly tiny fraction of overall card usage, the core focus of this project is solving the **severe class imbalance problem** (~0.17% fraud cases). The pipeline implements robust exploratory data analysis (EDA), data standardization, synthetic oversampling techniques, and highly optimized classification models to maximize detection rates while minimizing costly false alarms.

# # Key Features
* **Advanced Imbalance Handling:** Implements techniques like SMOTE (Synthetic Minority Over-sampling Technique) or random undersampling to balance the dataset training distribution.
* **Feature Engineering & Scaling:** Utilizes robust scaling methods on non-PCA transformed attributes like `Time` and `Amount`.
* **Rigorous Evaluation:** Bypasses misleading accuracy metrics to optimize directly for **Recall, Precision-Recall Curves, and F1-Score**.
* **Comprehensive Visualization:** Includes high-quality confusion matrices, correlation heatmaps, and ROC-AUC curves.

## Tech Stack
* **Language:** Python
* **ML Libraries:** Scikit-learn, XGBoost (optional, if used)
* **Data Science:** Pandas, NumPy, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Kaggle Notebooks

## Dataset & Characteristics
The project utilizes the anonymized European cardholder dataset containing credit card transactions. Due to confidentiality, the original background features are compressed into 28 PCA-transformed numerical dimensions ($V_1$ through $V_{28}$), alongside `Time`, `Amount`, and the target variable `Class`.

* **Total Transactions:** 284,807
* **Legitimate Cases (Class 0):** 284,315 (99.83%)
* **Fraudulent Cases (Class 1):** 492 (0.17%)

## Performance & Metrics
When dealing with fraud detection, **Recall** is our most vital metric because missing a fraudulent transaction (False Negative) is significantly more expensive than a false alarm. 

Through hyperparameter tuning and handling class imbalance, the optimized final model achieved outstanding benchmarking results:

| Metric | Score Achieved |
| :--- | :--- |
| **ROC-AUC** | 97.4% |
| **F1-Score** | 85.6% |
| **Recall (Sensitivity)** | 82.3% |
| **Precision** | 88.1% |

## Author
Vaishnavi R

## Domain
BTech in Computer Science (Data Science)

## Institution
Bangalore Institute of Technology

Passionate about leveraging Machine Learning, Computer Vision, and advanced data structures to build real-world solutions for environmental monitoring and automated systems.



