# ML-credit-risk-project

# 🏦 Lauki Finance: Credit Risk Prediction Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

**Lauki Finance** is an end-to-end Machine Learning web application designed to assess loan applicant risk in real-time. It predicts the **Probability of Default (PD)** and calculates a standardized **Credit Score (300-900)** based on financial and demographic data.

---

## 🚀 Key Features

* **Real-Time Inference:** Instant calculation of credit risk upon data entry.
* **Credit Scoring System:** Converts raw model log-odds into a consumer-friendly credit score (300-900 scale).
* **Risk Rating:** Automatically categorizes applicants into *Poor, Average, Good,* or *Excellent* tiers.
* **Smart Defaults:** Handles missing financial history data (like previous sanction amounts) using neutral imputation to prevent bias against new borrowers.
* **Interactive UI:** Built with Streamlit for a clean, responsive user experience.

---

## 🛠️ Technical Architecture

### The Machine Learning Pipeline
The core prediction engine is built on a **Logistic Regression** model trained on historical credit data. The pipeline includes:
1.  **Preprocessing:** One-Hot Encoding for categorical variables (e.g., Residence Type, Loan Purpose).
2.  **Scaling:** MinMax scaling applied to numerical features to match the training distribution.
3.  **Scoring Logic:**
    * **Raw Output:** The model calculates the log-odds ($x$).
    * **Probability:** Converted via Sigmoid function: $P(Default) = \frac{1}{1 + e^{-x}}$.
    * **Credit Score:** Derived from the non-default probability: $Score = 300 + (1 - P_{default}) \times 600$.

### Project Structure
```bash
ML-credit-risk-project/
├── Credit Artifacts/
│   └── model_data.joblib    # Pre-trained Model, Scaler, and Column Transformers
├── main.py                  # Streamlit Frontend Application
├── predictionhelper.py      # Backend Logic: Preprocessing & Scoring Engine
├── README.md                # Project Documentation
└── requirements.txt         # Python Dependencies
