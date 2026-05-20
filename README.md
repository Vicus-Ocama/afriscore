# 🌍 AfriScore — Alternative Credit Intelligence Platform

> ML-powered credit scoring for the 57% of Sub-Saharan Africans excluded from traditional banking.

**Built by Okurwoth Vicus Ocama**

---

## 📌 Overview

AfriScore is a machine learning web application that assesses the creditworthiness of individuals who have little or no formal banking history.

It takes in 6 months of an applicant's financial behaviour — payment delays, bill amounts, and repayment amounts — and produces:

- A **credit score** between 300 and 850
- A **default risk probability**
- A **SHAP explanation** showing exactly why the score was given

---

## 🎯 Problem Statement

Traditional credit scoring (like FICO) requires bank statements and formal employment records. Millions of Africans are excluded simply because they have never had a bank account — not because they are irresponsible.

AfriScore uses **alternative behavioural signals** to proxy creditworthiness, opening access to finance for the previously excluded.

---

## 🤖 Model

| Detail | Value |
|---|---|
| Algorithm | LightGBM (Gradient Boosted Trees) |
| Training Data | UCI Credit Default Dataset — 30,000 records |
| Features | 30 engineered signals |
| ROC-AUC | 0.78 |
| Explainability | SHAP TreeExplainer |
| Score Range | 300 (high risk) → 850 (low risk) |

### Engineered Features

Beyond the original 24 dataset columns, 6 additional signals were engineered:

| Feature | What It Captures |
|---|---|
| `delay_count` | Number of months with late payments |
| `max_delay` | Worst single payment delay |
| `avg_bill` | Average monthly bill over 6 months |
| `avg_pay_amt` | Average monthly payment made |
| `pay_ratio` | Average payment ÷ average bill |
| `utilization` | Last month bill ÷ credit limit |
| `bill_trend` | Last month bill minus 6-months-ago bill |

---

## 🚀 How to Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/afriscore.git
cd afriscore
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the app**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## ☁️ Live Demo

👉 [Open AfriScore on Streamlit Cloud](YOUR_STREAMLIT_LINK_HERE)

---

## 📁 Project Structure

```
afriscore/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── models/
│   ├── lgbm_model.pkl      # Trained LightGBM model
│   └── feature_cols.pkl    # Feature column order
└── README.md
```

---

## ⚖️ Fairness & Ethics

- Every decision is explained using **SHAP** — no black-box approvals
- A **manual review tier** exists for borderline cases
- Gender and education **bias monitoring** is built into the pipeline
- Aligns with **GDPR Article 22** on algorithmic transparency

---

## 🛠️ Tech Stack

`Python` · `LightGBM` · `SHAP` · `Streamlit` · `pandas` · `NumPy` · `Matplotlib`

---

## 📊 Data Source

UCI Machine Learning Repository —
[Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
30,000 credit card clients · Taiwan · 2005

---

*AfriScore is a demonstration project for educational and research purposes.*