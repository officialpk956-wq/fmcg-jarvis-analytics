📊 FMCG Jarvis Analytics

An AI-powered business analytics assistant for the FMCG (Fast-Moving Consumer Goods) domain, combining SQL-based reporting, machine learning forecasting, and what-if simulations, delivered through an interactive Streamlit web application.

This project demonstrates how data analytics and ML can be used together to support real-world business decision-making.

🚀 Project Overview

In FMCG businesses, decisions around inventory, promotions, and demand forecasting are time-sensitive and data-driven.
This project simulates a real internal analytics tool used by managers to answer questions such as:

How much did we sell?

Do promotions really work?

What happens if inventory drops?

Which factors drive sales the most?

FMCG Jarvis acts as a guided analytics assistant that answers these questions using historical data, ML models, and scenario analysis.

🧠 Key Features
📊 Descriptive Analytics (SQL)

Total units sold by time period

Promotion vs non-promotion sales comparison

Region and category-level performance insights

🔮 Predictive Analytics (Machine Learning)

Expected daily sales estimation

Demand forecasting using historical patterns

Feature-driven sales prediction using XGBoost

📉 Prescriptive Analytics (What-If Scenarios)

Impact of inventory reduction (e.g., 20% stock drop)

Promotion effectiveness under low stock conditions

Identification of key drivers affecting sales

🖥 Interactive Streamlit App

Executive-style dropdown-based question selection

Clean, readable business summaries

No raw tables or technical outputs exposed to the user

🏗️ Project Architecture
FMCG Data (CSV)
      ↓
SQLite Database (SQL Analytics)
      ↓
Feature Engineering (lags, rolling averages, ratios)
      ↓
ML Model (XGBoost)
      ↓
Scenario Simulations
      ↓
Streamlit Web App (Jarvis Interface)

🛠️ Tech Stack

Programming Language: Python

Database: SQLite

Data Analysis: Pandas, NumPy

Machine Learning: XGBoost, Scikit-learn

Visualization & UI: Streamlit

Version Control: Git & GitHub

📂 Project Structure
fmcg-jarvis-analytics/
│
├── app.py                  # Streamlit application
├── data/
│   └── fmcg_data.db        # SQLite database
├── tools/
│   ├── xgb_model.pkl       # Trained ML model
│   └── X_test.pkl          # Model features
├── fmcg_jarvis/
│   ├── jarvis.py           # Main orchestration logic
│   ├── jarvis_sql.py       # SQL analytics functions
│   ├── jarvis_ml.py        # ML & simulation logic
│   └── jarvis_nlp.py       # (Reserved for NLP extension)
├── notebooks/
│   └── analysis.ipynb      # Data exploration & modeling
├── README.md
└── .gitignore

📈 Example Business Questions Answered

Total units sold in 2024

Do promotions increase sales?

What if stock drops by 20%?

Should we promote when inventory is low?

What factors affect sales the most?

Each answer is backed by data, ML predictions, or controlled simulations.

🎯 Business Impact

Helps managers anticipate demand risks

Supports inventory and promotion planning

Demonstrates how ML can be used beyond prediction — for decision support
