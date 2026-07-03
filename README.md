# JCDSOHAM-006_Alpha
# Seller Churn Prediction in Brazilian E-Commerce Using Machine Learning

##Project Links

Tableau Dashboard: https://public.tableau.com/shared/5TDB44K8S?:display_count=n&:origin=viz_share_link

Streamlit Application: 

Google Colab Notebook: https://colab.research.google.com/drive/1Y-YefeFlTSCet0JfRT6wA3yPLnpXFPPx?usp=sharing

## Project Overview

This project develops a machine learning model to predict seller churn in a Brazilian e-commerce marketplace using the Olist public dataset. By identifying sellers who are likely to stop selling on the platform, the project aims to help business stakeholders implement proactive retention strategies, reduce revenue loss, and improve long-term seller engagement.

The project covers the complete data science workflow, including:

* Data collection and integration
* Data preprocessing and feature engineering
* Exploratory Data Analysis (EDA)
* Statistical analysis
* Machine learning model development
* Model evaluation and threshold optimization
* Seller risk segmentation
* Interactive business dashboard preparation

---

## Objectives

* Predict whether a seller will churn.
* Identify the most important factors influencing seller churn.
* Segment sellers into actionable risk categories.
* Estimate potential revenue at risk.
* Build business-ready datasets for dashboard visualization.

---

## Dataset

This project uses the **Brazilian E-Commerce Public Dataset by Olist**, which contains transactional data from a large Brazilian marketplace.

Main datasets include:

* `olist_orders_dataset.csv`
* `olist_order_items_dataset.csv`
* `olist_sellers_dataset.csv`
* `olist_products_dataset.csv`
* `olist_order_payments_dataset.csv`
* `olist_order_reviews_dataset.csv`

These datasets are integrated to construct seller-level behavioral features.

---

## Feature Engineering

Several seller-level features were created, including:

* Recency (days since last order)
* Order frequency
* Monetary value (total sales)
* Seller tenure
* Average review score
* Average delivery delay
* Average shipping cost
* Number of unique customers
* Number of unique products
* Seller state (Target Encoded)

A seller was labeled as **churned** when no transaction occurred within the predefined churn observation window.

---

## Exploratory Data Analysis

The EDA includes:

* Descriptive statistics
* Missing value analysis
* Distribution analysis
* Box plots
* Mann–Whitney U statistical testing
* Geographic churn analysis
* Seller activity analysis
* Correlation analysis

These analyses provide insights into behavioral differences between active and churned sellers.

---

## Machine Learning

The project evaluates multiple machine learning algorithms and selects the best-performing model based on predictive performance.

Typical workflow:

1. Data preprocessing
2. Feature encoding
3. Train-test split
4. Model training
5. Hyperparameter tuning
6. Threshold optimization
7. Performance evaluation

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Precision-Recall AUC
* Confusion Matrix

The optimal probability threshold is selected to maximize business usefulness rather than relying solely on the default 0.50 cutoff.

---

## Seller Risk Segmentation

Predicted probabilities are converted into business-friendly risk tiers:

| Risk Tier   |              Probability |
| ----------- | -----------------------: |
| High Risk   |                   ≥ 0.80 |
| Medium Risk |              0.50 – 0.79 |
| Low Risk    | Optimal Threshold – 0.49 |
| Active      |  Below Optimal Threshold |

Additional business metrics include:

* Revenue at Risk
* Activity Segment
* Recency Bucket
* Seller Churn Probability

---

## Dashboard

The project exports processed CSV files for interactive dashboard development (Tableau/Power BI), including:

* Seller Risk Profile
* Risk Tier Summary
* Geographic Summary
* Segment Summary

Dashboard KPIs include:

* Total sellers
* Churn rate
* Revenue at risk
* Risk distribution
* Seller activity
* Geographic performance
* Probability distribution

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Category Encoders
* Matplotlib
* Seaborn
* SciPy
* Google Colab
* Tableau

---

## Project Structure

```
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── Final_Project_Alpha_Team_Seller_Churn.ipynb
│
├── dashboard/
│   ├── dashboard_seller_risk.csv
│   ├── dashboard_segment_summary.cs
│   └── dashboard_risk_summary.csv
│
├── README.md
│
└── requirements.txt
```

---

## 🚀 How to Run

1. Clone the repository.

```bash
git clone https://github.com/your-username/seller-churn-prediction.git
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Download the Olist dataset and place all CSV files inside the `data/` directory.

4. Open the notebook.

```
Final_Project_Alpha_Team_Seller_Churn.ipynb
```

5. Run all cells sequentially.

6. Generated dashboard-ready CSV files will be saved automatically.

---

## Example Outputs

The project generates:

* Seller churn probability
* Predicted churn label
* Risk segmentation
* Revenue at risk estimation
* Dashboard-ready datasets
* Model evaluation metrics
* Feature importance analysis

---

## Business Impact

The predictive model enables marketplace operators to:

* Identify high-risk sellers early.
* Prioritize retention campaigns.
* Allocate marketing resources more efficiently.
* Estimate potential revenue loss.
* Monitor seller health through interactive dashboards.

---

## Team

Alpha Team

---

## License

This project is intended for academic and educational purposes.
