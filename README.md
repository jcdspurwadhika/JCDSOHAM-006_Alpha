## **Seller Churn Prediction Dashboard**

### **Live Demo**

**Streamlit App:** https://sellerchurnapp.streamlit.app/

**Tableau:** https://public.tableau.com/shared/5TDB44K8S?:display_count=n&:origin=viz_share_link

**Notebook:** https://colab.research.google.com/drive/1Y-YefeFlTSCet0JfRT6wA3yPLnpXFPPx?usp=sharing

**Link Drive Dataset:** https://drive.google.com/drive/folders/1LSrczQg4LvAMexmQCYv1F-FKqmIqG41A?usp=drive_link
### **Overview**

This project deploys a machine learning model to predict whether an e-commerce seller is likely to churn within the next 90 days.

The application is built using **Streamlit** and allows users to:

- Predict churn risk for a single seller.
- Predict churn risk for multiple sellers through CSV upload.
- View the predicted churn probability.
- Classify sellers into risk tiers.
- Download prediction results.

### **Model**

The prediction model is a **Random Forest Classifier** trained on seller behavioral data.

The model was optimized using:

- Hyperparameter tuning
- F2-Score optimization
- Threshold optimization

The deployment uses the optimized decision threshold instead of the default 0.50 threshold.

### **Model Bundle**

The trained model is serialized using the `pickle` library.

The `seller_churn_model.pkl` file contains:

- Trained Random Forest model
- Input feature list
- Optimized decision threshold
- Model evaluation metrics
  - F2-Score
  - Recall
  - Precision
  - ROC-AUC

The original training dataset is **not** stored inside the pickle file.

### **Features**

The model uses the following seller-level features:

- recency_days
- tenure_days
- frequency_orders
- monetary_total
- monetary_avg_order
- avg_price
- avg_freight_value
- n_distinct_products
- n_distinct_categories
- avg_review_score
- late_delivery_rate
- cancellation_rate
- orders_per_month
- seller_state_encoded

### **Risk Classification**

Predicted churn probabilities are categorized into four business risk levels:

| Probability           | Risk Tier |
| --------------------- | --------- |
| < Optimized Threshold | Low       |
| Threshold – 0.49      | Medium    |
| 0.50 – 0.79           | High      |
| ≥ 0.80                | Critical  |

### **Running the Application Locally**

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

### **Project Structure**

```
seller-churn-streamlit/
│
├── app.py
├── seller_churn_model.pkl
├── requirements.txt
└── README.md
```

---

### **Disclaimer**

This application is intended to support seller retention decisions by providing predicted churn probabilities. Predictions should be used together with business judgment rather than as the sole basis for decision-making.
