import pickle
import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Seller Churn Predictor",
    layout="wide",
)

# ---------------------------------------------------------------
# Load model bundle
# ---------------------------------------------------------------
@st.cache_resource
def load_model_bundle(path="seller_churn_model.pkl"):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_model_bundle()
model = bundle["model"]
FEATURES = bundle["features"]
OPTIMAL_THRESHOLD = float(bundle["optimal_threshold"])
F2_SCORE = bundle.get("f2_score")
RECALL = bundle.get("recall")
PRECISION = bundle.get("precision")
ROC_AUC = bundle.get("roc_auc")

STATE_LIST = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
]
STATE_TO_CODE = {state: idx for idx, state in enumerate(STATE_LIST)}

FEATURE_LABELS = {
    "recency_days": "Recency (days since last order)",
    "tenure_days": "Tenure (days since first sale)",
    "frequency_orders": "Total number of orders",
    "monetary_total": "Total revenue (R$)",
    "monetary_avg_order": "Average order value (R$)",
    "avg_price": "Average product price (R$)",
    "avg_freight_value": "Average freight value (R$)",
    "n_distinct_products": "Number of distinct products sold",
    "n_distinct_categories": "Number of distinct categories sold",
    "avg_review_score": "Average review score (1-5)",
    "late_delivery_rate": "Late delivery rate (0-1)",
    "cancellation_rate": "Cancellation rate (0-1)",
    "orders_per_month": "Orders per month",
    "seller_state_encoded": "Seller state",
}

FEATURE_IMPORTANCE = dict(zip(FEATURES, model.feature_importances_)) if hasattr(model, "feature_importances_") else {}

def risk_tier(prob: float) -> tuple[str, str]:
    if prob >= 0.80:
        return "🔴 Critical", "#c0392b"
    elif prob >= 0.50:
        return "🟠 High", "#e67e22"
    elif prob >= OPTIMAL_THRESHOLD:
        return "🟡 Medium", "#f1c40f"
    else:
        return "🟢 Low", "#2ecc71"

def make_gauge(prob: float) -> go.Figure:
    tier_label, color = risk_tier(prob)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": tier_label},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, OPTIMAL_THRESHOLD * 100], "color": "#eafaf1"},
                    {"range": [OPTIMAL_THRESHOLD * 100, 50], "color": "#fcf3cf"},
                    {"range": [50, 80], "color": "#fdebd0"},
                    {"range": [80, 100], "color": "#fdedec"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.85,
                    "value": OPTIMAL_THRESHOLD * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def build_feature_row(inputs: dict) -> pd.DataFrame:
    row = {f: inputs[f] for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("Seller Churn Predictor")
st.markdown("""
Predict the probability that a seller will churn within the next **90 days**
using a Random Forest classification model trained on historical seller behavior.
""")

tab1, tab2 = st.tabs(["Single Seller Prediction", "Batch Prediction (CSV)"])

# ---------------------------------------------------------------
# TAB 1 — Single prediction
# ---------------------------------------------------------------
with tab1:
    st.subheader("Enter seller information")

    left, right = st.columns(2)

    with left:
        recency_days = st.number_input(FEATURE_LABELS["recency_days"], min_value=0, value=90, step=1)
        tenure_days = st.number_input(FEATURE_LABELS["tenure_days"], min_value=0, value=365, step=1)
        frequency_orders = st.number_input(FEATURE_LABELS["frequency_orders"], min_value=0, value=20, step=1)
        monetary_total = st.number_input(FEATURE_LABELS["monetary_total"], min_value=0.0, value=2000.0, step=50.0)
        monetary_avg_order = st.number_input(FEATURE_LABELS["monetary_avg_order"], min_value=0.0, value=100.0, step=10.0)
        avg_price = st.number_input(FEATURE_LABELS["avg_price"], min_value=0.0, value=80.0, step=5.0)
        orders_per_month = st.number_input(FEATURE_LABELS["orders_per_month"], min_value=0.0, value=2.0, step=0.1)

    with right:
        avg_freight_value = st.number_input(FEATURE_LABELS["avg_freight_value"], min_value=0.0, value=20.0, step=1.0)
        n_distinct_products = st.number_input(FEATURE_LABELS["n_distinct_products"], min_value=0, value=15, step=1)
        n_distinct_categories = st.number_input(FEATURE_LABELS["n_distinct_categories"], min_value=0, value=3, step=1)
        avg_review_score = st.slider(FEATURE_LABELS["avg_review_score"], 1.0, 5.0, 4.0, 0.1)
        late_delivery_rate = st.slider(FEATURE_LABELS["late_delivery_rate"], 0.0, 1.0, 0.05, 0.01)
        cancellation_rate = st.slider(FEATURE_LABELS["cancellation_rate"], 0.0, 1.0, 0.02, 0.01)
        seller_state = st.selectbox(FEATURE_LABELS["seller_state_encoded"], STATE_LIST, index=STATE_LIST.index("SP"))

    predict_clicked = st.button("Predict churn risk", type="primary")

    if predict_clicked:
        inputs = {
            "recency_days": recency_days,
            "tenure_days": tenure_days,
            "frequency_orders": frequency_orders,
            "monetary_total": monetary_total,
            "monetary_avg_order": monetary_avg_order,
            "avg_price": avg_price,
            "avg_freight_value": avg_freight_value,
            "n_distinct_products": n_distinct_products,
            "n_distinct_categories": n_distinct_categories,
            "avg_review_score": avg_review_score,
            "late_delivery_rate": late_delivery_rate,
            "cancellation_rate": cancellation_rate,
            "orders_per_month": orders_per_month,
            "seller_state_encoded": STATE_TO_CODE[seller_state],
        }

        X = build_feature_row(inputs)
        prob = model.predict_proba(X)[0, 1]
        pred = int(prob >= OPTIMAL_THRESHOLD)
        tier_label, _ = risk_tier(prob)

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.plotly_chart(make_gauge(prob), use_container_width=True)
        with res_col2:
            st.metric("Churn probability", f"{prob:.1%}")
            st.metric("Predicted label", "Churn" if pred == 1 else "Retained")
            st.write(f"**Risk tier:** {tier_label}")
            if pred == 1:
                st.warning(
                    "This seller is predicted to churn. Consider proactive retention "
                    "outreach — retention is typically cheaper than acquiring a new seller."
                )
            else:
                st.success("This seller is predicted to stay active.")

        if FEATURE_IMPORTANCE:
            st.subheader("What drives this model's predictions (SHAP)")
            imp_df = (
                pd.DataFrame(
                    {"feature": list(FEATURE_IMPORTANCE.keys()), "importance": list(FEATURE_IMPORTANCE.values())}
                )
                .sort_values("importance", ascending=True)
            )
            imp_df["label"] = imp_df["feature"].map(FEATURE_LABELS)
            fig = go.Figure(go.Bar(x=imp_df["importance"], y=imp_df["label"], orientation="h"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Feature importance")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# TAB 2 — Batch prediction
# ---------------------------------------------------------------
with tab2:
    st.subheader("Upload a CSV to score many sellers at once")
    st.markdown(
        "The file must contain the following columns "
        f"(`seller_state` as a 2-letter code, e.g. `SP`, `RJ`):\n\n"
        f"`{', '.join([f for f in FEATURES if f != 'seller_state_encoded'] + ['seller_state'])}`"
    )
    st.markdown(
        "Download the template below, fill in the seller data, and upload it for batch prediction."
    )

    # -------------------------------
    # CSV Template
    # -------------------------------
    template = pd.DataFrame(columns=[
        "recency_days",
        "tenure_days",
        "frequency_orders",
        "monetary_total",
        "monetary_avg_order",
        "avg_price",
        "avg_freight_value",
        "n_distinct_products",
        "n_distinct_categories",
        "avg_review_score",
        "late_delivery_rate",
        "cancellation_rate",
        "orders_per_month",
        "seller_state",
    ])

    st.download_button(
        label="Download CSV Template",
        data=template.to_csv(index=False),
        file_name="seller_churn_template.csv",
        mime="text/csv",
    )
    
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            missing = [f for f in FEATURES if f != "seller_state_encoded" and f not in df.columns]
            if "seller_state" not in df.columns and "seller_state_encoded" not in df.columns:
                missing.append("seller_state (or seller_state_encoded)")

            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                df_scored = df.copy()

                if "seller_state_encoded" not in df_scored.columns:
                    unknown_states = set(df_scored["seller_state"].astype(str).str.upper()) - set(STATE_LIST)
                    if unknown_states:
                        st.warning(f"Unrecognized state codes will be dropped: {sorted(unknown_states)}")
                    df_scored["seller_state_encoded"] = (
                        df_scored["seller_state"].astype(str).str.upper().map(STATE_TO_CODE)
                    )

                valid_rows = df_scored[FEATURES].notna().all(axis=1)
                if (~valid_rows).any():
                    st.warning(f"{(~valid_rows).sum()} row(s) have missing/unmapped values and will be skipped.")

                X_batch = df_scored.loc[valid_rows, FEATURES]
                probs = model.predict_proba(X_batch)[:, 1]
                preds = (probs >= OPTIMAL_THRESHOLD).astype(int)

                df_scored = df_scored.loc[valid_rows].copy()
                df_scored["churn_probability"] = probs
                df_scored["predicted_churn"] = preds
                df_scored["risk_tier"] = df_scored["churn_probability"].apply(lambda p: risk_tier(p)[0])

                st.success(f"Scored {len(df_scored)} sellers.")

                c1, c2, c3 = st.columns(3)
                c1.metric("High risk", int((df_scored["churn_probability"] >= 0.7).sum()))
                c2.metric("Medium risk", int(((df_scored["churn_probability"] >= OPTIMAL_THRESHOLD) & (df_scored["churn_probability"] < 0.7)).sum()))
                c3.metric("Low risk", int((df_scored["churn_probability"] < OPTIMAL_THRESHOLD).sum()))

                st.dataframe(df_scored, use_container_width=True)

                csv_buffer = io.StringIO()
                df_scored.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download scored CSV",
                    data=csv_buffer.getvalue(),
                    file_name="seller_churn_predictions.csv",
                    mime="text/csv",
                )

st.divider()
st.caption(
    "This tool provides predicted churn probabilities to support seller retention decisions. "
    "Predictions should be used alongside business judgment."
)