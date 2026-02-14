import streamlit as st
import os
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Adult Income Classification", layout="wide")
st.title("Adult Income Classification App")

st.subheader("Download Adult Income Test Dataset")
if os.path.exists("model/test.csv"):
    with open("model/test.csv", "rb") as t:
        st.download_button(label="Download test.csv", data=t, file_name="test.csv", mime="text/csv")
else:
    st.warning("Test dataset not found!")
    
st.subheader("Upload a CSV file to evaluate different trained models.")

# Upload CSV
uploaded_file = st.file_uploader("Upload Adult Dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Basic cleaning
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Encode categorical columns
    encoders = joblib.load("model/encoders.pkl")
    for col in df.columns:
        if df[col].dtype == "object":
            le = encoders[col]
            df[col] = df[col].map(lambda x : x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    # Separate features and target
    X = df.drop("income", axis=1)
    y = df["income"]


    # Load scaler
    scaler = joblib.load("model/scaler.pkl")
    X = scaler.transform(X)

    st.subheader("Models")

    # Model selection dropdown
    model_choice = st.selectbox(
        "Select Model",
        ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
    )

    # Load selected model
    model = joblib.load(f"model/{model_choice}.pkl")

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        y_prob = None
        auc = None

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1:.4f}")
    col5.metric("MCC", f"{mcc:.4f}")
    col6.metric("AUC", f"{auc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(cm)

else:
    st.info("Please upload a CSV file to proceed.")