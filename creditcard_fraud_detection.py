import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load Model
# -------------------------------
saved_artifacts = joblib.load("fraud_detection_xgb_model.pkl")  # update to your model file
model = saved_artifacts["model"]

# -------------------------------
# Define expected feature order
# -------------------------------
EXPECTED_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

def prepare_input(df: pd.DataFrame, expected_cols: list, fill_value=0.0):
    """
    Reorders and fills missing columns to match the training feature order.
    Any extra columns are ignored, and missing ones are filled with `fill_value`.
    """
    # Identify missing and extra columns
    missing = [col for col in expected_cols if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_cols]

    # Create a clean version of df with expected columns only
    clean_df = df.copy()

    # Drop unexpected columns
    if extra:
        clean_df = clean_df.drop(columns=extra)

    # Add any missing columns and fill with default
    for col in missing:
        clean_df[col] = fill_value

    # Reorder columns
    clean_df = clean_df[expected_cols]

    return clean_df, missing, extra


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💳 Real-Time Fraud Detection App")
st.write("Upload a CSV of transaction data to classify into **Fraud**, **Review**, or **Clear** categories.")

uploaded_file = st.file_uploader("📂 Upload transaction CSV file", type=["csv"])

# -------------------------------
# Main Logic
# -------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # ✅ Clean and align columns automatically
    prepared_data, missing_cols, extra_cols = prepare_input(data, EXPECTED_COLS)

    if missing_cols:
        st.warning(f"⚠️ Missing columns were filled with 0: {', '.join(missing_cols)}")
    if extra_cols:
        st.info(f"ℹ️ Ignored extra columns: {', '.join(extra_cols)}")

    # -------------------------------
    # Fraud probability prediction
    # -------------------------------
    probs = model.predict_proba(prepared_data)[:, 1]

    # Thresholds
    high_thresh = 0.9  # confident fraud
    low_thresh = 0.7   # uncertain / human review

    # Apply decision logic
    data["Fraud_Probability"] = probs
    data["Fraud_Flag"] = np.select(
        [
            data["Fraud_Probability"] >= high_thresh,
            (data["Fraud_Probability"] >= low_thresh) & (data["Fraud_Probability"] < high_thresh),
            data["Fraud_Probability"] < low_thresh
        ],
        ["🚨 FRAUD", "🕵️ REVIEW", "✅ CLEAR"],
        default="✅ CLEAR"
    )

    # -------------------------------
    # Display results
    # -------------------------------
    st.success("✅ Prediction complete!")

    st.subheader("📊 Summary of Flags")
    summary = data["Fraud_Flag"].value_counts().reset_index()
    summary.columns = ["Flag Category", "Count"]
    st.dataframe(summary)

    st.subheader("🔍 Sample Predictions")
    st.dataframe(data[["Amount", "Fraud_Probability", "Fraud_Flag"]].head(10))

    # Option to download
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Full Results as CSV",
        data=csv,
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )
