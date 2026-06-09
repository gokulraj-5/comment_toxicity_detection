import streamlit as st
import pandas as pd

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

# ==========================
# HEADER
# ==========================

st.title("📊 Model Performance Dashboard")

st.markdown(
    """
    Evaluation results of the Bi-LSTM Toxic Comment Detection Model.
    """
)

# ==========================
# DATASET STATISTICS
# ==========================

st.subheader("📁 Dataset Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Samples",
        "159,571"
    )

with col2:
    st.metric(
        "Training Samples",
        "127,656"
    )

with col3:
    st.metric(
        "Validation Samples",
        "31,915"
    )

# ==========================
# MODEL INFO
# ==========================

st.write("---")

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Bi-LSTM"
    )

with col2:
    st.metric(
        "ROC-AUC",
        "0.9754"
    )

with col3:
    st.metric(
        "Threshold",
        "0.30"
    )

# ==========================
# CLASSIFICATION REPORT
# ==========================

st.write("---")

st.subheader("📋 Classification Report")

report_df = pd.DataFrame(
    {
        "Class": [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate"
        ],
        "Precision": [
            0.70,
            0.48,
            0.75,
            0.00,
            0.66,
            0.33
        ],
        "Recall": [
            0.85,
            0.43,
            0.87,
            0.00,
            0.78,
            0.02
        ],
        "F1 Score": [
            0.77,
            0.46,
            0.80,
            0.00,
            0.72,
            0.04
        ]
    }
)

st.dataframe(
    report_df,
    use_container_width=True
)

# ==========================
# CLASS DISTRIBUTION
# ==========================

st.write("---")

st.subheader("📈 Class Distribution")

class_distribution = pd.DataFrame(
    {
        "Class": [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate"
        ],
        "Count": [
            15294,
            1595,
            8449,
            478,
            7877,
            1405
        ]
    }
)

st.bar_chart(
    class_distribution.set_index("Class")
)

# ==========================
# OBSERVATIONS
# ==========================

st.write("---")

st.subheader("🔍 Key Observations")

st.success(
    """
    • Strong performance on Toxic, Obscene, and Insult classes.

    • Overall ROC-AUC of 0.9754 indicates excellent discrimination capability.

    • Threat and Identity Hate classes perform poorly due to severe class imbalance.

    • Lowering the prediction threshold from 0.5 to 0.3 improved minority class detection.
    """
)

# ==========================
# MODEL ARCHITECTURE
# ==========================

st.write("---")

st.subheader("🏗️ Model Architecture")

st.code(
    """
Embedding(10000, 128)
        ↓
Bidirectional(LSTM(64))
        ↓
Dropout(0.3)
        ↓
Dense(64, relu)
        ↓
Dense(6, sigmoid)
""",
    language="text"
)