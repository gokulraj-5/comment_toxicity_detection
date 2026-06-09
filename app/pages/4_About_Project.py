import streamlit as st

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="About Project",
    page_icon="ℹ️",
    layout="wide"
)

# ==========================
# HEADER
# ==========================

st.title("ℹ️ About Project")

st.markdown(
    """
    A Deep Learning based Multi-Label Toxic Comment Detection System
    built using TensorFlow, Keras, Bi-LSTM and Streamlit.
    """
)

# ==========================
# PROJECT OVERVIEW
# ==========================

st.write("---")

st.header("📌 Project Overview")

st.write(
    """
    Comment Toxicity Detection is a Natural Language Processing (NLP)
    project that automatically identifies toxic comments and classifies
    them into multiple toxicity categories.

    The system helps online platforms detect harmful content and
    improve moderation efficiency.
    """
)

# ==========================
# PROBLEM STATEMENT
# ==========================

st.write("---")

st.header("🎯 Problem Statement")

st.write(
    """
    Online communities generate millions of comments every day.
    Manually reviewing comments is time-consuming and expensive.

    The goal of this project is to automatically identify toxic
    comments and categorize them into different toxicity classes
    using Deep Learning.
    """
)

# ==========================
# DATASET
# ==========================

st.write("---")

st.header("📁 Dataset")

st.write(
    """
    Dataset Used:
    
    Jigsaw Toxic Comment Classification Dataset
    """
)

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

st.subheader("Target Labels")

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

for label in labels:
    st.write(f"• {label}")

# ==========================
# TECHNOLOGIES
# ==========================

st.write("---")

st.header("🛠️ Technologies Used")

tech_stack = [
    "Python",
    "TensorFlow",
    "Keras",
    "Pandas",
    "NumPy",
    "Scikit-Learn",
    "Streamlit",
    "Matplotlib",
    "WordCloud"
]

for tech in tech_stack:
    st.write(f"✅ {tech}")

# ==========================
# MODEL ARCHITECTURE
# ==========================

st.write("---")

st.header("🏗️ Model Architecture")

st.code(
"""
Embedding(10000, 128)
        ↓
Bidirectional(LSTM(64))
        ↓
Dropout(0.3)
        ↓
Dense(64, ReLU)
        ↓
Dense(6, Sigmoid)
""",
language="text"
)

# ==========================
# MODEL PERFORMANCE
# ==========================

st.write("---")

st.header("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "ROC-AUC Score",
        "0.9754"
    )

with col2:
    st.metric(
        "Prediction Threshold",
        "0.30"
    )

st.success(
    """
    Strong Performance:
    
    • Toxic
    
    • Obscene
    
    • Insult
    """
)

st.warning(
    """
    Challenging Classes:
    
    • Threat
    
    • Identity Hate
    
    Reason:
    Severe class imbalance in the dataset.
    """
)

# ==========================
# FEATURES
# ==========================

st.write("---")

st.header("✨ Features")

features = [
    "Single Comment Prediction",
    "CSV Batch Prediction",
    "Toxicity Score Visualization",
    "Model Performance Dashboard",
    "EDA Dashboard",
    "Download Prediction Results",
    "Multi-Label Classification"
]

for feature in features:
    st.write(f"✅ {feature}")

# ==========================
# FUTURE IMPROVEMENTS
# ==========================

st.write("---")

st.header("🚀 Future Improvements")

future_improvements = [
    "BERT Based Classification",
    "DistilBERT Integration",
    "Class Weighting",
    "Focal Loss",
    "REST API Deployment",
    "Real-Time Comment Monitoring",
    "Improved Minority Class Detection"
]

for item in future_improvements:
    st.write(f"🔹 {item}")

# ==========================
# FOOTER
# ==========================

st.write("---")

st.caption(
    "Comment Toxicity Detection System | Deep Learning Project"
)