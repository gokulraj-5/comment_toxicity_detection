import sys
import os
import streamlit as st

# ==========================
# ADD PROJECT ROOT
# ==========================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

# ==========================
# IMPORTS
# ==========================

from src.inference.predict import predict_comment

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Comment Toxicity Detection",
    page_icon="🛡️",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Metric cards */
div[data-testid="metric-container"] {
    background-color: #1f2937;
    border: 1px solid #374151;
    padding: 15px;
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# CONFIG
# ==========================

PREDICTION_THRESHOLD = 0.3

# ==========================
# HERO SECTION
# ==========================

st.markdown("""
<div style="
padding:30px;
border-radius:20px;
background: linear-gradient(135deg,#1e3c72,#2a5298);
color:white;
text-align:center;
margin-bottom:20px;
">
<h1>🛡️ Comment Toxicity Detection</h1>
<h3>AI-Powered Multi-Label Toxic Comment Classification</h3>
<p>
Detect Toxic, Severe Toxic, Obscene, Threat,
Insult and Identity Hate comments using Deep Learning
</p>
</div>
""", unsafe_allow_html=True)

# ==========================
# METRICS
# ==========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset", "159,571")

with col2:
    st.metric("ROC-AUC", "0.975")

with col3:
    st.metric("Classes", "6")

with col4:
    st.metric("Model", "Bi-LSTM")


# ==========================
# INPUT
# ==========================

st.subheader("💬 Enter a Comment")

comment = st.text_area(
    "",
    height=180,
    placeholder="Type a comment here..."
)

# ==========================
# PREDICTION
# ==========================

if st.button("🚀 Predict Toxicity"):

    if not comment.strip():

        st.warning("Please enter a comment.")

    else:

        with st.spinner("Analyzing comment..."):

            result = predict_comment(comment)

        # ==========================
        # RESULT
        # ==========================

        if result["prediction"] == "Toxic":

            st.markdown("""
            <div style="
            background:#dc2626;
            padding:15px;
            border-radius:12px;
            color:white;
            text-align:center;
            font-size:22px;
            font-weight:bold;
            ">
            ⚠️ TOXIC COMMENT DETECTED
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="
            background:#16a34a;
            padding:15px;
            border-radius:12px;
            color:white;
            text-align:center;
            font-size:22px;
            font-weight:bold;
            ">
            ✅ NON TOXIC COMMENT
            </div>
            """, unsafe_allow_html=True)

        # ==========================
        # DETECTED LABELS
        # ==========================

        st.subheader("🏷️ Detected Categories")

        if result["detected_labels"]:

            for label in result["detected_labels"]:
                st.error(f"⚠️ {label}")

        else:

            st.success("No toxic category detected.")

        # ==========================
        # SCORES
        # ==========================

        st.write("---")
        st.subheader("📊 Toxicity Scores")

        for label, score in result["scores"].items():

            st.write(f"**{label}** : {score:.4f}")

            st.progress(float(score))

# ==========================
# SAMPLE COMMENTS
# ==========================

st.write("---")

st.subheader("🧪 Sample Test Comments")

st.code("""
You are stupid.
Thank you for your support.
I will kill you.
This project is amazing.
People like you should not exist.
""")

# ==========================
# FOOTER
# ==========================

st.write("---")

st.caption(
    "Built using TensorFlow, Keras, Bi-LSTM and Streamlit"
)