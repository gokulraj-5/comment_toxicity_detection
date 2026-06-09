import sys
import os

# ==========================
# ADD PROJECT ROOT
# ==========================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

# ==========================
# IMPORTS
# ==========================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📈",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "train.csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ==========================
# HEADER
# ==========================

st.title("📈 Exploratory Data Analysis")

st.markdown(
    """
    Jigsaw Toxic Comment Dataset Analysis
    """
)

# ==========================
# DATASET OVERVIEW
# ==========================

st.subheader("📁 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Samples",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Features",
        df.shape[1]
    )

with col3:
    st.metric(
        "Target Labels",
        6
    )

# ==========================
# CLASS DISTRIBUTION
# ==========================

st.write("---")

st.subheader("📊 Class Distribution")

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

class_counts = df[labels].sum()

st.bar_chart(class_counts)

# ==========================
# TOXIC VS NON TOXIC
# ==========================

st.write("---")

st.subheader("☣️ Toxic vs Non-Toxic")

df["is_toxic"] = (
    df[labels].sum(axis=1) > 0
)

toxic_count = df["is_toxic"].sum()

non_toxic_count = (
    len(df) - toxic_count
)

fig, ax = plt.subplots(figsize=(5, 5))

ax.pie(
    [toxic_count, non_toxic_count],
    labels=[
        "Toxic",
        "Non Toxic"
    ],
    autopct="%1.1f%%"
)

st.pyplot(fig)

# ==========================
# COMMENT LENGTH
# ==========================

st.write("---")

st.subheader("📝 Comment Length Distribution")

df["comment_length"] = (
    df["comment_text"]
    .astype(str)
    .apply(len)
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    df["comment_length"],
    bins=50
)

ax.set_xlabel(
    "Comment Length"
)

ax.set_ylabel(
    "Frequency"
)

st.pyplot(fig)

# ==========================
# TOP WORDS
# ==========================

st.write("---")

st.subheader("🔤 Top 20 Most Frequent Words")

text = " ".join(
    df["comment_text"]
    .astype(str)
    .tolist()
)

words = text.lower().split()

word_counts = Counter(words)

top_words = word_counts.most_common(20)

top_df = pd.DataFrame(
    top_words,
    columns=[
        "Word",
        "Count"
    ]
)

st.dataframe(
    top_df,
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    top_df["Word"],
    top_df["Count"]
)

ax.set_xticklabels(
    top_df["Word"],
    rotation=45
)

st.pyplot(fig)

# ==========================
# WORD CLOUD
# ==========================

st.write("---")

st.subheader("☁️ Word Cloud")

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(text)

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.imshow(
    wordcloud,
    interpolation="bilinear"
)

ax.axis("off")

st.pyplot(fig)

# ==========================
# DATA PREVIEW
# ==========================

st.write("---")

st.subheader("🔍 Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)