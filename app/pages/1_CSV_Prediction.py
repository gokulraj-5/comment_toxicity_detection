import sys
import os
import pandas as pd
import streamlit as st

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
# IMPORT PREDICTION FUNCTION
# ==========================

from src.inference.predict import predict_comment

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="CSV Toxicity Prediction",
    page_icon="📂",
    layout="wide"
)

# ==========================
# HEADER
# ==========================

st.title("📂 CSV Toxicity Prediction")

st.markdown(
    """
    Upload a CSV file containing a column named
    **comment_text**.
    """
)

# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ==========================
# PROCESS FILE
# ==========================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("CSV Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # Check required column

        if "comment_text" not in df.columns:

            st.error(
                "CSV must contain a column named 'comment_text'"
            )

        else:

            if st.button("Run Prediction"):

                results = []

                progress_bar = st.progress(0)

                total_rows = len(df)

                for idx, comment in enumerate(df["comment_text"]):

                    prediction = predict_comment(
                        str(comment)
                    )

                    row = {
                        "prediction":
                        prediction["prediction"],

                        "detected_labels":
                        ", ".join(
                            prediction["detected_labels"]
                        )
                    }

                    # Add all scores

                    for label, score in prediction["scores"].items():

                        row[label] = round(
                            float(score),
                            4
                        )

                    results.append(row)

                    progress_bar.progress(
                        (idx + 1) / total_rows
                    )

                result_df = pd.concat(
                    [
                        df,
                        pd.DataFrame(results)
                    ],
                    axis=1
                )

                st.success(
                    "Prediction Completed Successfully"
                )

                st.subheader(
                    "Prediction Results"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                csv_file = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇ Download Results CSV",
                    data=csv_file,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )