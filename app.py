import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import spacy
import sqlite3

# ---------------------- Load spaCy ----------------------
nlp = spacy.load("en_core_web_sm")

def spacy_preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# ---------------------- Load Models ----------------------
vectorizer = joblib.load("tfidf_vectorizer.joblib")
lr_model = joblib.load("logistic_regression_model.joblib")
nb_model = joblib.load("naive_bayes_model.joblib")
svc_model = joblib.load("svc_model.joblib")

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="PolitiFact Analyzer", layout="wide")
mode = st.sidebar.radio("Choose Mode:", ["Fake vs Real Detector", "SQL Data Analysis Report", "Dataset Dashboard"])

# ==================== 1️⃣ Fake vs Real Detector ====================
if mode == "Fake vs Real Detector":
    st.title("🧠 PolitiFact True/False Prediction")
    input_option = st.radio("Select input type:", ["Single Statement", "Upload CSV"], index=0)

    if input_option == "Single Statement":
        text_input = st.text_area("Enter a political statement:")
        sub_option = st.radio("Output Type:", ["Accuracy & F1 Comparison", "Confidence Comparison"], key="single_nested")

        if st.button("Predict Statement"):
            if text_input.strip() == "":
                st.warning("Please enter a statement.")
            else:
                clean_text = spacy_preprocess(text_input)
                X_input = vectorizer.transform([clean_text])

                # Predictions
                pred_lr = lr_model.predict(X_input)[0]
                prob_lr = lr_model.predict_proba(X_input)[0][1] if hasattr(lr_model,"predict_proba") else 0
                pred_nb = nb_model.predict(X_input)[0]
                prob_nb = nb_model.predict_proba(X_input)[0][1] if hasattr(nb_model,"predict_proba") else 0
                pred_svc = svc_model.predict(X_input)[0]
                prob_svc = 0.9 if pred_svc==1 else 0.1

                st.subheader("Model Predictions")
                st.metric("Logistic Regression", "True" if pred_lr==1 else "False", f"{prob_lr*100:.1f}%")
                st.metric("Naive Bayes", "True" if pred_nb==1 else "False", f"{prob_nb*100:.1f}%")
                st.metric("SVC", "True" if pred_svc==1 else "False", f"{prob_svc*100:.1f}%")

                if sub_option == "Accuracy & F1 Comparison":
                    st.subheader("📊 Accuracy & F1 (SMOTE Balanced)")
                    # Full dataset
                    df_full = pd.read_csv("politifact_dataset.csv").dropna(subset=["statement","BinaryNumTarget"])
                    df_full["clean_statement"] = df_full["statement"].apply(spacy_preprocess)
                    X = vectorizer.transform(df_full["clean_statement"])
                    y = df_full["BinaryNumTarget"].astype(int)
                    smote = SMOTE(random_state=42)
                    X_res, y_res = smote.fit_resample(X, y)

                    y_pred_lr = lr_model.predict(X_res)
                    y_pred_nb = nb_model.predict(X_res)
                    y_pred_svc = svc_model.predict(X_res)

                    accuracies = {
                        "Logistic Regression": accuracy_score(y_res, y_pred_lr),
                        "Naive Bayes": accuracy_score(y_res, y_pred_nb),
                        "SVC": accuracy_score(y_res, y_pred_svc)
                    }
                    f1_scores = {
                        "Logistic Regression": f1_score(y_res, y_pred_lr),
                        "Naive Bayes": f1_score(y_res, y_pred_nb),
                        "SVC": f1_score(y_res, y_pred_svc)
                    }
                    st.bar_chart(pd.DataFrame({"Accuracy": accuracies,"F1-Score": f1_scores}))

                elif sub_option == "Confidence Comparison":
                    st.subheader("📊 Model Confidence for this Statement")
                    conf_scores = {"Logistic Regression": prob_lr, "Naive Bayes": prob_nb, "SVC": prob_svc}
                    st.bar_chart(conf_scores)

    elif input_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        sub_option = st.radio("Visualization Type:", ["Accuracy & F1 Comparison","Confidence Comparison"], key="csv_nested")

        if uploaded_file is not None:
            df_csv = pd.read_csv(uploaded_file)
            text_col = st.selectbox("Select text column:", df_csv.columns)
            target_col = st.selectbox("Select target column:", [None]+list(df_csv.columns))

            df_csv = df_csv.dropna(subset=[text_col])
            df_csv["clean_statement"] = df_csv[text_col].apply(spacy_preprocess)
            X_input = vectorizer.transform(df_csv["clean_statement"])

            # Predictions
            df_csv["LR_Prob"] = lr_model.predict_proba(X_input)[:,1]
            df_csv["NB_Prob"] = nb_model.predict_proba(X_input)[:,1]
            df_csv["SVC_Pred"] = svc_model.predict(X_input)
            df_csv["SVC_Prob"] = np.where(df_csv["SVC_Pred"]==1,0.9,0.1)

            st.dataframe(df_csv.head(10))

            if sub_option == "Confidence Comparison":
                st.subheader("📊 Average Model Confidence")
                conf_means = {"Logistic Regression": df_csv["LR_Prob"].mean(),
                              "Naive Bayes": df_csv["NB_Prob"].mean(),
                              "SVC": df_csv["SVC_Prob"].mean()}
                st.bar_chart(conf_means)

            if sub_option == "Accuracy & F1 Comparison" and target_col is not None:
                y_true = df_csv[target_col].astype(int)
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X_input, y_true)

                y_pred_lr = lr_model.predict(X_res)
                y_pred_nb = nb_model.predict(X_res)
                y_pred_svc = svc_model.predict(X_res)

                accuracies = {
                    "Logistic Regression": accuracy_score(y_res, y_pred_lr),
                    "Naive Bayes": accuracy_score(y_res, y_pred_nb),
                    "SVC": accuracy_score(y_res, y_pred_svc)
                }
                f1_scores = {
                    "Logistic Regression": f1_score(y_res, y_pred_lr),
                    "Naive Bayes": f1_score(y_res, y_pred_nb),
                    "SVC": f1_score(y_res, y_pred_svc)
                }
                st.subheader("📊 Accuracy & F1 Comparison (SMOTE Balanced)")
                st.bar_chart(pd.DataFrame({"Accuracy": accuracies,"F1-Score": f1_scores}))

# ==================== 2️⃣ SQL Data Analysis / Dataset Dashboard ====================
elif mode in ["SQL Data Analysis Report","Dataset Dashboard"]:
    st.title("📊 PolitiFact Dataset Dashboard / SQL Analysis")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        text_col = st.selectbox("Select text column:", df.columns)
        target_col = st.selectbox("Select target column:", [None]+list(df.columns))
        speaker_col = st.selectbox("Select speaker column:", df.columns)

        # Preprocess text column
        df[text_col] = df[text_col].astype(str)
        df["clean_text"] = df[text_col].apply(spacy_preprocess)

        # ---------------- SQL Queries ----------------
        conn = sqlite3.connect(":memory:")
        df.to_sql("dataset", conn, index=False, if_exists="replace")
        query_option = st.selectbox("Select Query:", [
            "Count Total Unique Speakers",
            "List All Unique Speakers",
            "Top 5 Speakers by Number of Statements",
            "Count True and False Statements per Speaker",
            "Count Statements by Rating",
            "Speakers with Only False Statements",
            "Top Speakers by True Statements"
        ])

        queries = {
            "Count Total Unique Speakers": f"SELECT COUNT(DISTINCT {speaker_col}) as total_unique_speakers FROM dataset;",
            "List All Unique Speakers": f"SELECT DISTINCT {speaker_col} as unique_speaker FROM dataset;",
            "Top 5 Speakers by Number of Statements": f"SELECT {speaker_col}, COUNT(*) as statement_count FROM dataset GROUP BY {speaker_col} ORDER BY statement_count DESC LIMIT 5;",
            "Count True and False Statements per Speaker": f"SELECT {speaker_col}, SUM({target_col}) as true_count, COUNT(*) - SUM({target_col}) as false_count FROM dataset GROUP BY {speaker_col};",
            "Count Statements by Rating": f"SELECT {target_col} as rating, COUNT(*) as count FROM dataset GROUP BY {target_col};",
            "Speakers with Only False Statements": f"SELECT {speaker_col} FROM dataset GROUP BY {speaker_col} HAVING SUM({target_col})=0;",
            "Top Speakers by True Statements": f"SELECT {speaker_col}, SUM({target_col}) as true_count FROM dataset GROUP BY {speaker_col} ORDER BY true_count DESC LIMIT 5;"
        }

        result = pd.read_sql_query(queries[query_option], conn)
        st.subheader(f"Result: {query_option}")
        st.dataframe(result)
        numeric_cols = [c for c in result.columns if "count" in c or "true_count" in c or "false_count" in c]
        if numeric_cols:
            st.bar_chart(result.set_index(result.columns[0])[numeric_cols[0]])

        # ---------------- Dataset ML Metrics ----------------
        if target_col is not None:
            st.subheader("📊 ML Metrics on Uploaded Dataset (SMOTE Balanced)")
            X = vectorizer.transform(df["clean_text"])
            y = df[target_col].astype(int)
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)

            y_pred_lr = lr_model.predict(X_res)
            y_pred_nb = nb_model.predict(X_res)
            y_pred_svc = svc_model.predict(X_res)

            accuracies = {
                "Logistic Regression": accuracy_score(y_res, y_pred_lr),
                "Naive Bayes": accuracy_score(y_res, y_pred_nb),
                "SVC": accuracy_score(y_res, y_pred_svc)
            }
            f1_scores = {
                "Logistic Regression": f1_score(y_res, y_pred_lr),
                "Naive Bayes": f1_score(y_res, y_pred_nb),
                "SVC": f1_score(y_res, y_pred_svc)
            }
            st.bar_chart(pd.DataFrame({"Accuracy": accuracies,"F1-Score": f1_scores}))
