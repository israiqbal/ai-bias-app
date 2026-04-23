import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import io
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# LLM imports handled dynamically

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Bias Analyzer", layout="wide")

# ------------------ HEADER ------------------
st.title("AI Bias Detection Platform")
st.caption("Enterprise-grade fairness analysis tool")

# ------------------ SIDEBAR ------------------
page = st.sidebar.radio("Go to", ["Home", "Analyze"])

# ------------------ HOME ------------------
if page == "Home":
    st.write("Upload dataset → Analyze bias → Get AI insights")

# =========================================================
# 🤖 LLM FUNCTION
# =========================================================
def generate_llm_explanation(prompt):

    # GEMINI
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return "🟢 Gemini:\n\n" + response.text
    except Exception as e:
        st.warning(f"Gemini failed → {e}")

    # GROQ
    try:
        from groq import Groq
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        chat = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return "🟡 Groq:\n\n" + chat.choices[0].message.content
    except Exception as e:
        st.warning(f"Groq failed → {e}")

    # LOCAL
    return "🔴 Local fallback: Bias detected. Use fairness techniques."

# =========================================================
# 📊 ANALYSIS
# =========================================================
if page == "Analyze":

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file).dropna()
        st.dataframe(df.head())

        target = st.selectbox("Target Column", df.columns)
        sensitive = st.selectbox("Sensitive Column", df.columns)

        if st.button("Run Analysis"):

            with st.spinner("Running model..."):

                # TARGET FIX
                if df[target].dtype == "object":
                    df[target] = df[target].astype("category").cat.codes

                X = df.drop(columns=[target])
                y = df[target]

                # VALIDATION
                if len(y.unique()) < 2:
                    st.error("Target must have at least 2 classes")
                    st.stop()

                X = pd.get_dummies(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                scaler = StandardScaler(with_mean=False)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = LogisticRegression(max_iter=5000, solver="liblinear")
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                # FORCE NUMERIC (CRITICAL FIX)
                preds = pd.Series(preds).astype(float)

                df_test = df.loc[y_test.index].copy()
                df_test["pred"] = preds.values

                groups = df_test[sensitive].unique()

                if len(groups) < 2:
                    st.error("Sensitive column must have at least 2 groups")
                    st.stop()

                # SAFE MEAN FUNCTION
                def safe_mean(data):
                    return pd.to_numeric(data, errors='coerce').mean()

                g1_data = df_test[df_test[sensitive] == groups[0]]["pred"]
                g2_data = df_test[df_test[sensitive] == groups[1]]["pred"]

                if len(g1_data) == 0 or len(g2_data) == 0:
                    st.error("Group split failed. Choose different sensitive column.")
                    st.stop()

                g1 = safe_mean(g1_data)
                g2 = safe_mean(g2_data)

                bias_score = abs(g1 - g2)
                di_ratio = g2 / g1 if g1 != 0 else 0

            st.success("Analysis Complete")

            # METRICS
            col1, col2 = st.columns(2)
            col1.metric("Bias Score", round(bias_score, 2))
            col2.metric("Disparate Impact", round(di_ratio, 2))

            # CHART
            chart_df = pd.DataFrame({
                "Group": ["Group 1", "Group 2"],
                "Value": [g1, g2]
            })

            fig = px.bar(chart_df, x="Group", y="Value")
            st.plotly_chart(fig)

            # LLM
            st.subheader("AI Insights")

            prompt = f"""
            Bias analysis:
            Feature: {sensitive}
            Target: {target}
            Group1: {g1}
            Group2: {g2}
            Bias: {bias_score}
            Explain and suggest fixes.
            """

            explanation = generate_llm_explanation(prompt)
            st.write(explanation)
