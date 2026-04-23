import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import io
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# LLM IMPORTS
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Bias Analyzer", layout="wide")

# ------------------ STYLING ------------------
st.markdown("""
<style>
body { background-color: #0f172a; color: white; }
.big-title { font-size: 42px; font-weight: 800; text-align: center; }
.subtitle { text-align: center; color: #94a3b8; margin-bottom: 30px; }
.card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="big-title">AI Bias Detection Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enterprise-grade fairness analysis tool</div>', unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analyze"])

# ------------------ HOME ------------------
if page == "Home":
    st.markdown("""
    ### 🚀 What this tool does:
    - Detects bias in ML predictions  
    - Applies mitigation techniques  
    - Generates professional reports  
    """)

# =========================================================
# 🤖 LLM FUNCTION (CORE)
# =========================================================
def generate_llm_explanation(prompt):

    # ---------- 1. GEMINI ----------
    try:
        import google.generativeai as genai

        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)

        return "🟢 Gemini Response:\n\n" + response.text

    except Exception as e:
        st.warning(f"Gemini failed → {e}")

    # ---------- 2. GROQ ----------
    try:
        from groq import Groq

        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        chat = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        return "🟡 Groq Response:\n\n" + chat.choices[0].message.content

    except Exception as e:
        st.warning(f"Groq failed → {e}")

    # ---------- 3. LOCAL FALLBACK ----------
    return f"""
🔴 Local Explanation (Fallback):

The model shows bias between groups.

Possible reasons:
- Imbalanced dataset
- Sensitive attribute influence
- Model learning skewed patterns

Suggested mitigation:
- Remove sensitive feature
- Balance dataset
- Use fairness-aware algorithms
"""

# ------------------ ANALYSIS ------------------
if page == "Analyze":

    file = st.file_uploader("📂 Upload CSV Dataset")

    if file:
        df = pd.read_csv(file).dropna()
        st.dataframe(df.head())

        target = st.selectbox("🎯 Select Target Column", df.columns)
        sensitive = st.selectbox("⚠️ Select Sensitive Attribute", df.columns)

        if st.button("🚀 Run Analysis"):

            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            with st.spinner("Running AI analysis..."):

                # ------------------ MODEL ------------------
                if df[target].dtype == "object":
                    df[target] = df[target].astype("category").cat.codes

                X = df.drop(columns=[target])
                y = df[target]

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

                model = LogisticRegression(max_iter=5000, solver='liblinear')
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                df_test = df.loc[y_test.index].copy()
                df_test['pred'] = preds

                groups = df_test[sensitive].unique()

                if len(groups) < 2:
                    st.error("Sensitive column must have at least 2 groups")
                    st.stop()

                g1 = df_test[df_test[sensitive] == groups[0]]['pred'].mean()
                g2 = df_test[df_test[sensitive] == groups[1]]['pred'].mean()

                di_ratio = g2 / g1 if g1 != 0 else 0
                bias_score = abs(g1 - g2)

            st.divider()

            # ------------------ METRICS ------------------
            st.subheader("📊 Key Metrics")

            col1, col2, col3 = st.columns(3)
            col1.metric("Bias Score", round(bias_score, 2))
            col2.metric("Disparate Impact", round(di_ratio, 2))
            col3.metric("Group Gap", round(abs(g1 - g2), 2))

            # ------------------ CHART ------------------
            chart_df = pd.DataFrame({
                "Group": ["Group 1", "Group 2"],
                "Value": [g1, g2]
            })

            fig = px.bar(chart_df, x="Group", y="Value", color="Group")
            st.plotly_chart(fig, use_container_width=True)

            # ------------------ LLM INSIGHTS ------------------
            st.subheader("🤖 AI Insights")

            prompt = f"""
            Analyze bias in a machine learning model.

            Sensitive feature: {sensitive}
            Target: {target}

            Group 1 rate: {g1}
            Group 2 rate: {g2}

            Bias score: {bias_score}

            Explain:
            1. Cause of bias
            2. Impact
            3. How to fix it
            """

            explanation = generate_llm_explanation(prompt)

            st.write(explanation)
