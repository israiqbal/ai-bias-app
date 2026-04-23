import streamlit as st
import pandas as pd
import plotly.express as px
import io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Bias Analyzer", layout="wide")

st.title("AI Bias Detection Platform")

page = st.sidebar.radio("Go to", ["Home", "Analyze"])

# ---------------- HOME ----------------
if page == "Home":
    st.write("Upload dataset → Analyze bias → Generate insights")

# ---------------- LLM ----------------
def generate_llm_explanation(prompt):
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-pro")
        res = model.generate_content(prompt)
        return res.text
    except:
        return f"""
        Bias detected in the model.

        Possible causes:
        - Imbalanced dataset
        - Sensitive feature influence

        Suggested actions:
        - Remove or regularize sensitive features
        - Balance dataset
        - Use fairness-aware models
        """

# ---------------- ANALYZE ----------------
if page == "Analyze":

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)

        # CLEAN
        df = df.dropna()
        df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

        st.dataframe(df.head())

        target = st.selectbox("Target Column", df.columns)
        sensitive = st.selectbox("Sensitive Column", df.columns)

        if st.button("Run Analysis"):

            # -------- TARGET FIX --------
            if df[target].dtype == "object":
                df[target] = df[target].astype("category").cat.codes

            X = pd.get_dummies(df.drop(columns=[target]))
            y = df[target]

            if len(y.unique()) < 2:
                st.error("Target must have 2 classes")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            scaler = StandardScaler(with_mean=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = LogisticRegression(max_iter=5000, solver="liblinear")
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            # 🔥 FIX: ALIGN DATA PROPERLY
            df_test = df.iloc[y_test.index].copy()
            df_test["pred"] = preds

            # 🔥 FIX: SAFE GROUPING
            grouped = df_test.groupby(sensitive)["pred"].mean()

            if len(grouped) < 2:
                st.error("Sensitive column must have at least 2 groups")
                st.stop()

            g1, g2 = grouped.iloc[0], grouped.iloc[1]

            if pd.isna(g1) or pd.isna(g2):
                st.error("Computation failed. Try another sensitive column.")
                st.stop()

            bias = abs(g1 - g2)

            st.success("Analysis Complete")

            # -------- METRICS --------
            col1, col2 = st.columns(2)
            col1.metric("Bias Score", round(bias, 3))
            col2.metric("Group Difference", round(abs(g1 - g2), 3))

            # -------- CHART --------
            chart_df = grouped.reset_index()
            fig = px.bar(chart_df, x=sensitive, y="pred")
            st.plotly_chart(fig)

            # -------- LLM --------
            st.subheader("AI Insights")

            prompt = f"""
            Analyze bias in ML model.

            Sensitive feature: {sensitive}
            Values:
            {grouped.to_dict()}

            Explain cause, impact and mitigation.
            """

            explanation = generate_llm_explanation(prompt)
            st.write(explanation)

            # -------- PDF REPORT --------
            st.subheader("Download Report")

            def create_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()

                content = []
                content.append(Paragraph("AI Bias Report", styles['Title']))
                content.append(Spacer(1, 10))
                content.append(Paragraph(f"Sensitive Feature: {sensitive}", styles['Normal']))
                content.append(Paragraph(f"Bias Score: {bias}", styles['Normal']))
                content.append(Spacer(1, 10))
                content.append(Paragraph(explanation, styles['Normal']))

                doc.build(content)
                buffer.seek(0)
                return buffer

            pdf = create_pdf()

            st.download_button(
                "Download Report",
                data=pdf,
                file_name="bias_report.pdf"
            )
