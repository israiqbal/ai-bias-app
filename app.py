import streamlit as st
import pandas as pd
import plotly.express as px
import io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
        return model.generate_content(prompt).text
    except:
        return """
Bias detected.

Causes:
- Data imbalance
- Sensitive feature influence

Fix:
- Balance dataset
- Remove sensitive feature
- Use fairness-aware models
"""

# ---------------- ANALYZE ----------------
if page == "Analyze":

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)

        df = df.dropna()

        for col in df.select_dtypes(include="object"):
            df[col] = df[col].astype(str).str.strip()

        st.dataframe(df.head())

        target = st.selectbox("Target Column", df.columns)
        sensitive = st.selectbox("Sensitive Column", df.columns)

        if st.button("Run Analysis"):

            # -------- TARGET FIX --------
            if df[target].dtype == "object":
                df[target] = df[target].replace({
                    "<=50K": 0,
                    ">50K": 1,
                    "<=50K.": 0,
                    ">50K.": 1
                })
                df[target] = pd.to_numeric(df[target], errors="coerce")

            df = df.dropna(subset=[target])

            X = df.drop(columns=[target])
            y = df[target]

            st.write("Target Distribution:", y.value_counts())

            if len(y.unique()) < 2:
                st.error("Target must have at least 2 classes")
                st.stop()

            sensitive_series = df[sensitive].astype(str)

            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            scaler = StandardScaler(with_mean=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            # -------- CLEAN ANALYSIS DF --------
            sens_test = sensitive_series.loc[y_test.index]
            preds = pd.to_numeric(preds, errors="coerce")
            analysis_df = pd.DataFrame({
                "sensitive": sens_test.tolist(),
                "pred": [int(p) if pd.notna(p) else 0 for p in preds]
            })

            # -------- PURE PYTHON GROUPING (NO PANDAS MEAN) --------
            groups = analysis_df["sensitive"].unique()

            if len(groups) < 2:
                st.error("Sensitive column must have at least 2 groups")
                st.stop()

            g1_vals = analysis_df[analysis_df["sensitive"] == groups[0]]["pred"].tolist()
            g2_vals = analysis_df[analysis_df["sensitive"] == groups[1]]["pred"].tolist()

            if len(g1_vals) == 0 or len(g2_vals) == 0:
                st.error("Grouping failed")
                st.stop()

            g1 = sum(g1_vals) / len(g1_vals)
            g2 = sum(g2_vals) / len(g2_vals)

            bias = abs(g1 - g2)

            st.success("Analysis Complete")

            # -------- METRICS --------
            col1, col2 = st.columns(2)
            col1.metric("Bias Score", round(bias, 3))
            col2.metric("Group Difference", round(abs(g1 - g2), 3))

            # -------- CHART --------
            chart_df = pd.DataFrame({
                "Group": [groups[0], groups[1]],
                "Value": [g1, g2]
            })

            fig = px.bar(chart_df, x="Group", y="Value")
            st.plotly_chart(fig)

            # -------- LLM --------
            st.subheader("AI Insights")

            prompt = f"""
Bias detected in {sensitive}.

Group values:
{groups[0]}: {g1}
{groups[1]}: {g2}

Explain cause and mitigation.
"""

            explanation = generate_llm_explanation(prompt)
            st.write(explanation)

            # -------- PDF --------
            st.subheader("Download Report")

            def create_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()

                content = []
                content.append(Paragraph("AI Bias Report", styles['Title']))
                content.append(Spacer(1, 10))
                content.append(Paragraph(f"Feature: {sensitive}", styles['Normal']))
                content.append(Paragraph(f"Bias: {bias}", styles['Normal']))
                content.append(Spacer(1, 10))
                content.append(Paragraph(explanation, styles['Normal']))

                doc.build(content)
                buffer.seek(0)
                return buffer

            st.download_button(
                "Download Report",
                data=create_pdf(),
                file_name="bias_report.pdf"
            )
