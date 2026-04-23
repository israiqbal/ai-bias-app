import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Bias Analyzer", layout="wide")

st.title("AI Bias Detection Platform")

# ---------------- NAV ----------------
page = st.sidebar.radio("Go to", ["Home", "Analyze"])

# ---------------- HOME ----------------
if page == "Home":
    st.write("Upload dataset → Analyze bias → Get insights")

# ---------------- LLM ----------------
def generate_llm_explanation(prompt):
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-pro")
        res = model.generate_content(prompt)
        return res.text
    except:
        return "Local fallback: Bias detected. Improve fairness."

# ---------------- ANALYZE ----------------
if page == "Analyze":

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)

        st.dataframe(df.head())

        target = st.selectbox("Target Column", df.columns)
        sensitive = st.selectbox("Sensitive Column", df.columns)

        if st.button("Run Analysis"):

            with st.spinner("Processing..."):

                df = df.dropna()

                # -------- CLEAN STRINGS --------
                df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

                # -------- TARGET FIX --------
                if df[target].dtype == "object":
                    unique_vals = df[target].unique()

                    if len(unique_vals) == 2:
                        df[target] = df[target].map({
                            unique_vals[0]: 0,
                            unique_vals[1]: 1
                        })
                    else:
                        df[target] = df[target].astype("category").cat.codes

                X = df.drop(columns=[target])
                y = df[target]

                if len(y.unique()) < 2:
                    st.error("Target must have 2 classes")
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

                # -------- CRITICAL FIX --------
                df_test = df.loc[y_test.index].copy()

                df_test["pred"] = pd.to_numeric(
                    pd.Series(preds, index=y_test.index),
                    errors="coerce"
                )

                # -------- GROUP HANDLING --------
                groups = df_test[sensitive].dropna().unique()

                if len(groups) < 2:
                    st.error("Sensitive column must have at least 2 groups")
                    st.stop()

                g1_series = pd.to_numeric(
                    df_test[df_test[sensitive] == groups[0]]["pred"],
                    errors="coerce"
                )

                g2_series = pd.to_numeric(
                    df_test[df_test[sensitive] == groups[1]]["pred"],
                    errors="coerce"
                )

                if g1_series.empty or g2_series.empty:
                    st.error("Group split failed. Choose a better sensitive column.")
                    st.stop()

                g1 = g1_series.mean()
                g2 = g2_series.mean()

                bias = abs(g1 - g2)

            st.success("Analysis Complete")

            # -------- OUTPUT --------
            col1, col2 = st.columns(2)
            col1.metric("Bias Score", round(bias, 3))
            col2.metric("Group Difference", round(abs(g1 - g2), 3))

            chart_df = pd.DataFrame({
                "Group": ["Group 1", "Group 2"],
                "Value": [g1, g2]
            })

            fig = px.bar(chart_df, x="Group", y="Value")
            st.plotly_chart(fig)

            # -------- LLM --------
            st.subheader("AI Insights")

            prompt = f"""
            Bias detected in feature {sensitive}.
            Group1: {g1}, Group2: {g2}.
            Explain cause and mitigation.
            """

            st.write(generate_llm_explanation(prompt))
