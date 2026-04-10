import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Bias Analyzer", layout="wide")

# ----------------------------
# CUSTOM CSS (Premium UI)
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.big-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("⚙️ Navigation")
section = st.sidebar.radio("Go to", ["Home", "Upload & Analyze"])

# ----------------------------
# HEADER
# ----------------------------
st.markdown('<div class="big-title">AI Bias Detection Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enterprise-grade fairness analysis tool</div>', unsafe_allow_html=True)

# ----------------------------
# HOME PAGE
# ----------------------------
if section == "Home":
    st.write("""
    ### 🚀 What this tool does:
    - Detects bias in ML predictions  
    - Applies mitigation techniques  
    - Generates professional reports  
    """)

# ----------------------------
# MAIN ANALYSIS
# ----------------------------
if section == "Upload & Analyze":

    file = st.file_uploader("Upload CSV Dataset")

    if file:
        df = pd.read_csv(file).dropna()
        st.dataframe(df.head())

        target = st.selectbox("Target Column", df.columns)
        sensitive = st.selectbox("Sensitive Attribute", df.columns)

        if st.button("Run Analysis"):

            with st.spinner("Analyzing bias... 🔍"):
                
                # ----------------------------
                # MODEL
                # ----------------------------
                df[target] = df[target].apply(lambda x: 1 if ">50K" in str(x) else 0)

                X = df.drop(columns=[target])
                y = df[target]

                X_encoded = pd.get_dummies(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.2, random_state=42
                )

                scaler = StandardScaler(with_mean=False)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = LogisticRegression(max_iter=5000, solver='liblinear')
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                df_test = df.loc[y_test.index].copy()
                df_test['pred'] = preds

                g1 = df_test[df_test[sensitive] == df[sensitive].unique()[0]]['pred'].mean()
                g2 = df_test[df_test[sensitive] == df[sensitive].unique()[1]]['pred'].mean()

                di_ratio = g2 / g1

                # ----------------------------
                # MITIGATION
                # ----------------------------
                X2 = df.drop(columns=[target, sensitive])
                y2 = df[target]

                X2 = pd.get_dummies(X2)

                X_train2, X_test2, y_train2, y_test2 = train_test_split(
                    X2, y2, test_size=0.2, random_state=42
                )

                idx = X_test2.index

                scaler2 = StandardScaler(with_mean=False)
                X_train2 = scaler2.fit_transform(X_train2)
                X_test2 = scaler2.transform(X_test2)

                model2 = LogisticRegression(max_iter=5000, solver='liblinear')
                model2.fit(X_train2, y_train2)
                preds2 = model2.predict(X_test2)

                df_test2 = df.loc[idx].copy()
                df_test2['pred'] = preds2

                g1_after = df_test2[df_test2[sensitive] == df[sensitive].unique()[0]]['pred'].mean()
                g2_after = df_test2[df_test2[sensitive] == df[sensitive].unique()[1]]['pred'].mean()

            # ----------------------------
            # METRICS DISPLAY
            # ----------------------------
            col1, col2 = st.columns(2)
            col1.metric("Before Bias (Group1)", round(g1,2))
            col2.metric("Before Bias (Group2)", round(g2,2))

            col3, col4 = st.columns(2)
            col3.metric("After Bias (Group1)", round(g1_after,2))
            col4.metric("After Bias (Group2)", round(g2_after,2))

            # ----------------------------
            # STATUS
            # ----------------------------
            if di_ratio < 0.8:
                st.error("⚠️ High Bias Detected")
            else:
                st.success("✅ Fair Model")

            # ----------------------------
            # CHARTS
            # ----------------------------
            fig, ax = plt.subplots()
            ax.bar(['Before G1','Before G2'], [g1, g2])
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.bar(['After G1','After G2'], [g1_after, g2_after])
            st.pyplot(fig2)

            # ----------------------------
            # PDF REPORT
            # ----------------------------
            def create_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()

                content = []

                content.append(Paragraph("AI Bias Report", styles['Title']))
                content.append(Spacer(1, 10))

                content.append(Paragraph(f"Before: {g1:.2f} vs {g2:.2f}", styles['Normal']))
                content.append(Paragraph(f"After: {g1_after:.2f} vs {g2_after:.2f}", styles['Normal']))

                # Save chart
                fig.savefig("chart.png")
                content.append(Image("chart.png", width=400, height=200))

                doc.build(content)
                buffer.seek(0)
                return buffer

            pdf = create_pdf()

            st.download_button(
                label="📄 Download Premium Report",
                data=pdf,
                file_name="bias_report.pdf",
                mime="application/pdf"
            )
