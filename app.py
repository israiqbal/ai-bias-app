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

# ------------------ ANALYSIS ------------------
if page == "Analyze":

    file = st.file_uploader("📂 Upload CSV Dataset")

    if file:
        df = pd.read_csv(file).dropna()
        st.dataframe(df.head())

        target = st.selectbox("🎯 Select Target Column", df.columns)
        sensitive = st.selectbox("⚠️ Select Sensitive Attribute", df.columns)

        if st.button("🚀 Run Analysis"):

            # Animation
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            with st.spinner("Running AI analysis..."):

                # ------------------ MODEL ------------------
                df[target] = df[target].apply(lambda x: 1 if ">50K" in str(x) else 0)

                X = df.drop(columns=[target])
                y = df[target]

                X = pd.get_dummies(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
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
                bias_score = abs(g1 - g2)

                # ------------------ MITIGATION ------------------
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

            st.divider()

            # ------------------ METRICS ------------------
            st.subheader("📊 Key Metrics")

            col1, col2, col3 = st.columns(3)
            col1.metric("Bias Score", round(bias_score, 2))
            col2.metric("Disparate Impact", round(di_ratio, 2))
            col3.metric("Improvement", round(abs(g1 - g2) - abs(g1_after - g2_after), 2))

            # ------------------ INTERACTIVE CHART ------------------
            st.subheader("📈 Interactive Comparison")

            chart_df = pd.DataFrame({
                "Group": ["Before G1", "Before G2", "After G1", "After G2"],
                "Value": [g1, g2, g1_after, g2_after]
            })

            fig = px.bar(chart_df, x="Group", y="Value", color="Group")
            st.plotly_chart(fig, use_container_width=True)

            # ------------------ STATUS ------------------
            if di_ratio < 0.8:
                st.error("⚠️ High Bias Detected")
            else:
                st.success("✅ Model is Fair")

            # ------------------ PDF ------------------
            def create_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AI Bias Analysis Report", styles['Title']))
    content.append(Spacer(1, 15))

    table_data = [
        ["Metric", "Before", "After"],
        ["Group 1", f"{g1:.2f}", f"{g1_after:.2f}"],
        ["Group 2", f"{g2:.2f}", f"{g2_after:.2f}"]
    ]

    table = Table(table_data)
    table.setStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ])

    # ✅ FIXED INDENTATION
    content.append(table)

    # ✅ matplotlib chart (INSIDE function)
    fig2, ax = plt.subplots()

    ax.bar(
        ['Before G1','Before G2','After G1','After G2'],
        [g1, g2, g1_after, g2_after],
        color=['#3b82f6', '#ef4444', '#10b981', '#f59e0b']
    )

    ax.set_title("Bias Comparison")
    ax.set_ylabel("Prediction Rate")

    fig2.savefig("chart.png")
    plt.close(fig2)

    content.append(Spacer(1, 20))
    content.append(Image("chart.png", width=400, height=250))

    # Build PDF
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

            st.markdown("---")
            st.markdown("Built with ❤️ by Isra Iqbal")
