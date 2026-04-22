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
body { background: linear-gradient(135deg,#0f172a,#020617); color:#e2e8f0;}
.title {font-size:42px;font-weight:800;text-align:center;}
.subtitle {text-align:center;color:#94a3b8;margin-bottom:25px;}
.card {
    background:rgba(30,41,59,0.6);
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ STATE ------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# ------------------ HEADER ------------------
st.markdown('<div class="title">AI Bias Detection Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect • Explain • Mitigate AI Bias</div>', unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Analyze", "📄 Report"])

# =========================================================
# 🏠 HOME
# =========================================================
if page == "🏠 Home":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><b>🎯 Objective</b><br>Detect bias in ML models</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><b>⚠️ Impact</b><br>Prevent unfair decisions</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card"><b>🚀 Applications</b><br>Finance, Hiring, Healthcare</div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Workflow")
    st.write("Upload → Analyze → Detect Bias → Mitigate → Generate Report")

# =========================================================
# 📊 ANALYSIS FUNCTION
# =========================================================
def run_analysis(df, target, sensitive):

    df[target] = df[target].apply(lambda x: 1 if ">50K" in str(x) else 0)

    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    df_test = df.loc[y_test.index].copy()
    df_test["pred"] = preds

    g1 = df_test[df_test[sensitive] == df[sensitive].unique()[0]]["pred"].mean()
    g2 = df_test[df_test[sensitive] == df[sensitive].unique()[1]]["pred"].mean()

    # mitigation
    X2 = pd.get_dummies(df.drop(columns=[target, sensitive]))
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y)

    model2 = LogisticRegression(max_iter=5000)
    model2.fit(X_train2, y_train2)
    preds2 = model2.predict(X_test2)

    df_test2 = df.loc[y_test2.index].copy()
    df_test2["pred"] = preds2

    g1_after = df_test2[df_test2[sensitive] == df[sensitive].unique()[0]]["pred"].mean()
    g2_after = df_test2[df_test2[sensitive] == df[sensitive].unique()[1]]["pred"].mean()

    return {
        "g1": g1, "g2": g2,
        "g1_after": g1_after, "g2_after": g2_after,
        "bias_before": abs(g1-g2),
        "bias_after": abs(g1_after-g2_after),
        "di": g2/g1 if g1 != 0 else 0,
        "target": target,
        "sensitive": sensitive
    }

# =========================================================
# 📊 ANALYZE PAGE
# =========================================================
if page == "📊 Analyze":

    file = st.file_uploader("Upload Dataset")

    if file:
        df = pd.read_csv(file).dropna()
        st.dataframe(df.head())

        target = st.selectbox("Target", df.columns)
        sensitive = st.selectbox("Sensitive Feature", df.columns)

        if st.button("Run Analysis"):

            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)

            result = run_analysis(df, target, sensitive)
            st.session_state.analysis = result

            st.toast("Analysis complete 🚀")

    # show results
    if st.session_state.analysis:
        r = st.session_state.analysis

        st.subheader("📊 Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bias Before", round(r["bias_before"],2))
        c2.metric("Bias After", round(r["bias_after"],2))
        c3.metric("Disparate Impact", round(r["di"],2))

        chart_df = pd.DataFrame({
            "Group":["Before G1","Before G2","After G1","After G2"],
            "Value":[r["g1"],r["g2"],r["g1_after"],r["g2_after"]]
        })

        fig = px.bar(chart_df,x="Group",y="Value",color="Group")
        st.plotly_chart(fig,use_container_width=True)

        # ---------------- LLM ----------------
        st.subheader("🤖 AI Insights")

        provider = st.selectbox("Model",["OpenAI","Claude"])

        if st.button("Generate Insights"):

            prompt = f"""
            Analyze bias in {r['sensitive']} for {r['target']}.
            Before: {r['g1']} vs {r['g2']}
            After: {r['g1_after']} vs {r['g2_after']}

            Provide:
            cause, mitigation steps, business explanation
            """

            try:
    if provider == "OpenAI":
        from openai import OpenAI

        if "OPENAI_API_KEY" not in st.secrets:
            st.error("OpenAI API key not found")
            st.stop()

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        res = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        # safer extraction
        output = res.output[0].content[0].text

    else:
        import anthropic

        if "ANTHROPIC_API_KEY" not in st.secrets:
            st.error("Claude API key not found")
            st.stop()

        client = anthropic.Anthropic(
            api_key=st.secrets["ANTHROPIC_API_KEY"]
        )

        res = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )

        # safer parsing
        output = res.content[0].text

    st.session_state.llm = output
    st.success("AI insights generated")
    st.write(output)

except Exception as e:
    st.error(f"API error: {str(e)}")
        if st.session_state.llm:
            st.markdown("### 💡 Insights")
            st.write(st.session_state.llm)

# =========================================================
# 📄 REPORT PAGE
# =========================================================
if page == "📄 Report":

    if not st.session_state.analysis:
        st.warning("Run analysis first")
    else:
        r = st.session_state.analysis

        st.header("📄 Report")

        st.write("Target:", r["target"])
        st.write("Sensitive:", r["sensitive"])

        st.write("Bias Before:", r["bias_before"])
        st.write("Bias After:", r["bias_after"])

        st.write("### 🤖 AI Insights")
        st.write(st.session_state.llm or "Generate insights first")

        # -------- PDF --------
        def create_pdf():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            content = []
            content.append(Paragraph("AI Bias Report", styles['Title']))

            table = Table([
                ["Metric","Before","After"],
                ["Group1",r["g1"],r["g1_after"]],
                ["Group2",r["g2"],r["g2_after"]]
            ])
            content.append(table)

            fig2, ax = plt.subplots()
            ax.bar(['B1','B2','A1','A2'],
                   [r["g1"],r["g2"],r["g1_after"],r["g2_after"]])
            fig2.savefig("chart.png")
            plt.close(fig2)

            content.append(Image("chart.png", width=300, height=150))

            content.append(Paragraph(st.session_state.llm or "", styles['Normal']))

            doc.build(content)
            buffer.seek(0)
            return buffer

        pdf = create_pdf()

        st.download_button("⬇️ Download Report", data=pdf, file_name="report.pdf")
