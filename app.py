import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import io, requests, time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet


# ================== CONFIG ==================
st.set_page_config(page_title="AI Bias SaaS Platform", layout="wide")


# ================== UI STYLE ==================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: #e2e8f0;
}
.section { margin-top: 60px; }
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.08);
}
.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
}
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ================== SESSION ==================
if "user" not in st.session_state:
    st.session_state.user = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ================== AUTH ==================
st.sidebar.title("🔐 Account")
mode = st.sidebar.radio("Mode", ["Login", "Signup"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if mode == "Signup":
    if st.sidebar.button("Create Account"):
        try:
            supabase.auth.sign_up({"email": email, "password": password})
            st.sidebar.success("Account created. Please login.")
        except Exception as e:
            st.sidebar.error(str(e))

if mode == "Login":
    if st.sidebar.button("Login"):
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state.user = res.user
            st.rerun()
        except Exception as e:
            st.sidebar.error("Login failed")

user = st.session_state.user
if not user:
    st.stop()

st.sidebar.success(f"Logged in: {user.email}")
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.llm = None
    st.session_state.analysis = None
    st.rerun()

# ================== HEADER ==================
st.markdown('<div class="main-title">AI Bias Detection Platform</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#94a3b8;'>Fair • Explainable • Reliable AI</div>", unsafe_allow_html=True)

# ================== PROMPT ==================
def build_prompt(target, sensitive, g1, g2, g1_after, g2_after, data_summary):
    return f"""
You are a senior AI fairness auditor.

DATASET SUMMARY:
{data_summary}

TARGET: {target}
SENSITIVE ATTRIBUTE: {sensitive}

BEFORE MITIGATION:
Group 1 rate: {g1:.3f}
Group 2 rate: {g2:.3f}
Bias gap: {abs(g1-g2):.3f}

AFTER MITIGATION:
Group 1 rate: {g1_after:.3f}
Group 2 rate: {g2_after:.3f}
Bias gap: {abs(g1_after-g2_after):.3f}

Provide:
1. Executive Summary
2. Root Cause
3. Risk Level
4. Mitigation Strategies
5. Business Impact
6. Final Recommendation
"""

# ================== LLM ==================
def generate_llm(prompt):
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-pro")
        return model.generate_content(prompt).text, "Gemini"
    except:
        st.warning("Gemini failed → Groq fallback")

    try:
        from groq import Groq
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        res = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content, "Groq"
    except:
        st.warning("Groq failed → Local fallback")

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        return res.json()["response"], "Local"
    except:
        return "All LLM providers failed", "None"

# ================== ABOUT ==================
st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.write("Detect bias, explain with AI, and generate actionable reports.")
st.markdown('</div>', unsafe_allow_html=True)

# ================== ANALYSIS ==================
st.markdown('<div class="section-title">Project</div>', unsafe_allow_html=True)
st.markdown('<div class="glass">', unsafe_allow_html=True)

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file).dropna()
    st.dataframe(df.head())

    target = st.selectbox("Target", df.columns)
    sensitive = st.selectbox("Sensitive", df.columns)

    if st.button("Run Analysis"):
        df[target] = df[target].apply(lambda x: 1 if ">50K" in str(x) else 0)

        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        df_test = df.loc[y_test.index].copy()
        df_test["pred"] = preds

        g1 = df_test[df_test[sensitive]==df[sensitive].unique()[0]]["pred"].mean()
        g2 = df_test[df_test[sensitive]==df[sensitive].unique()[1]]["pred"].mean()

        X2 = pd.get_dummies(df.drop(columns=[target, sensitive]))
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y)

        model2 = LogisticRegression(max_iter=5000)
        model2.fit(X_train2, y_train2)

        preds2 = model2.predict(X_test2)
        df_test2 = df.loc[y_test2.index].copy()
        df_test2["pred"] = preds2

        g1_after = df_test2[df_test2[sensitive]==df[sensitive].unique()[0]]["pred"].mean()
        g2_after = df_test2[df_test2[sensitive]==df[sensitive].unique()[1]]["pred"].mean()

        bias_before = abs(g1-g2)
        bias_after = abs(g1_after-g2_after)

        st.session_state.analysis = {
            "target": target, "sensitive": sensitive,
            "g1": g1, "g2": g2,
            "g1_after": g1_after, "g2_after": g2_after,
            "bias_before": bias_before, "bias_after": bias_after
        }

        st.success("Analysis complete")

        # Save to DB
        from datetime import datetime
        import json
        
        REPORTS_FILE = "reports.json"
        
        def load_reports():
            try:
                with open(REPORTS_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        
        def save_reports(data):
            with open(REPORTS_FILE, "w") as f:
                json.dump(data, f, indent=4)
        
        # ---------------- SAVE REPORT ----------------
        reports = load_reports()
        
        username = st.session_state.user
        
        reports.setdefault(username, {})
        
        report_id = str(datetime.now())
        
        reports[username][report_id] = {
            "target": target,
            "sensitive": sensitive,
            "g1": float(g1),
            "g2": float(g2),
            "g1_after": float(g1_after),
            "g2_after": float(g2_after),
            "bias_before": float(bias_before),
            "bias_after": float(bias_after),
            "created_at": report_id
        }
        
        save_reports(reports)
        
        st.success("✅ Report saved")

# Show metrics
if st.session_state.analysis:
    r = st.session_state.analysis
    c1,c2,c3 = st.columns(3)
    c1.metric("Bias Before", round(r["bias_before"],2))
    c2.metric("Bias After", round(r["bias_after"],2))
    c3.metric("Improvement", round(r["bias_before"]-r["bias_after"],2))

    fig = px.bar(x=["Before G1","Before G2","After G1","After G2"],
                 y=[r["g1"],r["g2"],r["g1_after"],r["g2_after"]])
    st.plotly_chart(fig, use_container_width=True)

# ================== AI ==================
if st.session_state.analysis:
    r = st.session_state.analysis
    st.markdown('<div class="section-title">AI Insights</div>', unsafe_allow_html=True)

    summary = df.describe().to_string() if file else ""

    prompt = build_prompt(
        r["target"], r["sensitive"],
        r["g1"], r["g2"], r["g1_after"], r["g2_after"],
        summary
    )

    if st.button("Generate AI Insights"):
        output, provider = generate_llm(prompt)
        st.session_state.llm = output
        st.success(f"Generated via {provider}")

    if st.session_state.llm:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        for line in st.session_state.llm.split("\n"):
            if any(k in line.lower() for k in ["summary","cause","risk","mitigation","impact","recommendation"]):
                st.markdown(f"### {line}")
            else:
                st.write(line)
        st.markdown('</div>', unsafe_allow_html=True)

# ================== REPORT ==================
if st.session_state.analysis:
    r = st.session_state.analysis
    def create_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("AI Bias Report", styles['Title']))

        table = Table([
            ["Metric","Before","After"],
            ["G1",round(r["g1"],2),round(r["g1_after"],2)],
            ["G2",round(r["g2"],2),round(r["g2_after"],2)]
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
    st.markdown('<div class="section-title">Report</div>', unsafe_allow_html=True)
    st.download_button("Download Report", data=pdf, file_name="bias_report.pdf")

st.markdown('</div>', unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("---")
st.markdown("<center style='color:#64748b;'>AI Bias Platform • Portfolio Project</center>", unsafe_allow_html=True)
