import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import json, hashlib, io, time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Bias Platform", layout="wide")

# ---------------- STYLING ----------------
st.markdown("""
<style>
[data-testid="stSidebar"] {display:none;}
.block-container {max-width:1100px;margin:auto;}
.title {font-size:42px;font-weight:800;text-align:center;}
.subtitle {text-align:center;color:gray;margin-bottom:20px;}
.card {
    background:rgba(255,255,255,0.05);
    padding:20px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FILES ----------------
USERS_FILE = "users.json"
REPORTS_FILE = "reports.json"

def init_files():
    for f in [USERS_FILE, REPORTS_FILE]:
        try:
            open(f)
        except:
            with open(f, "w") as file:
                json.dump({}, file)

init_files()

# ---------------- UTILS ----------------
def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

def load_json(f):
    with open(f) as file:
        return json.load(file)

def save_json(f, data):
    with open(f, "w") as file:
        json.dump(data, file, indent=4)

# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None

# =====================================================
# 🔐 LOGIN / SIGNUP
# =====================================================
if not st.session_state.user:

    st.markdown('<div class="title">Login / Signup</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            users = load_json(USERS_FILE)
            if u in users and users[u] == hash_pw(p):
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            users = load_json(USERS_FILE)
            if u in users:
                st.error("User exists")
            else:
                users[u] = hash_pw(p)
                save_json(USERS_FILE, users)
                st.success("Account created")

    st.stop()

# =====================================================
# 🧭 NAVIGATION
# =====================================================
st.markdown('<div class="title">AI Bias Detection Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect • Explain • Mitigate AI Bias</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "home"

with col2:
    if st.button("📊 Analyze"):
        st.session_state.page = "analyze"

with col3:
    if st.button("📄 Reports"):
        st.session_state.page = "reports"

if "page" not in st.session_state:
    st.session_state.page = "home"

page = st.session_state.page

if st.button("Logout"):
    st.session_state.user = None
    st.rerun()

# =====================================================
# 🏠 HOME
# =====================================================
if page == "home":

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="card"><b>🎯 Objective</b><br>Detect bias in ML models</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><b>⚠️ Impact</b><br>Prevent unfair decisions</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><b>🚀 Applications</b><br>Finance, Hiring, Healthcare</div>', unsafe_allow_html=True)

# =====================================================
# 📊 ANALYZE
# =====================================================
if page == "analyze":

    st.markdown("## Bias Analysis")

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file).dropna()
        st.dataframe(df.head())

        target = st.selectbox("Target Column", df.columns)
        sensitive = st.selectbox("Sensitive Column", df.columns)

        if st.button("Run Analysis"):

            # -------- TARGET FIX --------
            if df[target].dtype == "object":
                df[target] = df[target].astype("category").cat.codes

            X = pd.get_dummies(df.drop(columns=[target]))
            y = df[target]

            # -------- CLASS CHECK --------
            if len(y.unique()) < 2:
                st.error("Target must have at least 2 classes")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model = LogisticRegression(max_iter=5000)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            df_test = df.loc[y_test.index].copy()
            df_test["pred"] = preds

            g1 = df_test[df_test[sensitive] == df[sensitive].unique()[0]]["pred"].mean()
            g2 = df_test[df_test[sensitive] == df[sensitive].unique()[1]]["pred"].mean()

            bias = abs(g1 - g2)

            st.session_state.analysis = {
                "g1": g1,
                "g2": g2,
                "bias": bias,
                "target": target,
                "sensitive": sensitive
            }

            # SAVE REPORT
            reports = load_json(REPORTS_FILE)
            reports.setdefault(st.session_state.user, {})
            rid = str(datetime.now())
            reports[st.session_state.user][rid] = st.session_state.analysis
            save_json(REPORTS_FILE, reports)

            st.success("Analysis complete")

    # -------- DISPLAY (IMPORTANT FIX) --------
    if st.session_state.analysis:

        r = st.session_state.analysis

        st.metric("Bias Score", round(r["bias"], 2))

        chart_df = pd.DataFrame({
            "Group": ["G1", "G2"],
            "Value": [r["g1"], r["g2"]]
        })

        fig = px.bar(chart_df, x="Group", y="Value")
        st.plotly_chart(fig)

        # -------- SIMPLE AI INSIGHTS (NO API) --------
        st.markdown("### AI Insights")

        if st.button("Generate Insights"):
            explanation = f"""
            The model shows a bias of {r['bias']:.2f} between groups 
            in {r['sensitive']}.

            This indicates unequal outcomes.

            Suggested mitigation:
            - Remove sensitive feature
            - Balance dataset
            - Apply fairness-aware models
            """

            st.write(explanation)

# =====================================================
# 📄 REPORTS
# =====================================================
if page == "reports":

    reports = load_json(REPORTS_FILE).get(st.session_state.user, {})

    if not reports:
        st.info("No reports yet")

    for rid, r in reports.items():
        st.markdown("---")
        st.write("Date:", rid)
        st.write("Target:", r["target"])
        st.write("Sensitive:", r["sensitive"])
        st.write("Bias:", r["bias"])
