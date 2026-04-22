import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import json, hashlib, io, time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Bias Platform", layout="wide")

# ------------------ FILES ------------------
USERS_FILE = "users.json"
REPORTS_FILE = "reports.json"

# ------------------ INIT FILES ------------------
def init_files():
    for file in [USERS_FILE, REPORTS_FILE]:
        try:
            open(file, "r")
        except:
            with open(file, "w") as f:
                json.dump({}, f)

init_files()

# ------------------ UTIL ------------------
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_json(file):
    with open(file, "r") as f:
        return json.load(f)

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

# ------------------ AUTH ------------------
def signup(username, password):
    users = load_json(USERS_FILE)
    if username in users:
        return False
    users[username] = hash_password(password)
    save_json(USERS_FILE, users)
    return True

def login(username, password):
    users = load_json(USERS_FILE)
    if username in users and users[username] == hash_password(password):
        return True
    return False

# ------------------ SESSION ------------------
if "user" not in st.session_state:
    st.session_state.user = None

# ------------------ AUTH UI ------------------
if not st.session_state.user:

    st.title("🔐 Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login(u, p):
                st.session_state.user = u
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")

        if st.button("Signup"):
            if signup(u, p):
                st.success("Account created!")
            else:
                st.error("User exists")

    st.stop()

# ------------------ LOGGED IN ------------------
st.sidebar.success(f"Logged in as {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

page = st.sidebar.radio("Navigate", ["Analyze", "Reports"])

# ------------------ ANALYSIS ------------------
if page == "Analyze":

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file).dropna()

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

            bias = abs(g1-g2)

            st.metric("Bias Score", round(bias,2))

            chart_df = pd.DataFrame({
                "Group":["G1","G2"],
                "Value":[g1,g2]
            })

            fig = px.bar(chart_df,x="Group",y="Value")
            st.plotly_chart(fig)

            # ---------------- SAVE REPORT ----------------
            reports = load_json(REPORTS_FILE)

            report_id = str(datetime.now())

            reports.setdefault(st.session_state.user, {})[report_id] = {
                "target": target,
                "sensitive": sensitive,
                "g1": float(g1),
                "g2": float(g2),
                "bias": float(bias)
            }

            save_json(REPORTS_FILE, reports)

            st.success("Report saved!")

# ------------------ REPORT HISTORY ------------------
if page == "Reports":

    st.header("📂 Your Reports")

    reports = load_json(REPORTS_FILE)
    user_reports = reports.get(st.session_state.user, {})

    if not user_reports:
        st.info("No reports yet")

    for rid, r in user_reports.items():

        st.markdown("---")
        st.write(f"📅 {rid}")
        st.write(f"Target: {r['target']}")
        st.write(f"Sensitive: {r['sensitive']}")
        st.write(f"Bias: {r['bias']}")

        # PDF
        def make_pdf():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            content = []
            content.append(Paragraph("Bias Report", styles['Title']))
            content.append(Paragraph(f"Bias: {r['bias']}", styles['Normal']))

            doc.build(content)
            buffer.seek(0)
            return buffer

        pdf = make_pdf()

        st.download_button("Download PDF", data=pdf, file_name=f"{rid}.pdf")
