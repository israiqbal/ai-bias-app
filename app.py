import streamlit as st
import pandas as pd
import plotly.express as px
import json, hashlib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Bias Platform", layout="wide")

# ---------------- UI CLEAN ----------------
st.markdown("""
<style>
[data-testid="stSidebar"] {display:none;}
.block-container {max-width:1100px;margin:auto;}
.title {font-size:42px;font-weight:800;text-align:center;}
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
# LOGIN
# =====================================================
if not st.session_state.user:
    st.title("Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_json(USERS_FILE)
        if u in users and users[u] == hash_pw(p):
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# =====================================================
# MAIN
# =====================================================
st.markdown('<div class="title">AI Bias Detection Platform</div>', unsafe_allow_html=True)

file = st.file_uploader("Upload CSV")

# ---------------- DEBUG ----------------
st.write("Session analysis exists:", st.session_state.analysis is not None)

if file:
    df = pd.read_csv(file)

    st.write("Dataset loaded:", df.shape)
    st.dataframe(df.head())

    target = st.selectbox("Target Column", df.columns)
    sensitive = st.selectbox("Sensitive Column", df.columns)

    if st.button("Run Analysis"):

        st.write("🚀 Button clicked")  # DEBUG

        try:
            df = df.dropna()

            # TARGET FIX
            if df[target].dtype == "object":
                df[target] = df[target].astype("category").cat.codes

            X = pd.get_dummies(df.drop(columns=[target]))
            y = df[target]

            st.write("Class distribution:", y.value_counts())  # DEBUG

            if len(y.unique()) < 2:
                st.error("❌ Target must have at least 2 classes")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model = LogisticRegression(max_iter=5000)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            df_test = df.loc[y_test.index].copy()
            df_test["pred"] = preds

            groups = df_test[sensitive].unique()

            if len(groups) < 2:
                st.error("❌ Sensitive column must have 2 groups")
                st.stop()

            g1 = df_test[df_test[sensitive] == groups[0]]["pred"].mean()
            g2 = df_test[df_test[sensitive] == groups[1]]["pred"].mean()

            bias = abs(g1 - g2)

            st.session_state.analysis = {
                "g1": g1,
                "g2": g2,
                "bias": bias,
                "target": target,
                "sensitive": sensitive
            }

            st.success("✅ Analysis stored")

        except Exception as e:
            st.error(f"ERROR: {e}")

# ---------------- OUTPUT ----------------
if st.session_state.analysis:

    st.markdown("## Results")

    r = st.session_state.analysis

    st.write("DEBUG: Showing results")  # DEBUG

    st.metric("Bias Score", round(r["bias"], 2))

    chart_df = pd.DataFrame({
        "Group": ["G1", "G2"],
        "Value": [r["g1"], r["g2"]]
    })

    fig = px.bar(chart_df, x="Group", y="Value")
    st.plotly_chart(fig)
