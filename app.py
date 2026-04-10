import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title("🧠 AI Bias Detection Tool")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    df = df.dropna()

    st.write("Preview:")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)
    sensitive = st.selectbox("Select Sensitive Column", df.columns)

    if st.button("Run Analysis"):

        # Convert target
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

        # BEFORE
        df_test = df.loc[y_test.index].copy()
        df_test['pred'] = preds

        g1 = df_test[df_test[sensitive] == df[sensitive].unique()[0]]['pred'].mean()
        g2 = df_test[df_test[sensitive] == df[sensitive].unique()[1]]['pred'].mean()

        st.subheader("📊 Before Mitigation")
        st.write(g1, g2)

        # MITIGATION
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

        st.subheader("📊 After Mitigation")
        st.write(g1_after, g2_after)