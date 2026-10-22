import joblib
import pandas as pd
import streamlit as st

model = joblib.load("dropout_model.pkl")
columns = joblib.load("model_columns.pkl")
le = joblib.load("label_encoder.pkl")
defaults = joblib.load("defaults.pkl")
cat_cols = joblib.load("cat_cols.pkl")

st.title("Student Dropout Risk Predictor")

row = dict(defaults)
for c in cat_cols:
    row[c] = int(row[c])

row["Admission grade"] = st.slider("Admission grade", 95.0, 190.0, 125.0)
row["Age at enrollment"] = st.slider("Age at enrollment", 17, 70, 20)
row["Debtor"] = int(st.checkbox("Student is a debtor"))
row["Tuition fees up to date"] = int(st.checkbox("Tuition fees up to date", value=True))
row["Scholarship holder"] = int(st.checkbox("Scholarship holder"))

for sem in ["1st", "2nd"]:
    st.subheader(f"{sem} semester")
    enrolled = st.number_input(f"Units enrolled ({sem})", 0, 30, 6, key=f"e{sem}")
    approved = st.number_input(f"Units approved ({sem})", 0, 30, 5, key=f"a{sem}")
    grade = st.slider(f"Average grade ({sem})", 0.0, 20.0, 12.0, key=f"g{sem}")
    row[f"Curricular units {sem} sem (enrolled)"] = enrolled
    row[f"Curricular units {sem} sem (approved)"] = approved
    row[f"Curricular units {sem} sem (grade)"] = grade
    row[f"approval_rate_{sem}"] = approved / enrolled if enrolled > 0 else 0

row["grade_change"] = (row["Curricular units 2nd sem (grade)"]
                       - row["Curricular units 1st sem (grade)"])
row["financial_risk"] = int(row["Debtor"] == 1
                            or row["Tuition fees up to date"] == 0
                            or row["Scholarship holder"] == 0)

if st.button("Predict"):
    df_in = pd.get_dummies(pd.DataFrame([row]), columns=cat_cols)
    df_in = df_in.reindex(columns=columns, fill_value=0).astype(float)
    probs = model.predict_proba(df_in)[0]
    st.success(f"Predicted outcome: {le.classes_[probs.argmax()]}")
    st.bar_chart(pd.Series(probs, index=le.classes_))

