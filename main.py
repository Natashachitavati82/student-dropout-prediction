import pandas as pd
df=pd.read_csv("dataset.csv")
print("Shape:",df.shape)
print(df.info())
print(df.describe())
print(df.head())
import matplotlib.pyplot as plt
print(df.columns.tolist())
print(df["Target"].value_counts())
print(df["Target"].value_counts(normalize=True)*100)
df["Target"].value_counts().plot(kind="bar")
plt.title("Student outcomes")
plt.ylabel("Number of Students")
plt.show()
missing=df.isnull().sum()
print("Columns with missing values:\n",[missing>0])
print("Total missing:",df.isnull().sum().sum())
print("Duplicate rows:",df.duplicated().sum())
print(df.describe())
df[["Previous qualification","Age at enrollment"]].plot(kind="box",subplots=True, layout=(1,2), figsize=(8,4))
plt.show()
pd.set_option('display.max_rows', None)
for sem in["1st","2nd"]:
    enrolled=df[f"Curricular units {sem} sem (enrolled)"]
    approved=df[f"Curricular units {sem} sem (approved)"]
    df[f"approval_rate_{sem}"]=(approved/enrolled.replace(0,float("nan"))).fillna(0)
df["grade_change"] = (df["Curricular units 2nd sem (grade)"]
                      - df["Curricular units 1st sem (grade)"])
df["financial_risk"] = (
            (df["Debtor"] == 1)
            | (df["Tuition fees up to date"] == 0)
            | (df["Scholarship holder"] == 0)
).astype(int)
print(df[["approval_rate_1st","approval_rate_2nd","grade_change","financial_risk"]].describe())
print(df.groupby("Target")[["approval_rate_2nd", "grade_change", "financial_risk"]].mean())
x=df.drop("Target",axis=1)
y=df["Target"]
cat_cols=["Marital status","Application mode","Course","Nacionality","Mother's qualification","Father's qualification"]
x_encoded=pd.get_dummies(x,columns=cat_cols,drop_first=True)
print("Before:",x.shape)
print("After:",x_encoded.shape)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_encoded=le.fit_transform(y)
print(dict(zip(le.classes_,le.transform(le.classes_))))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    x_encoded, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train class counts:", np.bincount(y_train))
print("Test class counts:", np.bincount(y_test))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, recall_score

models = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"), True),
    "Random Forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42), False),
    "Gradient Boosting": (HistGradientBoostingClassifier(class_weight="balanced", random_state=42), False),
}

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print("\n====", name, "====")
    print("Macro F1:", round(f1_score(y_test, pred, average="macro"), 3))
    print("Dropout recall:", round(recall_score(y_test, pred, labels=[0], average=None)[0], 3))
    print(classification_report(y_test, pred, target_names=le.classes_))
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
}
for name, m in candidates.items():
    scores = cross_val_score(m, x_encoded, y_encoded, cv=cv, scoring="f1_macro")
    print(name, "CV macro F1:", round(scores.mean(), 3), "+/-", round(scores.std(), 3))

# Confusion matrix for the Random Forest on the test set
rf = models["Random Forest"][0]
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, display_labels=le.classes_)
plt.show()
import joblib

final_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
final_model.fit(x_encoded, y_encoded)  # train on all data

joblib.dump(final_model, "dropout_model.pkl")
joblib.dump(list(x_encoded.columns), "model_columns.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Saved!")

defaults = x.mode().iloc[0]
joblib.dump(defaults.to_dict(), "defaults.pkl")
joblib.dump(cat_cols, "cat_cols.pkl")