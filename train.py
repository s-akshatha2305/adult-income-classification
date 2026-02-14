import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_score,
    matthews_corrcoef
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("adult.csv")
print("\nThe first 10 rows of dataset are\n", df.head(3))
print("**********************************************")
print("\nInformation:\n", df.describe())
print("**********************************************")
print("\nChecking null values\n")
print(df.isnull().sum())
print("**********************************************")
print("\nChecking for ? in dataset\n")
for col in df.columns:
    print(col, df[col].value_counts().get("?",0))

# ---------------- Data Pre Processing -------------------
# drop missing values (?)
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# ------------ Encoding categorical columns --------------
encoders = {}
for column in df.columns:
    if df[column].dtype == "object":
        attr = df[column]
        label_encode = LabelEncoder()
        df[column] = label_encode.fit_transform(attr)
        encoders[column] = label_encode
        
joblib.dump(encoders, "model/encoders.pkl")

X = df.drop("income", axis=1)
y = df["income"]   # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

test_df = pd.DataFrame(X_test, columns=df.drop("income", axis=1).columns)
test_df["income"] = y_test
test_df.to_csv("model/test.csv", index=False)
print("Test data saved")

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

models_dict={
    "logistic": LogisticRegression(max_iter=1000),
    "knn": KNeighborsClassifier(),
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "xgboost": XGBClassifier(eval_metric="logloss"),
    "naive_bayes": GaussianNB()
}

results = []

for model_name, model in models_dict.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None
    
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC": auc
    }
    
    results.append(metrics)
    
    joblib.dump(model, f"model/{model_name}.pkl")
    
results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv("model/model_results.csv", index=False)





