import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv("clean_data.csv")

X = df.drop("loan_paid_back", axis=1)
y = df["loan_paid_back"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, pred)
pre = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
auc = roc_auc_score(y_test, prob)
cm = confusion_matrix(y_test, pred)

print("Accuracy:", round(acc * 100, 2), "%")
print("Precision:", round(pre, 4))
print("Recall:", round(rec, 4))
print("F1 Score:", round(f1, 4))
print("ROC AUC:", round(auc, 4))
print("Confusion Matrix:")
print(cm)

joblib.dump(model, "model.pkl")

print("Model saved")