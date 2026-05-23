import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("loan_data.csv")

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.replace(",", "")

encoders = {}

for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

df = df.dropna()

df.to_csv("clean_data.csv", index=False)

joblib.dump(encoders, "encoders.pkl")

print("Clean dataset saved")
print("Encoders saved")
print("Rows:", len(df))