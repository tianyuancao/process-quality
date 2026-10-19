import pandas as pd
from snow import query

df = query("SELECT * FROM SECOM.DBT_SCHEMA.STG_SECOM_FEATURES ORDER BY MEASURED_AT")

y = df["IS_FAIL"].astype(int)
X = df.drop(columns=["IS_FAIL", "WAFER_ID", "MEASURED_AT"])

cut = int(len(df) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

print(f"train {len(X_train)} rows, {y_train.mean():.2%} fail")
print(f"test  {len(X_test)} rows, {y_test.mean():.2%} fail")