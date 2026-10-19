from pathlib import Path
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
from snow import connect

DATA = Path("data")

# 590 sensor columns, whitespace-separated, "NaN" for missing
X = pd.read_csv(DATA / "secom.data", sep=r"\s+", header=None, na_values="NaN")
X.columns = [f"SENSOR_{i+1:03d}" for i in range(X.shape[1])]

# labels: -1 = pass, 1 = fail, plus a timestamp
y = pd.read_csv(DATA / "secom_labels.data", sep=r"\s+", header=None,
                names=["LABEL", "DATE", "TIME"])
ts = pd.to_datetime(y["DATE"],
                    format="%d/%m/%Y %H:%M:%S")

df = pd.concat([X], axis=1)
df.insert(0, "WAFER_ID", range(1, len(df) + 1))
df.insert(1, "MEASURED_AT", ts)
df.insert(2, "IS_FAIL", (y["LABEL"] == 1).astype(int))
df = df.sort_values("MEASURED_AT").reset_index(drop=True)
df["MEASURED_AT"] = df["MEASURED_AT"].dt.strftime("%Y-%m-%d %H:%M:%S")

print(f"rows={len(df)}  fail_rate={df.IS_FAIL.mean():.3%}")
print(f"constant columns: {sum(X[c].nunique(dropna=True) <= 1 for c in X.columns)}")
print(f"columns >50% null: {(X.isna().mean() > 0.5).sum()}")


sensor_cols = [c for c in df.columns if c.startswith("SENSOR_")]

ddl = f"""
CREATE OR REPLACE TABLE SECOM.RAW.SECOM_RAW (
    WAFER_ID     NUMBER,
    MEASURED_AT  TIMESTAMP_NTZ,
    IS_FAIL      NUMBER(1),
    {", ".join(f"{c} FLOAT" for c in sensor_cols)}
)
"""

conn = connect(database="SECOM", schema="RAW")
conn.cursor().execute(ddl)

ok, nchunks, nrows, _ = write_pandas(
    conn, df, "SECOM_RAW", auto_create_table=False, overwrite=False
)
print("loaded:", ok, nrows)
conn.close()

