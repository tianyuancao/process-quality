from pathlib import Path
import yaml
import snowflake.connector

PROFILE_NAME = "secom"   # matches the name in profiles.yml
TARGET = "dev"

print(Path.home())

cfg = yaml.safe_load((Path.home() / ".dbt" / "profiles.yml").read_text())
out = cfg[PROFILE_NAME]["outputs"][TARGET]

# connect without a database — it doesn't exist yet
conn = snowflake.connector.connect(
    account=out["account"],
    user=out["user"],
    private_key_file=out["private_key_path"],
    role=out.get("role"),
    warehouse=out.get("warehouse"),
)
print("connect to snowflake")
statements = [
    "CREATE DATABASE IF NOT EXISTS SECOM",
    "CREATE SCHEMA IF NOT EXISTS SECOM.RAW",
    "ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60",
]

cur = conn.cursor()
try:
    for sql in statements:
        cur.execute(sql)
        print("OK:", sql)
    cur.execute("SELECT CURRENT_ACCOUNT(), CURRENT_ROLE(), CURRENT_WAREHOUSE()")
    print("Connected as:", cur.fetchone())
finally:
    cur.close()
    conn.close()