"""Snowflake connection helper.

Reads credentials from ~/.dbt/profiles.yml so the Python scripts and dbt
share one source of truth, and nothing secret lives in the repo.
"""

from pathlib import Path
from contextlib import contextmanager
import yaml
import snowflake.connector

PROFILES = Path.home() / ".dbt" / "profiles.yml"


def _profile(profile="secom", target="dev"):
    cfg = yaml.safe_load(PROFILES.read_text())
    if profile not in cfg:
        raise KeyError(f"Profile '{profile}' not in {PROFILES}. Found: {list(cfg)}")
    return cfg[profile]["outputs"][target]


def connect(database=None, schema=None, profile="secom", target="dev"):
    """Open a Snowflake connection using key-pair auth. Caller must close it."""
    out = _profile(profile, target)
    return snowflake.connector.connect(
        account=out["account"],
        user=out["user"],
        private_key_file=out["private_key_path"],
        role=out.get("role"),
        warehouse=out.get("warehouse"),
        database=database or out.get("database"),
        schema=schema or out.get("schema"),
    )


@contextmanager
def session(database=None, schema=None, **kw):
    """Connection that closes itself.

    with session(database="SECOM", schema="RAW") as conn:
        ...
    """
    conn = connect(database, schema, **kw)
    try:
        yield conn
    finally:
        conn.close()


def query(sql, database=None, schema=None, **kw):
    """Run SQL, return a DataFrame."""
    with session(database, schema, **kw) as conn:
        return conn.cursor().execute(sql).fetch_pandas_all()