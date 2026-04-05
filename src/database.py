from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

BASE = Path(__file__).resolve().parents[1]
LOCAL_DB = BASE / "data" / "runtime" / "neuroguia_final.db"
LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)

def get_database_url() -> str:
    return st.secrets.get("DATABASE_URL") or os.getenv("NEUROGUIA_DATABASE_URL") or f"sqlite:///{LOCAL_DB}"

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    url = get_database_url()
    if url.startswith("sqlite"):
        return create_engine(url, future=True, pool_pre_ping=True, connect_args={"check_same_thread": False})
    return create_engine(url, future=True, pool_pre_ping=True, poolclass=NullPool)

def init_db(schema_path: Optional[Path] = None) -> None:
    engine = get_engine()
    schema_path = schema_path or (BASE / "sql_schema.sql")
    sql = schema_path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        for statement in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(statement))

def read_sql_df(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    return pd.read_sql(text(query), get_engine(), params=params or {})
