import io
import os

import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

DB_HOST = "amvera-hum0d0botyw2-cnpg-dataset-rw"
DB_PORT = "5432"
DB_NAME = "dataset"
DB_USER = "dataset"
DB_PASSWORD = "Qwerty123"
TABLE = "train"

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True)

app = FastAPI(title="Pyaterochka RTO API")


@app.get("/dataset")
def dataset() -> Response:
    df = pd.read_sql(f'SELECT * FROM {TABLE}', engine)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv")


@app.get("/stores/{new_id}")
def store(new_id: int):
    q = text(f'SELECT * FROM {TABLE} WHERE new_id = :id ORDER BY "Год", "Месяц"')
    df = pd.read_sql(q, engine, params={"id": new_id})
    return df.to_dict(orient="records")


class NewRecord(BaseModel):
    new_id: int
    god: int = Field(ge=2020, le=2030, alias="Year")
    mesyac: int = Field(ge=1, le=12, alias="Month")
    rto: float = Field(gt=0, alias="РТО")

    model_config = {"populate_by_name": True}


@app.post("/records", status_code=201)
def add_record(rec: NewRecord):
    q = text(f'INSERT INTO {TABLE} ("new_id", "Год", "Месяц", "РТО") VALUES (:new_id, :god, :mesyac, :rto)')
    with engine.begin() as conn:
        conn.execute(q, rec.model_dump())
    return {"status": "ok", **rec.model_dump(by_alias=True)}


@app.get("/stats")
def get_stats():
    query = text(
        f'SELECT COUNT(*) as total_records, AVG("РТО") as avg_rto, MAX("РТО") as max_rto, MIN("РТО") as min_rto, COUNT(DISTINCT new_id) as total_stores FROM {TABLE}')
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()
    return {
        "total_records": result[0],
        "avg_rto": round(result[1], 2),
        "total_stores": result[4],
        "max_rto": round(result[2], 2),
        "min_rto": round(result[3], 2)
    }
