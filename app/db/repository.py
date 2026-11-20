"""Persistência em SQLite e consultas/exportação com Pandas."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT NOT NULL DEFAULT 'default',
  name TEXT,
  created_at TEXT NOT NULL,
  source_file TEXT,
  element TEXT,
  n_points INTEGER,
  overall_status TEXT,
  params_json TEXT
);
CREATE TABLE IF NOT EXISTS measurements (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
  key TEXT NOT NULL,
  label TEXT,
  unit TEXT,
  measured REAL,
  reference_value REAL,
  deviation REAL,
  tolerance REAL,
  status TEXT
);
CREATE INDEX IF NOT EXISTS idx_meas_scan ON measurements(scan_id);
CREATE INDEX IF NOT EXISTS idx_meas_key ON measurements(key);
CREATE INDEX IF NOT EXISTS idx_scans_project ON scans(project, created_at);
"""


@contextmanager
def connect(path: str | Path):
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: str | Path) -> None:
    """Cria o esquema atual e migra bancos de versões anteriores do protótipo, preservando
    o histórico já gravado (nenhuma linha é apagada; só colunas obsoletas são removidas/renomeadas)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as c:
        c.executescript(SCHEMA)
        meas_cols = {row[1] for row in c.execute("PRAGMA table_info(measurements)")}
        if "reference_value" not in meas_cols and "bim" in meas_cols:
            c.execute("ALTER TABLE measurements RENAME COLUMN bim TO reference_value")
        scan_cols = {row[1] for row in c.execute("PRAGMA table_info(scans)")}
        for obsolete in ("mode", "bim_file"):          # alinhamento por modo/BIM: removido do sistema
            if obsolete in scan_cols:
                try:
                    c.execute(f"ALTER TABLE scans DROP COLUMN {obsolete}")
                except sqlite3.OperationalError:
                    pass                                # SQLite < 3.35: coluna fica, mas não é mais usada


def save_inspection(conn, result: dict, *, project="default", name=None, source_file=None,
                    element=None, params: dict | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO scans (project,name,created_at,source_file,element,n_points,overall_status,params_json)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (project, name, datetime.now(timezone.utc).isoformat(timespec="seconds"), source_file,
         element, result.get("n_points_used"), result.get("overall_status"), json.dumps(params or {})))
    scan_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO measurements (scan_id,key,label,unit,measured,reference_value,deviation,tolerance,status) VALUES (?,?,?,?,?,?,?,?,?)",
        [(scan_id, m["key"], m["label"], m["unit"], m["measured"], m["reference"], m["deviation"], m["tolerance"],
          m["status"]) for m in result["measurements"]])
    return scan_id


def list_scans(conn, project: str | None = None, limit: int = 200) -> pd.DataFrame:
    q, p = "SELECT * FROM scans", []
    if project:
        q += " WHERE project = ?"
        p.append(project)
    q += " ORDER BY id DESC LIMIT ?"
    p.append(limit)
    return pd.read_sql_query(q, conn, params=p)


def history(conn, key: str, project: str | None = None) -> pd.DataFrame:
    q = ("SELECT s.id AS scan_id, s.created_at, s.name, m.measured, m.reference_value AS reference, m.deviation, m.tolerance, m.status "
         "FROM measurements m JOIN scans s ON s.id = m.scan_id WHERE m.key = ?")
    p: list = [key]
    if project:
        q += " AND s.project = ?"
        p.append(project)
    return pd.read_sql_query(q + " ORDER BY s.id", conn, params=p)


def measurements_table(conn, project: str | None = None) -> pd.DataFrame:
    """Tabela plana (uma linha por medida) usada no CSV exportado."""
    q = ("SELECT s.id AS scan_id, s.project, s.name AS inspecao, s.created_at AS data_hora, s.source_file AS nuvem, "
         "s.element AS elemento, m.key AS medida, m.label AS descricao, "
         "m.unit AS unidade, m.measured AS medido, m.reference_value AS referencia, "
         "m.deviation AS desvio, m.tolerance AS tolerancia, m.status AS status "
         "FROM measurements m JOIN scans s ON s.id = m.scan_id")
    p: list = []
    if project:
        q += " WHERE s.project = ?"
        p.append(project)
    return pd.read_sql_query(q + " ORDER BY s.id, m.id", conn, params=p)


def export_csv(conn, dest, project: str | None = None) -> None:
    """`dest`: caminho ou buffer. utf-8-sig e ';' abrem direto no Excel em pt-BR."""
    measurements_table(conn, project).to_csv(dest, index=False, sep=";", decimal=",", encoding="utf-8-sig")
