"""API FastAPI. Execute com: uvicorn app.main:app --reload."""
from __future__ import annotations

import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.db import repository as repo
from app.geometry.analyze import result_to_dict, run_inspection
from app.geometry.types import ReferenceDimensions, Tolerances
from app.ingest.point_cloud import SUPPORTED, InputError, load_point_cloud

DATA = Path(os.getenv("DATA_DIR", "data"))
UPLOADS, DB_PATH = DATA / "uploads", DATA / "inspecoes.db"
UPLOADS.mkdir(parents=True, exist_ok=True)
repo.init_db(DB_PATH)
ID_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_UPLOAD = 200 * 1024 * 1024

app = FastAPI(title="Inspeção geométrica de nuvens de pontos")


# ---------- modelos de requisição ----------
class ReferenceIn(BaseModel):
    """Dimensões de referência informadas manualmente, em metros."""
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class TolIn(BaseModel):
    dimension_mm: float = 10.0
    plumb_mm_per_m: float = 5.0


class InspectIn(BaseModel):
    cloud_id: str
    reference: ReferenceIn
    tolerances: TolIn = Field(default_factory=TolIn)
    ransac_threshold_mm: float = Field(default=10.0, gt=0, le=1000)
    save: bool = True
    name: Optional[str] = None
    project: str = "default"


# ---------- utilidades ----------
def _check_id(i: str) -> str:
    if not ID_RE.match(i or ""):
        raise HTTPException(400, "identificador inválido")
    return i


def _save_upload(file: UploadFile, allowed: set[str]) -> tuple[str, Path]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(415, f"Formato {ext or '?'} não suportado. Use: {', '.join(sorted(allowed))}")
    fid = uuid.uuid4().hex
    path = UPLOADS / f"{fid}{ext}"
    size = 0
    with open(path, "wb") as out:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                out.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, "Arquivo maior que 200 MB.")
            out.write(chunk)
    (UPLOADS / f"{fid}.json").write_text(json.dumps({"filename": file.filename, "ext": ext}))
    return fid, path


def _stored(fid: str) -> tuple[Path, dict]:
    _check_id(fid)
    meta_p = UPLOADS / f"{fid}.json"
    if not meta_p.exists():
        raise HTTPException(404, "arquivo não encontrado; refaça o upload")
    meta = json.loads(meta_p.read_text())
    return UPLOADS / f"{fid}{meta['ext']}", meta


def _cloud(fid: str, unit: Optional[str] = None):
    path, meta = _stored(fid)
    unit = unit or meta.get("unit", "m")
    try:
        return load_point_cloud(path, unit), meta
    except InputError as e:
        raise HTTPException(422, str(e))


def _resolve_reference(r: ReferenceIn) -> ReferenceDimensions:
    return ReferenceDimensions(r.width, r.height)


def _tol(t: TolIn) -> Tolerances:
    return Tolerances(**t.model_dump())


# ---------- nuvem ----------
@app.post("/api/cloud")
def upload_cloud(file: UploadFile = File(...), unit: str = Form("m")):
    fid, path = _save_upload(file, SUPPORTED)
    meta_p = UPLOADS / f"{fid}.json"
    meta = json.loads(meta_p.read_text())
    meta["unit"] = unit
    meta_p.write_text(json.dumps(meta))
    try:
        c = load_point_cloud(path, unit)
    except InputError as e:
        path.unlink(missing_ok=True)
        meta_p.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    return {"cloud_id": fid, "filename": meta["filename"], "unit": unit, "n_points": len(c.points),
            "bounds": c.bounds, "warnings": c.warnings}


@app.get("/api/cloud/{cloud_id}/preview")
def cloud_preview(cloud_id: str, max_points: int = 30000):
    c, _ = _cloud(cloud_id)
    pts = c.points
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    return {"points": np.round(pts, 4).ravel().tolist()}


# ---------- inspeção ----------
@app.post("/api/inspect")
def inspect(req: InspectIn):
    cloud, meta = _cloud(req.cloud_id)
    ref = _resolve_reference(req.reference)
    try:
        res = run_inspection(cloud.points, ref, _tol(req.tolerances),
                             ransac_threshold=req.ransac_threshold_mm / 1000.0)
    except ValueError as e:
        raise HTTPException(422, str(e))
    out = result_to_dict(res)
    out["cloud_warnings"] = cloud.warnings
    if req.save:
        with repo.connect(DB_PATH) as c:
            out["scan_id"] = repo.save_inspection(
                c, out, project=req.project, name=req.name or meta["filename"], source_file=meta["filename"],
                element=ref.label, params=req.model_dump())
    return out


# ---------- histórico / exportação ----------
# Sem botões próprios na interface (removidos a pedido: a seção "Histórico" ficou só com
# baixar CSV e imprimir), mas mantidos como API — úteis por linha de comando ou uma futura tela,
# e usados diretamente pelos testes de banco de dados.
@app.get("/api/history")
def api_history(key: str, project: Optional[str] = None):
    with repo.connect(DB_PATH) as c:
        return json.loads(repo.history(c, key, project).to_json(orient="records"))


@app.get("/api/scans")
def api_scans(project: Optional[str] = None):
    with repo.connect(DB_PATH) as c:
        return json.loads(repo.list_scans(c, project).drop(columns=["params_json"]).to_json(orient="records"))


@app.get("/api/export.csv")
def api_export(project: Optional[str] = None):
    buf = io.StringIO()
    with repo.connect(DB_PATH) as c:
        repo.export_csv(c, buf, project)
    return Response(buf.getvalue().encode("utf-8-sig"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="inspecoes.csv"'})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
