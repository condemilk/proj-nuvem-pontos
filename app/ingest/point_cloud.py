"""Leitura e validação de nuvens de pontos (.xyz/.txt/.csv/.pts, .pcd, .ply). Saída em METROS."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001}
SUPPORTED = {".xyz", ".txt", ".csv", ".pts", ".pcd", ".ply"}
_SPLIT = re.compile(r"[,;\s]+")


class InputError(ValueError):
    """Erro de entrada legível para o usuário."""


@dataclass
class LoadedCloud:
    points: np.ndarray
    n_raw: int
    warnings: list[str] = field(default_factory=list)

    @property
    def bounds(self) -> dict:
        return {"min": self.points.min(axis=0).tolist(), "max": self.points.max(axis=0).tolist()}


def _read_text_table(path: Path) -> np.ndarray:
    rows, header, bad = [], None, 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#/":
                continue
            parts = [p for p in _SPLIT.split(s) if p]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                if header is None and not rows:
                    header = [p.lower() for p in parts]
                else:
                    bad += 1
                continue
            if len(vals) < 3:
                bad += 1
                continue
            rows.append(vals)
    if not rows:
        raise InputError("Nenhuma linha numérica com ao menos 3 colunas (x y z) encontrada.")
    if bad > 0.05 * (len(rows) + bad):
        raise InputError(f"{bad} linhas inválidas: o arquivo não parece ser uma nuvem x y z.")
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype=float)
    cols = [0, 1, 2]
    if header and all(c in header for c in ("x", "y", "z")):
        cols = [header.index(c) for c in ("x", "y", "z")]
    if max(cols) >= arr.shape[1]:
        raise InputError("Número de colunas incompatível com o cabeçalho.")
    return arr[:, cols]


def _read_pcd_ascii(path: Path) -> np.ndarray | None:
    """PCD ASCII em NumPy puro; retorna None se for binário (delegar ao Open3D)."""
    fields, data_idx = None, None
    with open(path, "rb") as f:
        lines = []
        for raw in f:
            try:
                s = raw.decode("ascii").strip()
            except UnicodeDecodeError:
                return None
            lines.append(s)
            if s.upper().startswith("FIELDS"):
                fields = s.split()[1:]
            if s.upper().startswith("DATA"):
                if s.split()[1].lower() != "ascii":
                    return None
                data_idx = len(lines)
                break
        if fields is None or data_idx is None or not all(c in fields for c in "xyz"):
            raise InputError("Cabeçalho PCD inválido (FIELDS x y z / DATA ascii ausentes).")
        body = np.loadtxt(f, dtype=float, ndmin=2)
    return body[:, [fields.index(c) for c in "xyz"]]


def _read_with_open3d(path: Path) -> np.ndarray:
    try:
        import open3d as o3d
    except Exception as e:
        raise InputError(f"Para ler {path.suffix} (binário/PLY) instale o Open3D: pip install open3d") from e
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise InputError("Open3D não conseguiu extrair pontos do arquivo.")
    return pts


def load_point_cloud(path: str | Path, unit: str = "m", min_points: int = 30) -> LoadedCloud:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise InputError(f"Formato {ext or '(sem extensão)'} não suportado. Use: {', '.join(sorted(SUPPORTED))}.")
    if unit not in UNIT_SCALE:
        raise InputError(f"Unidade inválida: {unit}. Use m, cm ou mm.")
    if ext == ".pcd":
        pts = _read_pcd_ascii(path)
        pts = _read_with_open3d(path) if pts is None else pts
    elif ext == ".ply":
        pts = _read_with_open3d(path)
    else:
        pts = _read_text_table(path)

    n_raw = len(pts)
    warnings: list[str] = []
    pts = np.asarray(pts, dtype=float) * UNIT_SCALE[unit]
    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        warnings.append(f"{int((~finite).sum())} pontos com NaN/inf foram descartados.")
        pts = pts[finite]
    if len(pts) < min_points:
        raise InputError(f"Poucos pontos válidos ({len(pts)}); mínimo {min_points}.")
    ext_max = float((pts.max(axis=0) - pts.min(axis=0)).max())
    if ext_max < 0.05:
        warnings.append(f"Extensão total de {ext_max * 1000:.1f} mm: a unidade informada ({unit}) está correta?")
    elif ext_max > 200:
        warnings.append(f"Extensão total de {ext_max:.0f} m: a unidade informada ({unit}) está correta?")
    if np.allclose(pts, pts[0]):
        raise InputError("Todos os pontos são idênticos.")
    return LoadedCloud(pts, n_raw, warnings)
