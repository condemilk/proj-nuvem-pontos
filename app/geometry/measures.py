"""Medidas sobre um plano ajustado: dimensões e inclinação/desaprumo."""
from __future__ import annotations

import numpy as np

from .types import UP, Plane, frame_from_normal


def plane_extents(points: np.ndarray, plane: Plane, trim_pct: float = 0.0) -> dict:
    """Extensões nos eixos locais (u horizontal, v 'para cima' no plano).
    Por padrão usa mín/máx dos inliers (percentis cortariam ~2*trim% da dimensão em nuvens densas e uniformes)."""
    u, v = frame_from_normal(plane.normal)
    rel = points - plane.centroid
    a, b = rel @ u, rel @ v
    a0, a1 = np.percentile(a, [trim_pct, 100 - trim_pct])
    b0, b1 = np.percentile(b, [trim_pct, 100 - trim_pct])
    return {"width": float(a1 - a0), "height": float(b1 - b0),
            "center_u": float((a0 + a1) / 2), "center_v": float((b0 + b1) / 2)}


def is_line_scan(points: np.ndarray, ratio: float = 0.05) -> bool:
    """True se os pontos são ~colineares (varredura em um único plano horizontal)."""
    if len(points) < 3:
        return True
    s = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return bool(s[0] > 0 and s[1] / s[0] < ratio)


def tilt_metrics(normal, up=UP) -> dict:
    """Desaprumo do plano em relação à vertical (eixo Z global; ver premissa em types.py).
    `normal` deve apontar para o lado do sensor. `plumb_mm_per_m` é None se o plano estiver
    longe demais da vertical para "desaprumo" fazer sentido (ex.: piso/teto escaneado por engano)."""
    c = float(np.clip(np.asarray(normal) @ up, -1.0, 1.0))
    plumb_dev_deg = float(np.degrees(np.arcsin(abs(c))))       # desvio do plano em relação à vertical
    plumb_mm_m = c / np.sqrt(1 - c * c) * 1000 if abs(c) < 0.7 else None   # assinado; None se não vertical
    return {"plumb_dev_deg": plumb_dev_deg,
            "plumb_mm_per_m": None if plumb_mm_m is None else float(plumb_mm_m)}


def element_kind(normal, up=UP) -> str:
    c = abs(float(np.asarray(normal) @ up))
    return "vertical" if c < 0.2 else ("horizontal" if c > 0.98 else "inclined")

