"""Gerador interno de nuvens sintéticas com erros conhecidos, utilizado nos testes automatizados.

Não há nenhuma simulação de pose/posição de sensor aqui: os pontos já nascem no referencial em
que seriam recebidos pelo pipeline, com o sensor implicitamente na origem (0,0,0) — mesma premissa
usada por `analyze.run_inspection` (via `orient_toward`, que orienta a normal medida para a
origem) — e a parede posicionada a `distance` metros dele, com a normal da face apontando da
parede para o sensor. Os defeitos (desaprumo, abaulamento, erro de dimensão etc.) são embutidos
diretamente na própria superfície gerada.
"""
from __future__ import annotations

import numpy as np

from .types import ReferenceDimensions


def make_wall(width=3.0, height=2.5, n_points=8000, noise_std=0.002, distance=2.0,
             plumb_mm_per_m=0.0, bulge_mm=0.0, width_error_mm=0.0, height_error_mm=0.0,
             offset_mm=0.0, outlier_frac=0.01, floor_clutter=False, coverage=1.0,
             line_scan_height=None, sort_by_height=False, seed=0) -> np.ndarray:
    """Nuvem de pontos de uma parede com defeitos conhecidos (m), a `distance` m do sensor
    (que fica implicitamente na origem). Base em z=0; face voltada para o sensor.

    plumb_mm_per_m > 0: topo se afasta do sensor. bulge_mm > 0: abaulamento saliente no centro
    (para o lado do sensor). offset_mm > 0: parede inteira deslocada para o lado do sensor.
    """
    rng = np.random.default_rng(seed)
    W = width + width_error_mm / 1000
    H = height + height_error_mm / 1000
    a = rng.uniform(-W / 2 * coverage, W / 2 * coverage, n_points)
    b = rng.uniform(0, H, n_points) if line_scan_height is None else np.full(n_points, line_scan_height)
    if sort_by_height:
        order = np.argsort(b)
        a, b = a[order], b[order]
    alpha = np.arctan(plumb_mm_per_m / 1000)
    n_dir = np.array([0.0, -1.0, 0.0])                    # aponta da parede para o sensor (origem)
    pts = np.zeros((n_points, 3))
    pts[:, 0] = a
    pts[:, 1] = distance                                  # parede a `distance` m do sensor
    pts[:, 2] = b * np.cos(alpha)
    n_comp = -b * np.sin(alpha) + offset_mm / 1000        # topo se afasta → -n (mais longe do sensor)
    sigma = 0.25 * min(W, H)
    n_comp += bulge_mm / 1000 * np.exp(-((a ** 2) + (b - H / 2) ** 2) / (2 * sigma ** 2))
    pts += n_comp[:, None] * n_dir[None, :]
    pts += rng.normal(0, noise_std, pts.shape)
    out = [pts]
    n_out = int(outlier_frac * n_points)
    if n_out:                                             # entre o sensor e a parede, e um pouco além dela
        o = np.column_stack([rng.uniform(-W / 2, W / 2, n_out), rng.uniform(0.1 * distance, 1.1 * distance, n_out),
                             rng.uniform(0, H, n_out)])
        out.append(o)
    if floor_clutter:                                     # piso entre o sensor e a parede
        m = n_points // 5
        out.append(np.column_stack([rng.uniform(-W / 2, W / 2, m), rng.uniform(0.05, distance, m),
                                    rng.normal(0, noise_std, m)]))
    return np.vstack(out) if len(out) > 1 else pts


def make_scan(width=3.0, height=2.5, **kw):
    """Retorna (pontos [m], ReferenceDimensions) de uma parede sintética, sem nenhuma pose/sensor envolvida."""
    return make_wall(width=width, height=height, **kw), ReferenceDimensions(width, height)
