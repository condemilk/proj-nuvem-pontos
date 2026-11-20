"""Ajuste de planos: SVD (mínimos quadrados) e RANSAC vetorizado em NumPy."""
from __future__ import annotations

import numpy as np

from .types import Plane


def fit_plane_svd(points: np.ndarray) -> Plane:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("são necessários ao menos 3 pontos para ajustar um plano")
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    return Plane(normal=n, d=-float(n @ c), centroid=c)


def ransac_plane(points: np.ndarray, threshold: float = 0.01, max_iters: int = 1000,
                 confidence: float = 0.999, seed: int | None = 0, batch: int = 64):
    """Retorna (Plane refinado por SVD nos inliers, máscara de inliers)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        raise ValueError("pontos insuficientes para RANSAC")
    rng = np.random.default_rng(seed)
    best_cnt, best = 0, None
    iters, done = max_iters, 0
    while done < iters:
        b = min(batch, iters - done)
        idx = rng.integers(0, n, (b, 3))
        p0, p1, p2 = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm, axis=1)
        ok = norm > 1e-12
        nrm[ok] /= norm[ok, None]
        dist = np.abs(nrm @ pts.T - np.einsum("bk,bk->b", nrm, p0)[:, None])
        cnt = (dist < threshold).sum(axis=1)
        cnt[~ok] = 0
        j = int(cnt.argmax())
        if cnt[j] > best_cnt:
            best_cnt, best = int(cnt[j]), (nrm[j].copy(), p0[j].copy())
            w = best_cnt / n
            if 0 < w < 1:
                iters = min(max_iters, max(done + b, int(np.log(1 - confidence) / np.log(1 - w ** 3)) + 1))
            elif w >= 1:
                iters = done + b
        done += b
    if best is None:
        raise ValueError("RANSAC não encontrou plano (pontos degenerados)")
    nrm, p0 = best
    plane = Plane(normal=nrm, d=-float(nrm @ p0), centroid=p0)
    mask = np.abs(plane.signed_distance(pts)) < threshold
    for _ in range(3):                      # refinamento: SVD nos inliers, reavalia máscara
        if mask.sum() < 3:
            break
        plane = fit_plane_svd(pts[mask])
        mask = np.abs(plane.signed_distance(pts)) < threshold
    return plane, mask


def orient_toward(plane: Plane, viewpoint) -> Plane:
    """Inverte a normal, se preciso, para apontar do plano em direção a `viewpoint`."""
    if plane.normal @ (np.asarray(viewpoint, float) - plane.centroid) < 0:
        return Plane(normal=-plane.normal, d=-plane.d, centroid=plane.centroid)
    return plane
