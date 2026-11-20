"""Limpeza da nuvem: NaN/inf, outliers estatísticos, voxel downsample."""
from __future__ import annotations

import numpy as np

try:                                    # Open3D é opcional
    import open3d as o3d
except Exception:                       # pragma: no cover
    o3d = None


def finite_only(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points[np.isfinite(points).all(axis=1)]


def remove_statistical_outliers(points, nb_neighbors=20, std_ratio=2.5, use_open3d=True) -> np.ndarray:
    """Mesma semântica do Open3D: descarta pontos cuja distância média aos k vizinhos
    excede média_global + std_ratio * desvio_global."""
    points = np.asarray(points, dtype=float)
    if len(points) <= nb_neighbors + 1:
        return points
    if use_open3d and o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return points[np.asarray(idx)]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(points).query(points, k=nb_neighbors + 1)
    mean_d = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std_ratio * mean_d.std()
    return points[mean_d <= thr]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Substitui os pontos de cada voxel pelo seu centroide."""
    points = np.asarray(points, dtype=float)
    if voxel <= 0 or len(points) == 0:
        return points
    idx = np.floor((points - points.min(axis=0)) / voxel).astype(np.int64)
    dims = idx.max(axis=0) + 1
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inv.ravel(), points)
    return sums / counts[:, None]


def clean(points, nb_neighbors=20, std_ratio=2.5, voxel=None, use_open3d=True) -> np.ndarray:
    pts = finite_only(points)
    pts = remove_statistical_outliers(pts, nb_neighbors, std_ratio, use_open3d)
    if voxel:
        pts = voxel_downsample(pts, voxel)
    return pts
