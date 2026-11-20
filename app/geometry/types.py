"""Tipos e convenções geométricas.

Unidades internas: METROS.

PREMISSA DE ORIENTAÇÃO (leia antes de interpretar os resultados): esta versão não
alinha a nuvem a nenhuma referência externa. O eixo Z da nuvem recebida é assumido
como a vertical real (direção da gravidade) — ou seja, o sensor precisa estar nivelado
(ou a nuvem já corrigida para isso) antes da captura. Os eixos horizontal/vertical da
parede são então derivados automaticamente a partir da normal do plano ajustado por
RANSAC e desse Z global (ver `frame_from_normal`); não é preciso informar pose alguma.
- Desaprumo assinado: positivo = topo se afasta do sensor; negativo = topo se aproxima.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("vetor nulo")
    return v / n


def frame_from_normal(normal, up=UP):
    """Base (u, v) do plano: u horizontal ao longo do plano, v 'para cima' no plano.
    (u, v, n) é destra. Depende apenas da normal medida e do Z global (ver premissa acima)."""
    n = unit(normal)
    u = np.cross(up, n)
    if np.linalg.norm(u) < 1e-6:          # plano horizontal: u arbitrário
        u = np.cross([1.0, 0.0, 0.0], n)
    u = unit(u)
    v = np.cross(n, u)
    return u, v


@dataclass
class Plane:
    normal: np.ndarray      # unitária
    d: float                # n·x + d = 0
    centroid: np.ndarray

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.d


@dataclass
class ReferenceDimensions:
    """Dimensões de referência de uma parede, informadas manualmente pelo usuário (m)."""
    width: float
    height: float
    label: str = "referência manual"


@dataclass
class Tolerances:
    """Valores padrão são PLACEHOLDERS: defina conforme projeto e normas aplicáveis."""
    dimension_mm: float = 10.0
    plumb_mm_per_m: float = 5.0
    warn_fraction: float = 0.8      # entre warn_fraction e 1.0 da tolerância → "atencao"

    def for_key(self, key: str) -> float:
        return {"width": self.dimension_mm, "height": self.dimension_mm,
                "plumb": self.plumb_mm_per_m}[key]


@dataclass
class Measurement:
    key: str
    label: str
    unit: str
    measured: float | None
    reference: float | None
    deviation: float | None
    tolerance: float | None
    status: str                 # ok | atencao | fora | n/d
    note: str = ""


@dataclass
class InspectionResult:
    measurements: list[Measurement]
    overall_status: str
    warnings: list[str]
    n_points_used: int
    reference: dict
    viz: dict = field(default_factory=dict)
