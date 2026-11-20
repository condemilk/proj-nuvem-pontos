"""Pipeline: nuvem de pontos (em metros) + dimensões de referência manuais -> medidas.

Fluxo (sem BIM/IFC e sem alinhamento por pose — ver premissa de orientação em types.py):
1. pré-processamento (remoção de NaN/inf e outliers estatísticos);
2. plano dominante por RANSAC, orientado para o lado do sensor;
3. eixos horizontal/vertical do plano a partir da sua normal e do Z global;
4. largura, altura e desaprumo medidos diretamente nesses eixos;
5. comparação com as dimensões de referência informadas manualmente e aplicação das tolerâncias.
"""
from __future__ import annotations

import numpy as np

from . import deviation as dv
from . import measures as ms
from .planes import orient_toward, ransac_plane
from .preprocess import clean
from .types import InspectionResult, Measurement, ReferenceDimensions, Tolerances, frame_from_normal

MIN_POINTS = 30


def _meas(key, label, unit, measured, reference, tol: Tolerances, note="") -> Measurement:
    """Medida comparada a um valor de referência conhecido (largura, altura)."""
    if measured is None or reference is None:
        return Measurement(key, label, unit, measured, reference, None, tol.for_key(key), "n/d", note)
    dev = measured - reference
    status = dv.status_for(abs(dev), tol.for_key(key), tol.warn_fraction)
    return Measurement(key, label, unit, round(measured, 3), round(reference, 3), round(dev, 3),
                       tol.for_key(key), status, note)


def _reference_rectangle(plane, ext: dict, ref: ReferenceDimensions) -> np.ndarray:
    """4 cantos (m) de um retângulo ref.width x ref.height, apoiado no plano medido e centrado
    na própria cobertura medida (não há posição absoluta conhecida sem BIM: é só um overlay visual)."""
    u, v = frame_from_normal(plane.normal)
    center = plane.centroid + ext["center_u"] * u + ext["center_v"] * v
    hw, hh = ref.width / 2, ref.height / 2
    return np.array([center - u * hw - v * hh, center + u * hw - v * hh,
                     center + u * hw + v * hh, center - u * hw + v * hh])


def run_inspection(points_m: np.ndarray, ref: ReferenceDimensions, tol: Tolerances | None = None,
                   ransac_threshold: float = 0.01, seed: int = 0,
                   max_viz_points: int = 20000, use_open3d: bool = True) -> InspectionResult:
    tol = tol or Tolerances()
    warnings: list[str] = []

    raw = np.asarray(points_m, dtype=float)
    raw_count = len(raw)
    finite_count = int(np.isfinite(raw).all(axis=1).sum())
    pts = clean(raw, use_open3d=use_open3d)
    if len(pts) < MIN_POINTS:
        raise ValueError(
            f"Pontos insuficientes após a limpeza: {len(pts)} pontos válidos de {raw_count} "
            f"originais; são necessários pelo menos {MIN_POINTS}. "
            "Verifique o arquivo, a unidade e a quantidade de pontos capturados."
        )

    plane, inliers_clean = ransac_plane(pts, threshold=ransac_threshold, seed=seed)
    plane = orient_toward(plane, np.zeros(3))             # normal apontando para o sensor (origem)
    inlier_count = int(inliers_clean.sum())
    frac = inlier_count / len(pts)
    if inlier_count < MIN_POINTS:
        threshold_mm = ransac_threshold * 1000.0
        removed = raw_count - len(pts)
        raise ValueError(
            f"O RANSAC encontrou apenas {inlier_count} pontos compatíveis com o plano dominante "
            f"({frac:.1%} dos {len(pts)} pontos restantes após a limpeza; havia {raw_count} pontos "
            f"no arquivo e {finite_count} com coordenadas válidas). São necessários pelo menos "
            f"{MIN_POINTS} pontos no mesmo plano. Limiar atual: {threshold_mm:.1f} mm. "
            f"A limpeza removeu {removed} pontos. Verifique: (1) se a unidade do arquivo está correta; "
            "(2) se a nuvem contém uma região de superfície suficientemente grande; "
            "(3) se há piso, objetos ou outras superfícies misturados; "
            "(4) se o limiar RANSAC é compatível com o ruído do sensor. "
            "Você pode aumentar o limiar gradualmente, mas um valor alto demais pode juntar superfícies diferentes."
        )
    if frac < 0.3:
        warnings.append(f"Apenas {frac:.0%} dos pontos pertencem ao plano dominante; "
                        "o elemento pode estar parcialmente coberto ou há muito ruído/obstáculo.")

    # As medidas usam os pontos BRUTOS (não filtrados) que caem sobre o plano ajustado: o filtro
    # estatístico corrói bordas e cantos em varreduras de grade regular, encurtando largura/altura.
    dist = np.abs(plane.signed_distance(raw))
    inliers = dist < ransac_threshold
    band = dist < 3 * ransac_threshold                    # pontos "da parede", para a visualização

    line = ms.is_line_scan(raw[inliers])
    if line:
        warnings.append("A nuvem parece uma varredura em um único plano (linha). Altura e desaprumo "
                        "não podem ser determinados; faça a varredura cobrindo também o eixo vertical.")

    kind = ms.element_kind(plane.normal)
    if kind != "vertical":
        warnings.append("O plano identificado não é predominantemente vertical; confirme se a superfície "
                        "escaneada é realmente uma parede (o desaprumo pode não ser significativo aqui).")

    ext = ms.plane_extents(raw[inliers], plane)
    tilt = ms.tilt_metrics(plane.normal)

    na = "indeterminado (varredura em linha)"
    plumb_measured = None if line else tilt["plumb_mm_per_m"]
    plumb_status = "n/d" if plumb_measured is None else dv.status_for(
        abs(plumb_measured), tol.plumb_mm_per_m, tol.warn_fraction)
    plumb_note = na if line else ("" if plumb_measured is not None else
                                   "plano longe demais da vertical para calcular desaprumo")

    out = [
        _meas("width", "Largura", "mm", ext["width"] * 1000, ref.width * 1000, tol),
        _meas("height", "Altura", "mm", None if line else ext["height"] * 1000, ref.height * 1000, tol,
              note=na if line else ""),
        Measurement("plumb", "Desaprumo (+ topo afasta do sensor)", "mm/m",
                    None if plumb_measured is None else round(plumb_measured, 3),
                    None, None, tol.plumb_mm_per_m, plumb_status, plumb_note),
    ]

    for key, dim in (("width", ext["width"]), ("height", ext["height"])):
        m = next(x for x in out if x.key == key)
        ref_dim = ref.width if key == "width" else ref.height
        if m.measured is not None and dim < 0.9 * ref_dim:
            m.note = (m.note + " " if m.note else "") + "Medido < 90% da referência: cobertura possivelmente parcial."

    overall = dv.worst_status([m.status for m in out])

    wall = raw[band]
    rng = np.random.default_rng(seed)
    sel = np.arange(len(wall)) if len(wall) <= max_viz_points else rng.choice(len(wall), max_viz_points, replace=False)
    viz = {"points": np.round(wall[sel], 4).ravel().tolist(),
           "corners": np.round(_reference_rectangle(plane, ext, ref), 4).tolist()}

    return InspectionResult(
        measurements=out, overall_status=overall, warnings=warnings, n_points_used=int(band.sum()),
        reference={"label": ref.label, "width_mm": ref.width * 1000, "height_mm": ref.height * 1000},
        viz=viz)


def result_to_dict(res: InspectionResult, include_viz: bool = True) -> dict:
    d = {"measurements": [m.__dict__ for m in res.measurements], "overall_status": res.overall_status,
         "warnings": res.warnings, "n_points_used": res.n_points_used, "reference": res.reference}
    if include_viz:
        d["viz"] = res.viz
    return d
