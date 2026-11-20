"""Testes do núcleo geométrico com nuvens sintéticas de erro conhecido."""
import numpy as np

from app.geometry.analyze import run_inspection
from app.geometry.planes import fit_plane_svd, ransac_plane
from app.geometry.preprocess import clean, voxel_downsample
from app.geometry.synthetic import make_scan
from app.geometry.types import ReferenceDimensions


def get(res, key):
    return next(m for m in res.measurements if m.key == key)


def test_ransac_ignores_floor_clutter():
    pts, _ = make_scan(floor_clutter=True, plumb_mm_per_m=0)
    plane, mask = ransac_plane(pts, threshold=0.01, seed=1)
    assert mask.mean() > 0.6                      # parede é o plano dominante
    assert abs(plane.normal[2]) < 0.05            # normal ~horizontal (não é o piso)


def test_perfect_wall_is_within_tolerance():
    pts, ref = make_scan(noise_std=0.002)
    res = run_inspection(pts, ref)
    assert abs(get(res, "width").deviation) < 15
    assert abs(get(res, "height").deviation) < 15
    assert abs(get(res, "plumb").measured) < 1.5
    assert res.overall_status in ("ok", "atencao")


def test_plumb_sign_and_magnitude():
    for sign in (+1, -1):
        pts, ref = make_scan(plumb_mm_per_m=sign * 12.0, noise_std=0.001, n_points=15000)
        res = run_inspection(pts, ref)
        m = get(res, "plumb")
        assert abs(m.measured - sign * 12.0) < 1.0, (sign, m.measured)
        assert m.status == "fora"                  # 12 mm/m > 5 mm/m (tolerância placeholder)
        assert m.reference is None and m.deviation is None   # não há "referência" de desaprumo, só medido x tolerância


def test_width_height_errors_detected():
    pts, ref = make_scan(width_error_mm=-40, height_error_mm=25, noise_std=0.0005,
                         n_points=20000, outlier_frac=0)
    res = run_inspection(pts, ref)
    assert abs(get(res, "width").deviation - (-40)) < 8
    assert abs(get(res, "height").deviation - 25) < 8


def test_regular_grid_edges_are_not_eroded():
    """Varredura angular em grade regular: o filtro estatístico não pode encurtar largura/altura."""
    xs, zs = np.meshgrid(np.linspace(-1.5, 1.5, 61), np.linspace(0, 2.5, 51))   # passo de 5 cm, cobre 3,0 x 2,5 m
    pts = np.column_stack([xs.ravel(), np.full(xs.size, 2.0), zs.ravel()])
    pts += np.random.default_rng(0).normal(0, 0.001, pts.shape)
    res = run_inspection(pts, ReferenceDimensions(3.0, 2.5))
    assert abs(get(res, "width").deviation) < 15 and abs(get(res, "height").deviation) < 15


def test_global_rotation_about_z_does_not_change_measurements():
    """Girar toda a nuvem em torno do eixo Z (ex.: a parede não está exatamente de frente para
    quem capturou) não deve alterar largura, altura nem desaprumo: eles são medidos nos eixos
    do próprio plano ajustado por RANSAC, não em eixos fixos do mundo. Não há aqui nenhum
    conceito de "pose": é só uma rotação simples aplicada à nuvem para testar essa invariância."""
    pts, ref = make_scan(plumb_mm_per_m=8.0, noise_std=0.001, n_points=15000)
    ang = np.radians(25.0)
    c, s = np.cos(ang), np.sin(ang)
    rotated = pts @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T
    res = run_inspection(rotated, ref)
    assert abs(get(res, "plumb").measured - 8.0) < 1.0


def test_point_cloud_must_already_be_level_or_plumb_is_biased():
    """Limitação conhecida e documentada: o sistema não alinha nem corrige a nuvem recebida; ele
    assume que o eixo Z já é a vertical real (ver premissa em types.py). Se a nuvem inteira
    estiver inclinada — por exemplo, porque o equipamento de captura não estava nivelado —, essa
    inclinação aparece como desaprumo da parede, mesmo que a parede em si esteja perfeitamente
    aprumada. O teste aplica uma rotação simples de 3° à nuvem toda para demonstrar esse viés;
    não há nenhuma simulação de pose de sensor envolvida."""
    pts, ref = make_scan(plumb_mm_per_m=0.0, noise_std=0.001, n_points=15000)
    ang = np.radians(3.0)
    c, s = np.cos(ang), np.sin(ang)
    tilted = pts @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]).T
    res = run_inspection(tilted, ref)
    assert abs(get(res, "plumb").measured) > 10        # ~tan(3°)*1000 ≈ 52 mm/m de viés


def test_line_scan_is_flagged():
    pts, ref = make_scan(line_scan_height=1.2, noise_std=0.001, outlier_frac=0)
    res = run_inspection(pts, ref)
    assert get(res, "height").status == "n/d" and get(res, "plumb").status == "n/d"
    assert any("linha" in w for w in res.warnings)


def test_non_vertical_plane_is_warned():
    """Escanear um piso/teto por engano: o sistema avisa, pois o plano não é vertical."""
    rng = np.random.default_rng(0)
    floor = np.column_stack([rng.uniform(-1, 1, 4000), rng.uniform(-1, 1, 4000),
                             rng.normal(0, 0.001, 4000)])          # plano horizontal (normal ~Z)
    res = run_inspection(floor, ReferenceDimensions(2.0, 2.0))
    assert any("vertical" in w for w in res.warnings)


def test_voxel_downsample_reduces_points():
    pts, _ = make_scan(n_points=20000)
    out = voxel_downsample(pts, 0.05)
    assert 0 < len(out) < len(pts) / 3


def test_clean_drops_nan_and_outliers():
    pts, _ = make_scan(n_points=5000, outlier_frac=0.02)
    pts = np.vstack([pts, [[np.nan, 0, 0]]])
    out = clean(pts, use_open3d=False)
    assert np.isfinite(out).all() and len(out) < len(pts)


def test_ransac_error_explains_low_dominant_plane_support():
    rng = np.random.default_rng(123)
    # Pontos distribuídos em volume: não há uma superfície plana dominante.
    pts = rng.uniform(-1.0, 1.0, size=(250, 3))
    _, ref = make_scan(n_points=500, outlier_frac=0)
    try:
        run_inspection(pts, ref, ransac_threshold=0.001, use_open3d=False)
    except ValueError as exc:
        message = str(exc)
        assert "RANSAC encontrou apenas" in message
        assert "Limiar atual" in message
        assert "unidade do arquivo" in message
    else:
        raise AssertionError("Uma nuvem sem plano dominante deveria gerar diagnóstico de RANSAC.")
