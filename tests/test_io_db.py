import io
import tempfile
from pathlib import Path

import numpy as np

from app.db import repository as repo
from app.geometry.analyze import result_to_dict, run_inspection
from app.geometry.synthetic import make_scan
from app.ingest.point_cloud import InputError, load_point_cloud


def _write(tmp, name, text):
    p = Path(tmp) / name
    p.write_text(text)
    return p


def _grid_text(sep=" ", n=200, scale=100.0):
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 3, n), rng.normal(0, .002, n), rng.uniform(0, 2.5, n)]) * scale
    return pts, "\n".join(sep.join(f"{v:.4f}" for v in r) for r in pts)


def test_xyz_with_extra_columns_and_unit_conversion():
    with tempfile.TemporaryDirectory() as t:
        pts, txt = _grid_text(scale=100.0)                       # arquivo em cm
        txt = "\n".join(l + " 255 0 0" for l in txt.splitlines())  # colunas extras (RGB)
        c = load_point_cloud(_write(t, "a.xyz", txt), unit="cm")
        assert c.points.shape == (200, 3) and abs(c.points[:, 0].max() - pts[:, 0].max() / 100) < 1e-6


def test_csv_with_header_and_semicolon_and_column_order():
    with tempfile.TemporaryDirectory() as t:
        txt = "z;y;x\n" + "\n".join(f"{i * .01};{0.1};{i * .02}" for i in range(100))
        c = load_point_cloud(_write(t, "b.csv", txt))
        assert abs(c.points[-1, 0] - 99 * .02) < 1e-9 and abs(c.points[-1, 2] - 99 * .01) < 1e-9


def test_ascii_pcd():
    with tempfile.TemporaryDirectory() as t:
        pts, _ = _grid_text(scale=1.0)
        body = "\n".join(" ".join(f"{v:.5f}" for v in r) for r in pts)
        hdr = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
               f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA ascii\n")
        c = load_point_cloud(_write(t, "c.pcd", hdr + body))
        assert len(c.points) == len(pts)


def test_invalid_inputs_raise_readable_errors():
    with tempfile.TemporaryDirectory() as t:
        for name, txt in [("d.xyz", "isto não é nuvem\nnem isto"), ("e.xyz", "1 2 3\n4 5 6"),
                          ("f.docx", "x"), ("g.xyz", "\n".join(["1 1 1"] * 50))]:
            try:
                load_point_cloud(_write(t, name, txt))
            except InputError:
                continue
            raise AssertionError(f"{name} deveria falhar")


def test_unit_warning():
    with tempfile.TemporaryDirectory() as t:
        _, txt = _grid_text(scale=1.0)                            # ~3 m, mas declarado em mm → 3 mm
        c = load_point_cloud(_write(t, "h.xyz", txt), unit="mm")
        assert any("unidade" in w for w in c.warnings)


def test_db_roundtrip_history_and_csv():
    with tempfile.TemporaryDirectory() as t:
        db = Path(t) / "x.db"
        repo.init_db(db)
        for plumb in (2.0, 6.0, 12.0):
            pts, ref = make_scan(plumb_mm_per_m=plumb, noise_std=0.001, n_points=8000)
            res = result_to_dict(run_inspection(pts, ref), include_viz=False)
            with repo.connect(db) as c:
                repo.save_inspection(c, res, name=f"p{plumb}", source_file="a.xyz", element=ref.label)
        with repo.connect(db) as c:
            h = repo.history(c, "plumb")
            assert len(h) == 3 and h["measured"].is_monotonic_increasing
            assert len(repo.list_scans(c)) == 3
            buf = io.StringIO()
            repo.export_csv(c, buf)
            lines = buf.getvalue().splitlines()
            assert lines[0].startswith("\ufeffscan_id;project") or lines[0].startswith("scan_id;project")
            assert len(lines) == 1 + 3 * 3             # 3 medidas por inspeção (largura, altura, desaprumo)


def test_db_migration_drops_mode_column_without_losing_rows():
    """Um banco de uma versão anterior (com a coluna `mode`, de quando havia best_fit/pose) deve
    continuar funcionando e preservar as linhas já gravadas, mesmo sem a coluna."""
    with tempfile.TemporaryDirectory() as t:
        db = Path(t) / "legacy.db"
        conn = __import__("sqlite3").connect(str(db))
        conn.execute("CREATE TABLE scans (id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT NOT NULL DEFAULT "
                    "'default', name TEXT, created_at TEXT NOT NULL, source_file TEXT, element TEXT, "
                    "mode TEXT, n_points INTEGER, overall_status TEXT, params_json TEXT)")
        conn.execute("INSERT INTO scans (project,name,created_at,source_file,element,mode,n_points,"
                    "overall_status,params_json) VALUES ('default','antiga','2026-01-01T00:00:00',"
                    "'velho.xyz','parede (manual)','best_fit',100,'ok','{}')")
        conn.execute("CREATE TABLE measurements (id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER, "
                    "key TEXT, label TEXT, unit TEXT, measured REAL, bim REAL, deviation REAL, "
                    "tolerance REAL, status TEXT)")
        conn.commit(); conn.close()

        repo.init_db(db)                                # deve migrar sem apagar a linha antiga
        with repo.connect(db) as c:
            scans = repo.list_scans(c)
            assert len(scans) == 1 and scans.iloc[0]["name"] == "antiga"
            assert "mode" not in scans.columns           # coluna obsoleta removida (ou nunca populada de novo)
            cols = {row[1] for row in c.execute("PRAGMA table_info(measurements)")}
            assert "reference_value" in cols and "bim" not in cols
