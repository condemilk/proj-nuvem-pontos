"""Aplicação de tolerâncias: status de uma medida e status geral da inspeção."""
from __future__ import annotations


def status_for(abs_dev: float | None, tol: float | None, warn_fraction: float = 0.8) -> str:
    if abs_dev is None or tol is None or tol <= 0:
        return "n/d"
    if abs_dev > tol:
        return "fora"
    return "atencao" if abs_dev >= warn_fraction * tol else "ok"


def worst_status(statuses) -> str:
    order = {"fora": 3, "atencao": 2, "ok": 1, "n/d": 0}
    return max(statuses, key=lambda s: order.get(s, 0), default="n/d")
