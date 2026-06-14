"""
cli/formatters.py
---------------------
Output formatters for the CLI.  Each formatter accepts a list[ScalpResult] and
returns a string ready for stdout or file writing.
"""

from __future__ import annotations

import csv
import io
import json


from ..core.models import ScalpResult

#---------------------------------------------------------------------

_COLS = ["target", "mni_x", "mni_y", "mni_z", "entry_x", "entry_y", "entry_z", "dist_mm"]
_TAL_COLS = ["target", "tal_x", "tal_y", "tal_z", "mni_x", "mni_y", "mni_z", "entry_x", "entry_y", "entry_z", "dist_mm"]


def _result_row(i: int, r: ScalpResult) -> dict:
    bt = r.brain_target_mni
    se = r.scalp_entry_mni or [None, None, None]
    tal = r.tal_target or [None, None, None]
    return {
        "target":  i,
        "tal_x": round(tal[0], 3) if tal[0] is not None else None,
        "tal_y": round(tal[1], 3) if tal[1] is not None else None,
        "tal_z": round(tal[2], 3) if tal[2] is not None else None,
        "mni_x": round(bt[0], 3), "mni_y": round(bt[1], 3), "mni_z": round(bt[2], 3),
        "entry_x": round(se[0], 3) if se[0] is not None else None,
        "entry_y": round(se[1], 3) if se[1] is not None else None,
        "entry_z": round(se[2], 3) if se[2] is not None else None,
        "dist_mm": round(r.distance_mm, 3) if r.distance_mm is not None else None,
    }


def format_text(results: list[ScalpResult]) -> str:
    """Aligned table: target | [tal_x/y/z |] mni_x/y/z | entry_x/y/z | dist_mm."""
    cols = _TAL_COLS if any(r.is_talairach for r in results) else _COLS
    rows = [_result_row(i, r) for i, r in enumerate(results)]
    col_w = {c: len(c) for c in cols}
    for row in rows:
        for c in cols:
            col_w[c] = max(col_w[c], len(str(row[c])))

    header = "  ".join(f"{c:>{col_w[c]}}" for c in cols)
    lines  = [header]
    for row in rows:
        lines.append("  ".join(f"{str(row[c]):>{col_w[c]}}" for c in cols))
    return "\n".join(lines)


def format_json(results: list[ScalpResult], indent: int = 2) -> str:
    """JSON with nested mni/entry objects."""
    def _to_obj(i: int, r: ScalpResult) -> dict:
        bt = r.brain_target_mni
        se = r.scalp_entry_mni
        obj: dict = {
            "target":      i,
            "mni":         {"x": round(bt[0], 3), "y": round(bt[1], 3), "z": round(bt[2], 3)},
            "entry":       {"x": round(se[0], 3), "y": round(se[1], 3), "z": round(se[2], 3)} if se else None,
            "distance_mm": round(r.distance_mm, 3) if r.distance_mm is not None else None,
        }
        if r.is_talairach and r.tal_target is not None:
            t = r.tal_target
            obj["tal_original"] = {"x": round(t[0], 3), "y": round(t[1], 3), "z": round(t[2], 3)}
        return obj

    objs = [_to_obj(i, r) for i, r in enumerate(results)]
    return json.dumps(objs[0] if len(objs) == 1 else objs, indent=indent)


def format_csv(results: list[ScalpResult]) -> str:
    """CSV with one row per target, same columns as format_text."""
    cols = _TAL_COLS if any(r.is_talairach for r in results) else _COLS
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    for i, result in enumerate(results):
        row = _result_row(i, result)
        writer.writerow([row[c] for c in cols])
    return buf.getvalue()