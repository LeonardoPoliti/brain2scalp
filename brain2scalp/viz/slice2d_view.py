"""
viz/slice2d_view.py
-------------------

applied_steps()    – 6-panel pipeline step-by-step view.
visualize_target() – 3-panel orthogonal slice view (sagittal / coronal / axial)
                     centred on the brain target. One fig for target.
"""

from __future__ import annotations

import numpy as np

from ..core.models import PipelineState

# ---------------------------------------------------------------------------
# Color scheme
# Single target: brain = _BRAIN_COLOR_SINGLE, scalp = _SCALP_COLOR_SINGLE
# Multiple targets: each pair shares _COLORS[i]

_BRAIN_COLOR_SINGLE = "#7F77DD"
_SCALP_COLOR_SINGLE = "#EF9F27"

_COLORS = [
    "#7F77DD", "#EF9F27", "#2CA02C", "#D62728",
    "#17BECF", "#9467BD", "#8C564B", "#E377C2",
]


def _brain_color(i: int, n: int) -> str:
    return _BRAIN_COLOR_SINGLE if n == 1 else _COLORS[i % len(_COLORS)]


def _scalp_color(i: int, n: int) -> str:
    return _SCALP_COLOR_SINGLE if n == 1 else _COLORS[i % len(_COLORS)]


def _fmt_mni(arr: np.ndarray) -> str:
    a = np.round(arr, 1)
    return f"[{a[0]:7.1f} {a[1]:7.1f} {a[2]:7.1f}]"


def _target_title_lines(states: list[PipelineState]) -> list[str]:
    """Per-target summary lines (fixed-width coords for monospace alignment)."""
    single = len(states) == 1
    lines = []
    for i, s in enumerate(states):
        assert s.scalp_entry_mni is not None and s.distance_mm is not None
        ct_mni = np.array(s.brain_target_mni)
        cl_mni = np.array(s.scalp_entry_mni)
        coord_label = (
            f"Tal {_fmt_mni(np.asarray(s.tal_target))} → MNI {_fmt_mni(ct_mni)}"
            if s.is_talairach and s.tal_target is not None
            else f"MNI {_fmt_mni(ct_mni)}"
        )
        prefix = f"T{i}: " if not single else ""
        lines.append(
            f"{prefix}{coord_label}  →  entry {_fmt_mni(cl_mni)}"
            f"  |  {s.distance_mm:5.1f} mm  |  {s.projection_method}"
        )
    return lines


# ---------------------------------------------------------------------------


def applied_steps(
    states: list[PipelineState],
    slice_axis: int | None = None,
    save_fig: str | None = None,
    show: bool = True,
) -> None:
    """
    Params:
    states : list[PipelineState]
        One or more full pipeline results. Volumetric data (mask, filled, eroded,
        surface) is taken from the first state, identical across targets that
        share the same NIfTI.
    slice_axis : int or None
        0 = sagittal, 1 = coronal, 2 = axial.
        None = auto-select based on the first target/entry pair.
    save_fig : str or None
        If given, save the figure to this PNG path.
    show : bool
        If True (default), display the figure.
    """

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    if not show:
        plt.switch_backend("Agg")

    # Volumetric data from first state
    s0      = states[0]
    data    = s0.data
    affine  = s0.affine
    mask    = s0.mask
    filled  = s0.filled
    eroded  = s0.eroded
    surface = s0.surface
    n       = len(states)

    assert (
        data is not None and affine is not None and mask is not None
        and filled is not None and eroded is not None and surface is not None
    ), "PipelineState must contain volumetric arrays for visualisation"

    inv = np.linalg.inv(affine)

    def mni_to_vox(pt: np.ndarray) -> np.ndarray:
        return (inv @ np.array([*pt, 1.0]))[:3]

    # Slice position and axis driven by first target
    ct_mni_0 = np.array(s0.brain_target_mni)
    cl_mni_0 = np.array(s0.scalp_entry_mni)
    ct_vox_0 = mni_to_vox(ct_mni_0)
    cl_vox_0 = mni_to_vox(cl_mni_0)

    if slice_axis is None:
        diffs      = np.abs(ct_vox_0 - cl_vox_0)
        slice_axis = int(np.argmin(diffs))

    slice_idx = int(np.clip(np.round(ct_vox_0[slice_axis]), 0,
                            data.shape[slice_axis] - 1))

    def proj2d(mni_pt: np.ndarray):
        vox  = mni_to_vox(mni_pt)
        axes = [0, 1, 2]
        axes.remove(slice_axis)
        col_ax, row_ax = axes[0], axes[1]
        return float(vox[col_ax]), float(vox[row_ax])

    # Pre-compute per-target 2D geometry
    geom = []
    for state in states:
        assert state.scalp_entry_mni is not None and state.distance_mm is not None, \
            "PipelineState missing scalp_entry_mni or distance_mm"
        ct_mni    = np.array(state.brain_target_mni)
        cl_mni    = np.array(state.scalp_entry_mni)
        ct_2d     = proj2d(ct_mni)
        cl_2d     = proj2d(cl_mni)
        direction = cl_mni - ct_mni
        norm      = np.linalg.norm(direction)
        ext_mni   = cl_mni + (direction / norm) * 18.0 if norm > 1e-6 else cl_mni
        ext_2d    = proj2d(ext_mni)
        geom.append(dict(
            ct_mni=ct_mni, cl_mni=cl_mni,
            ct_2d=ct_2d, cl_2d=cl_2d, ext_2d=ext_2d,
            dist=state.distance_mm, method=state.projection_method,
        ))

    def sl(vol: np.ndarray) -> np.ndarray:
        if slice_axis == 0:
            return vol[slice_idx, :, :]
        if slice_axis == 1:
            return vol[:, slice_idx, :]
        return vol[:, :, slice_idx]

    raw_sl  = sl(data)
    mask_sl = sl(mask)
    fill_sl = sl(filled)
    ero_sl  = sl(eroded)
    surf_sl = sl(surface)

    _, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="#0e1117")

    _edge_labels = {
        0: dict(h_neg="P", h_pos="A", v_neg="I", v_pos="S"),
        1: dict(h_neg="L", h_pos="R", v_neg="I", v_pos="S"),
        2: dict(h_neg="L", h_pos="R", v_neg="P", v_pos="A"),
    }[slice_axis]
    _edge_kw = dict(
        fontsize=8, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="#0e1117",
                  edgecolor="none", alpha=0.7),
    )

    def add_edge_labels(ax) -> None:
        for txt, x, y, ha, va in [
            (_edge_labels["h_neg"], 0.01, 0.50, "left",   "center"),
            (_edge_labels["h_pos"], 0.99, 0.50, "right",  "center"),
            (_edge_labels["v_neg"], 0.50, 0.01, "center", "bottom"),
            (_edge_labels["v_pos"], 0.50, 0.99, "center", "top"),
        ]:
            ax.text(x, y, txt, transform=ax.transAxes,
                    ha=ha, va=va, **_edge_kw)

    imopts = dict(origin="lower", aspect="auto")
    ax_flat = axes.flatten()

    def style_ax(ax, title: str) -> None:
        ax.set_title(title, color="#aaaaaa", fontsize=10, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#333344")

    def mark_all_brains(ax, alpha: float = 1.0) -> None:
        for i, g in enumerate(geom):
            ax.plot(*g["ct_2d"], "o", ms=8,
                    color=_brain_color(i, n),
                    markeredgecolor="white", markeredgewidth=1,
                    zorder=10, alpha=alpha)

    def mark_all_scalps(ax, alpha: float = 1.0) -> None:
        for i, g in enumerate(geom):
            ax.plot(*g["cl_2d"], "o", ms=9,
                    color=_scalp_color(i, n),
                    markeredgecolor="white", markeredgewidth=1.2,
                    zorder=10, alpha=alpha)

    def draw_all_projections(ax) -> None:
        for i, g in enumerate(geom):
            c = _scalp_color(i, n)
            ax.plot([g["ct_2d"][0], g["ext_2d"][0]],
                    [g["ct_2d"][1], g["ext_2d"][1]],
                    "--", color=c, linewidth=1.2, alpha=0.85, zorder=9)
            ax.plot([g["ct_2d"][0], g["cl_2d"][0]],
                    [g["ct_2d"][1], g["cl_2d"][1]],
                    "-",  color=c, linewidth=1.8, alpha=0.95, zorder=9)

    ax = ax_flat[0]
    im = ax.imshow(raw_sl.T, cmap="gray", **imopts)
    style_ax(ax, "1 · Raw NIfTI")
    mark_all_brains(ax, alpha=0.7)
    add_edge_labels(ax)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.yaxis.set_tick_params(
        color="gray", labelcolor="gray"
    )

    ax = ax_flat[1]
    ax.imshow(mask_sl.T, cmap=ListedColormap(["#0e1117", "#1D9E75"]),
              vmin=0, vmax=1, **imopts)
    style_ax(ax, f"2 · Binary head mask  (thresh > {s0.threshold_used:.1f})")
    mark_all_brains(ax, alpha=0.7)

    ax = ax_flat[2]
    ax.imshow(fill_sl.T, cmap=ListedColormap(["#0e1117", "#0F6E56"]),
              vmin=0, vmax=1, **imopts)
    style_ax(ax, "3 · Filled head mask")
    mark_all_brains(ax, alpha=0.7)

    ax = ax_flat[3]
    ero_rgb = np.zeros((*fill_sl.shape, 3))
    ero_rgb[fill_sl == 1] = [0.58, 0.92, 0.72]
    ero_rgb[ero_sl  == 1] = [0.12, 0.62, 0.47]  
    ax.imshow(np.transpose(ero_rgb, (1, 0, 2)), origin="lower", aspect="auto")
    style_ax(ax, "4 · Eroded head mask")
    mark_all_brains(ax, alpha=0.7)
    p1 = mpatches.Patch(color="#94EBB8", label="shell (removed)")
    p2 = mpatches.Patch(color="#1E9E78", label="interior (kept)")
    ax.legend(handles=[p1, p2], loc="lower right", fontsize=7,
              facecolor="#1a1a2e", edgecolor="none", labelcolor="gray")

    ax = ax_flat[4]
    surf_rgb = np.zeros((*surf_sl.shape, 3))
    surf_rgb[surf_sl == 1] = [0.85, 0.35, 0.19]
    ax.imshow(np.transpose(surf_rgb, (1, 0, 2)), origin="lower", aspect="auto")
    style_ax(ax, f"5 · Surface shell  (filled − eroded , {s0.surface_thickness_mm:.1f} mm)")
    mark_all_brains(ax, alpha=0.7)

    ax = ax_flat[5]
    ax.imshow(raw_sl.T, cmap="gray", alpha=0.55, **imopts)
    surf_rgba = np.zeros((*surf_sl.shape, 4))
    surf_rgba[surf_sl == 1] = [0.85, 0.35, 0.19, 0.9]
    ax.imshow(np.transpose(surf_rgba, (1, 0, 2)), origin="lower", aspect="auto")
    draw_all_projections(ax)
    mark_all_brains(ax)
    mark_all_scalps(ax)
    style_ax(ax, "6 · Result: entry point + projection")

    if n > 1:
        for i, _ in enumerate(geom):
            ax.text(
                0.01 + i * 0.06, 0.97, f"T{i}",
                transform=ax.transAxes, color=_brain_color(i, n),
                fontsize=8, fontweight="bold", va="top",
                bbox=dict(facecolor="#0e1117", edgecolor="none", alpha=0.6, pad=1),
            )

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches="tight", facecolor="#0e1117")
        print(f"  Figure saved to: {save_fig}")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close("all")


# ---------------------------------------------------------------------------


def visualize_target(
    states: list[PipelineState],
    save_fig: str | None = None,
    show: bool = True,
) -> None:
    """
    Params:
    states : list[PipelineState]
        One or more full pipeline results.
    save_fig : str or None
        If given, save the figure to this PNG path.
    show : bool
        If True (default), display the figure.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if not show:
        plt.switch_backend("Agg")

    s0     = states[0]
    data   = s0.data
    affine = s0.affine
    n      = len(states)

    assert data is not None and affine is not None, \
        "PipelineState must contain volumetric arrays for visualisation"

    inv = np.linalg.inv(affine)

    def _to_vox(pt: np.ndarray) -> np.ndarray:
        return (inv @ np.array([*pt, 1.0]))[:3]

    # Slice positions: centroid of all brain targets
    all_ct_mni   = np.array([s.brain_target_mni for s in states])
    centroid_mni = all_ct_mni.mean(axis=0)
    centroid_vox = _to_vox(centroid_mni)
    nx, ny, nz   = data.shape
    xi = int(np.clip(round(centroid_vox[0]), 0, nx - 1))
    yi = int(np.clip(round(centroid_vox[1]), 0, ny - 1))
    zi = int(np.clip(round(centroid_vox[2]), 0, nz - 1))

    def _mm_ticks(vox_ax: int, nn: int, n_ticks: int = 6):
        v  = np.linspace(0, nn - 1, n_ticks)
        mm = affine[vox_ax, vox_ax] * v + affine[vox_ax, 3]
        return v, mm

    ticks = {i: _mm_ticks(i, s) for i, s in enumerate([nx, ny, nz])}

    panels = [
        dict(
            label="Sagittal", axis_label="x", mni_mm=centroid_mni[0],
            sl=data[xi, :, :],
            col_ax=1, row_ax=2,
            xlabel="← P   mm   A →", ylabel="← I   mm   S →",
        ),
        dict(
            label="Coronal", axis_label="y", mni_mm=centroid_mni[1],
            sl=data[:, yi, :],
            col_ax=0, row_ax=2,
            xlabel="← L   mm   R →", ylabel="← I   mm   S →",
        ),
        dict(
            label="Axial", axis_label="z", mni_mm=centroid_mni[2],
            sl=data[:, :, zi],
            col_ax=0, row_ax=1,
            xlabel="← L   mm   R →", ylabel="← P   mm   A →",
        ),
    ]

    # Per-target voxel positions
    target_voxs = [_to_vox(np.array(s.brain_target_mni)) for s in states]

    target_lines = _target_title_lines(states)
    if n > 1:
        suptitle = "\n".join(target_lines)
    else:
        suptitle = target_lines[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0e1117")
    fig.suptitle(suptitle, color="white", fontsize=10, family="monospace")

    for ax, cfg in zip(axes, panels):
        col_ax, row_ax = cfg["col_ax"], cfg["row_ax"]
        vox_col, mm_col = ticks[col_ax]
        vox_row, mm_row = ticks[row_ax]

        ax.imshow(cfg["sl"].T, cmap="gray", origin="lower", aspect="auto")

        ax.set_xticks(vox_col)
        ax.set_xticklabels([f"{v:.0f}" for v in mm_col], color="#666", fontsize=7)
        ax.set_yticks(vox_row)
        ax.set_yticklabels([f"{v:.0f}" for v in mm_row], color="#666", fontsize=7)
        ax.tick_params(colors="#444", length=3, width=0.5)
        ax.set_xlabel(cfg["xlabel"], color="#999", fontsize=9, labelpad=4)
        ax.set_ylabel(cfg["ylabel"], color="#999", fontsize=9, labelpad=4)

        # Crosshair only for single target
        if n == 1:
            ct_vox = target_voxs[0]
            ct_col = float(ct_vox[col_ax])
            ct_row = float(ct_vox[row_ax])
            ax.axvline(ct_col, color=_BRAIN_COLOR_SINGLE, lw=0.8, ls="--",
                       alpha=0.5, zorder=5)
            ax.axhline(ct_row, color=_BRAIN_COLOR_SINGLE, lw=0.8, ls="--",
                       alpha=0.5, zorder=5)

        # Dot for each target
        for i, ct_vox in enumerate(target_voxs):
            ct_col = float(ct_vox[col_ax])
            ct_row = float(ct_vox[row_ax])
            ax.plot(ct_col, ct_row, "o",
                    ms=9, color=_brain_color(i, n),
                    markeredgecolor="white", markeredgewidth=1.2, zorder=10)

        ax.set_title(
            f"{cfg['label']}  ({cfg['axis_label']} = {cfg['mni_mm']:.1f} mm)",
            color="#aaaaaa", fontsize=10, pad=5,
        )
        ax.set_facecolor("#0e1117")
        for sp in ax.spines.values():
            sp.set_color("#333344")

    # Legend for multiple targets
    if n > 1:
        leg_items = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_brain_color(i, n), markersize=9,
                   label=f"T{i}: {np.round(s.brain_target_mni, 1)}")
            for i, s in enumerate(states)
        ]
        axes[-1].legend(handles=leg_items, loc="lower right", fontsize=7,
                        facecolor="#1a1a2e", edgecolor="none", labelcolor="#cccccc")

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches="tight", facecolor="#0e1117")
        print(f"  Target view saved to: {save_fig}")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close("all")