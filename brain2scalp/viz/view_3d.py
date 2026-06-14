"""
viz/view_3d.py
--------------
Interactive 3D viewer with target-to-scalp projection.
"""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path

import numpy as np

from ..core.models import PipelineState

#-----------------------------------------------------------------------------------

# One color per target pair (brain + scalp entry share the same color).
_BRAIN_COLORS = [
    "#7F77DD", "#EF9F27", "#2CA02C", "#D62728",
    "#17BECF", "#9467BD", "#8C564B", "#E377C2",
]
MAX_3D_TARGETS = len(_BRAIN_COLORS)  # hard cap to prevent visualization disruption 
_SCALP_COLOR_SINGLE = "#EF9F27"
_DEFAULT_ISOMIN_FRAC = 1 / 3  # initial isomin as a fraction of the data max

#-----------------------------------------------------------------------------------

def _volume_html(fig, vmin: float, vmax: float, default_isomin: float, surface_idx: int = -1) -> str:
    fig.update_layout(autosize=True)
    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="vol")
    has_surface = surface_idx >= 0
    surf_html = ""
    if has_surface:
        surf_html = f"""
  <div class="sep"></div>
  <div style="text-align:center;">
    <button id="surf-btn" onclick="toggleSurface(this)">Show Surface</button>
  </div>
  <div class="ctrl-group" id="surf-alpha-row" style="display:none;">
    <div class="ctrl-header">
      <span>Surface opacity (0–1)</span>
      <span class="val" id="sov">0.2</span>
    </div>
    <input type="range" min="1" max="100" value="20" step="1"
           oninput="let v=(+this.value/100).toFixed(2);
                    document.getElementById('sov').textContent=v;
                    Plotly.restyle('vol',{{'marker.opacity':[+v]}},[{surface_idx}])">
  </div>"""
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ height: 100%; background: #0e1117; overflow: hidden; font-family: sans-serif; }}
body {{ position: relative; }}
#plot {{ position: absolute; inset: 0; display: flex; flex-direction: column; }}
#vol {{ flex: 1; min-height: 0; }}
#vol > div {{ height: 100% !important; }}
#controls {{
  position: absolute; z-index: 10; visibility: hidden;
  background: rgba(14,17,23,0.85); border-radius: 7px;
  padding: 14px 16px; display: flex; flex-direction: column; gap: 12px;
  min-width: 230px;
}}
.ctrl-group {{ display: flex; flex-direction: column; gap: 9px; }}
.ctrl-header {{
  display: flex; justify-content: space-between; align-items: baseline;
  color: #aaa; font-size: 11px;
}}
.ctrl-header .val {{ color: #eee; font-variant-numeric: tabular-nums; }}
input[type=range] {{
  -webkit-appearance: none; appearance: none;
  width: 100%; cursor: pointer; height: 4px;
  background: #444; border-radius: 2px; outline: none;
}}
input[type=range]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 11px; height: 11px; border-radius: 50%;
  background: #5DCAA5; cursor: pointer;
}}
input[type=range]::-moz-range-thumb {{
  width: 11px; height: 11px; border-radius: 50%;
  background: #5DCAA5; cursor: pointer; border: none;
}}
input[type=range]::-webkit-slider-runnable-track {{
  background: #444; border-radius: 2px; height: 4px;
}}
.sep {{ border-top: 1px solid #333; }}
#surf-btn {{
  background: #1e2533; color: #aaa; border: 1px solid #444;
  border-radius: 5px; padding: 4px 16px; cursor: pointer;
  font-size: 11px;
}}
#surf-btn.active {{ background: #1a3a30; border-color: #5DCAA5; color: #5DCAA5; }}
</style></head>
<body>
<div id="plot">{plot_div}</div>
<div id="controls">
  <div class="ctrl-group">
    <div class="ctrl-header">
      <span>Threshold ({vmin:.0f}–{vmax:.0f})</span>
      <span class="val" id="tv">{default_isomin:.0f}</span>
    </div>
    <input type="range"
           min="{vmin:.1f}" max="{vmax:.1f}" value="{default_isomin:.0f}" step="0.5"
           oninput="document.getElementById('tv').textContent=(+this.value).toFixed(0);
                    Plotly.restyle('vol',{{isomin:[+this.value]}},[0])">
  </div>
  <div class="ctrl-group">
    <div class="ctrl-header">
      <span>Opacity (0–1)</span>
      <span class="val" id="ov">0.15</span>
    </div>
    <input type="range" min="1" max="100" value="15" step="1"
           oninput="let v=(+this.value/100).toFixed(2);
                    document.getElementById('ov').textContent=v;
                    Plotly.restyle('vol',{{opacity:[+v]}},[0])">
  </div>{surf_html}
</div>
<script>
function toggleSurface(btn) {{
  var on = btn.classList.toggle('active');
  Plotly.restyle('vol', {{visible: [on]}}, [{surface_idx}]);
  document.getElementById('surf-alpha-row').style.display = on ? 'flex' : 'none';
}}
function positionControls() {{
  var legend = document.querySelector('.legend');
  var ctrl   = document.getElementById('controls');
  if (!legend || !ctrl) return;
  var r   = legend.getBoundingClientRect();
  var gap = window.innerWidth - r.right;
  ctrl.style.right      = Math.max(gap, 16) + 'px';
  ctrl.style.left       = 'auto';
  ctrl.style.top        = (r.bottom + 24) + 'px';
  ctrl.style.width      = Math.max(r.width, 230) + 'px';
  ctrl.style.visibility = 'visible';
}}
window.addEventListener('load', function() {{
  var h = document.getElementById('plot').clientHeight;
  var w = document.getElementById('plot').clientWidth;
  Plotly.relayout('vol', {{height: h, width: w}});
  setTimeout(positionControls, 400);
  window.addEventListener('resize', function() {{
    Plotly.relayout('vol', {{
      height: document.getElementById('plot').clientHeight,
      width:  document.getElementById('plot').clientWidth,
    }});
    setTimeout(positionControls, 400);
  }});
}});
</script>
</body></html>"""


def visualize_volume(
    states: list[PipelineState],
    save_html: str | None = None,
    show: bool = True,
    stride: int = 3,
) -> None:
    """
    Interactive 3D volume render with threshold and opacity sliders.

    Params:
    states : list[PipelineState]
    save_html : str or None - save to HTML if given
    show : bool - open in browser
    stride : int - subsample every N voxels per axis (default 3)
    """
    import plotly.graph_objects as go

    if len(states) > MAX_3D_TARGETS:
        print(
            f"  Warning: --viz3d supports at most {MAX_3D_TARGETS} targets; "
            f"truncating from {len(states)} to {MAX_3D_TARGETS}."
        )
        states = states[:MAX_3D_TARGETS]

    s0 = states[0]
    assert s0.data is not None and s0.affine is not None, \
        "PipelineState must contain volumetric data for volume rendering"

    data   = s0.data
    affine = s0.affine
    single = len(states) == 1

    # Subsample voxel grid
    nx, ny, nz = data.shape
    I, J, K = np.meshgrid(
        np.arange(0, nx, stride),
        np.arange(0, ny, stride),
        np.arange(0, nz, stride),
        indexing="ij",
    )
    vals = data[I, J, K].ravel().astype(float)
    vox  = np.stack([I.ravel(), J.ravel(), K.ravel(), np.ones(I.size)], axis=1)
    mm   = (affine @ vox.T).T
    X, Y, Z = mm[:, 0], mm[:, 1], mm[:, 2]

    keep          = vals > 0
    X, Y, Z, vals = X[keep], Y[keep], Z[keep], vals[keep]
    if not len(vals):
        print("  Warning: no non-zero voxels in subsampled volume; try a lower --volume-stride.")
        return
    vmin          = float(vals.min())
    vmax          = float(vals.max())
    default_isomin = vmax * _DEFAULT_ISOMIN_FRAC

    fig = go.Figure()

    # Volume trace - index 0, targeted by both sliders
    fig.add_trace(go.Volume(
        x=X, y=Y, z=Z,
        value=vals,
        isomin=default_isomin, isomax=vmax,
        opacity=0.15,
        surface_count=10,
        colorscale="gray",
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="MRI volume",
        showlegend=True,
    ))

    # Target + entry overlays
    for i, state in enumerate(states):
        color       = _BRAIN_COLORS[i % len(_BRAIN_COLORS)]
        entry_color = _SCALP_COLOR_SINGLE if single else color
        prefix      = f"T{i}: " if not single else ""

        assert state.scalp_entry_mni is not None and state.distance_mm is not None
        ct_mni = np.array(state.brain_target_mni)
        cl_mni = np.array(state.scalp_entry_mni)
        dist   = state.distance_mm

        direction = cl_mni - ct_mni
        norm      = np.linalg.norm(direction)
        ext_mni   = cl_mni + (direction / norm) * 20.0 if norm > 1e-6 else cl_mni

        fig.add_trace(go.Scatter3d(
            x=[ct_mni[0], cl_mni[0]], y=[ct_mni[1], cl_mni[1]], z=[ct_mni[2], cl_mni[2]],
            mode="lines", line=dict(color=entry_color, width=5),
            name=f"{prefix}Projection ({dist:.1f} mm, {state.projection_method})",
        ))
        fig.add_trace(go.Scatter3d(
            x=[cl_mni[0], ext_mni[0]], y=[cl_mni[1], ext_mni[1]], z=[cl_mni[2], ext_mni[2]],
            mode="lines", line=dict(color=entry_color, width=2, dash="dash"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[ct_mni[0]], y=[ct_mni[1]], z=[ct_mni[2]],
            mode="markers",
            marker=dict(size=6, color=color, symbol="circle", line=dict(color="white", width=1)),
            name=f"{prefix}Brain target  {np.round(ct_mni, 1)}",
        ))
        fig.add_trace(go.Scatter3d(
            x=[cl_mni[0]], y=[cl_mni[1]], z=[cl_mni[2]],
            mode="markers",
            marker=dict(size=6, color=entry_color, symbol="circle", line=dict(color="white", width=1)),
            name=f"{prefix}Scalp entry  {np.round(cl_mni, 1)}",
        ))

    # Scalp surface overlay (hidden by default, toggled from UI)
    surface_trace_idx = -1
    scalp_mni = states[0].scalp_surface_mni
    if scalp_mni is not None:
        sub = np.arange(0, len(scalp_mni), 2)
        sp  = scalp_mni[sub]
        surface_trace_idx = 1 + 4 * len(states)
        fig.add_trace(go.Scatter3d(
            x=sp[:, 0], y=sp[:, 1], z=sp[:, 2],
            mode="markers",
            marker=dict(size=1.0, color="#5DCAA5", opacity=0.2),
            name="Scalp surface",
            hoverinfo="skip",
            visible=False,
        ))

    fig.update_layout(
        paper_bgcolor="#0e1117",
        scene=dict(
            bgcolor="#0e1117",
            xaxis=dict(title="Sagittal (mm)", color="#888", gridcolor="#333", showbackground=False),
            yaxis=dict(title="Coronal (mm)", color="#888", gridcolor="#333", showbackground=False),
            zaxis=dict(title="Axial (mm)", color="#888", gridcolor="#333", showbackground=False),
            aspectmode="data",
        ),
        legend=dict(font=dict(color="#cccccc", size=11), bgcolor="rgba(20,20,30,0.8)",
                    bordercolor="#444", borderwidth=1),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    html = _volume_html(fig, float(vmin), float(vmax), default_isomin, surface_trace_idx)

    if save_html:
        Path(save_html).write_text(html, encoding="utf-8")
        print(f"  Volume viewer saved to: {save_html}")
    if show:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
            f.write(html)
            webbrowser.open(f"file://{f.name}")