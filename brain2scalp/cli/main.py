"""
cli/main.py
-------------------
Command-line interface for brain2scalp.
"""

from __future__ import annotations

import csv
import sys
import argparse
from pathlib import Path

from ..core.pipeline import run_full_multi, run_multi
from .formatters import format_csv, format_json, format_text

#-----------------------------------------------------------------------------------------

_EXT_FORMAT: dict[str, str] = {".json": "json", ".csv": "csv", ".txt": "txt"}

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brain2scalp",
        description="Project MNI, Talairach or native-space brain coordinates "
                    "to scalp entry points on NIfTI head scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # INPUT
    inp = p.add_argument_group("input")
    inp.add_argument(
        "--nii", required=True, metavar="PATH",
        help="Path to .nii or .nii.gz file (must contain scalp/skull).",
    )
    inp.add_argument(
        "--target", type=str, default=None, metavar="COORDS",
        help=(
            "Brain target coordinates in mm.  "
            "Single: '10 20 30'  "
            "Multiple (comma-separated): '10 20 30, -10 5 45' "
        ),
    )
    inp.add_argument(
        "--target-file", metavar="PATH", default=None,
        help=(
            "CSV file containing one target per row (x,y,z).  "
            "An optional header row is detected and skipped automatically.  "
            "When combined with --target, file targets come first and CLI targets "
            "are appended; duplicate coordinates are silently dropped."
        ),
    )
    inp.add_argument(
        "--space", default="mni", choices=["mni", "talairach", "native"],
        help=(
            "Coordinate space of --target (default: mni).  "
            "'talairach': convert to MNI via Lancaster et al. (2007) before processing.  "
            "'native': target is in the scan's own mm space. Not registered to MNI."
        ),
    )

    # PROCESSING
    proc = p.add_argument_group("processing")
    proc.add_argument(
        "--threshold", type=float, default=None,
        help="Override auto intensity threshold for head mask.",
    )
    proc.add_argument(
        "--surface-thickness", type=float, default=1.5, metavar="MM",
        help="Surface shell thickness in mm used for erosion (default: 1.5).",
    )
    proc.add_argument(
        "--k-local", type=int, default=100, metavar="K",
        help=(
            "Number of nearest scalp voxels used to estimate the local ray "
            "direction.  Smaller = more local, larger = smoother (default: 100)."
        ),
    )

    # OUTPUT
    out = p.add_argument_group("output")
    out.add_argument(
        "--output", "-o", default=None, metavar="PATH",
        help="Save result to this file (.txt, .json, or .csv).",
    )
    out.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output.",
    )

    # VISUALIZATION
    viz = p.add_argument_group("visualization")
    viz.add_argument(
        "--vizSteps", dest="viz_steps", nargs="*", type=int, metavar="IDX",
        help=(
            "Show 6-panel step-by-step pipeline figure.  "
            "Pass target indices to select which targets are shown (default: all), "
            "e.g. --vizSteps 0 2."
        ),
    )
    viz.add_argument(
        "--step-axis", type=int, default=None, choices=[0, 1, 2],
        help="Anatomical axis for --vizSteps: 0=sagittal, 1=coronal, 2=axial (default: auto).",
    )
    viz.add_argument(
        "--vizTarget", dest="viz_target", nargs="*", type=int, metavar="IDX",
        help=(
            "Show 3-panel orthogonal slice view centred on the brain target.  "
            "Pass target indices to limit which targets are shown (default: all), "
            "e.g. --vizTarget 1."
        ),
    )
    viz.add_argument(
        "--viz3d", nargs="*", type=int, metavar="IDX",
        help=(
            "Open interactive 3D volume viewer in browser.  "
            "Pass target indices to limit which targets are shown (default: all), "
            "e.g. --viz3d 0 1."
        ),
    )
    viz.add_argument(
        "--volume-stride", type=_positive_int, default=3, metavar="N",
        help=(
            "Subsample every Nth voxel per axis for the 3D volume render (default: 3). "
            "Lower = denser point cloud, slower browser render and higher memory use. "
            "Higher = faster and lighter, but less details."
        ),
    )
    viz.add_argument(
        "--save-fig", default=None, metavar="DIR",
        help=(
            "Save all enabled visualizations to this directory. "
            "Created automatically if it does not exist."
        ),
    )
    viz.add_argument(
        "--no-plot", dest="no_plot", action="store_true",
        help="Suppress display of all figures (combine with --save-fig to save without showing).",
    )

    return p


# ---------------------------------------------------------------------------
# Argument type helpers

def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


# ---------------------------------------------------------------------------
# Target parsing helpers

def _parse_target_str(raw: str) -> list[list[float]]:
    """Parse comma-separated triplets from --target.

    '10 20 30'            -> [[10, 20, 30]]
    '10 20 30, -10 5 45'  -> [[10, 20, 30], [-10, 5, 45]]
    """
    targets = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        if len(tokens) != 3:
            raise argparse.ArgumentTypeError(
                f"each triplet must have exactly 3 values; got {len(tokens)} in {part!r}"
            )
        targets.append([float(t) for t in tokens])
    if not targets:
        raise argparse.ArgumentTypeError("--target produced no valid coordinates")
    return targets


def _parse_target_file(path: str) -> list[list[float]]:
    """Load targets from a CSV file (x,y,z per row, optional header)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for i, raw_row in enumerate(reader):
            cells = [c.strip() for c in raw_row]
            if not any(cells):
                continue  # blank line
            try:
                if len(cells) < 3:
                    raise ValueError(f"expected 3 columns, got {len(cells)}")
                vals = [float(v) for v in cells[:3]]
                rows.append(vals)
            except ValueError:
                if i == 0:
                    continue  # skip header row
                raise ValueError(
                    f"could not parse row {i + 1} of {path!r} as three floats: {raw_row}"
                )
    if not rows:
        raise ValueError(f"no valid target rows found in {path!r}")
    return rows


def _merge_targets(
    file_targets: list[list[float]],
    cli_targets: list[list[float]],
) -> list[list[float]]:
    """Combine file + CLI targets, dropping duplicates (first occurrence remain)."""
    seen: set[tuple[float, ...]] = set()
    merged: list[list[float]] = []
    for t in file_targets + cli_targets:
        key = tuple(t)
        if key not in seen:
            seen.add(key)
            merged.append(t)
    return merged


def _resolve_indices(flag: list[int] | None, n: int) -> list[int]:
    """Indices to visualize for a viz flag.

    None  -> []
    []    -> flag passed with no indices, visualize all
    [i…]  -> visualize only the given indices
    """
    if flag is None:
        return []
    return list(range(n)) if not flag else flag


# ---------------------------------------------------------------------------
# Output helpers

def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.parent / f"{path.stem}_{i}{path.suffix}"
        if not candidate.exists():
            return candidate
        i += 1


#==============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    # Validate output extension
    if args.output is not None:
        out_path = Path(args.output)
        if out_path.is_dir():
            parser.error(
                f"--output must be a file path, not a directory: {out_path}\n"
            )
        if _EXT_FORMAT.get(out_path.suffix.lower()) is None:
            parser.error(
                f"unrecognized output extension {out_path.suffix!r}. Use .txt, .json, or .csv."
            )

    # Collect and merge targets
    if args.target is None and args.target_file is None:
        parser.error("at least one of --target or --target-file is required")

    try:
        file_targets: list[list[float]] = (
            _parse_target_file(args.target_file) if args.target_file else []
        )
        cli_targets: list[list[float]] = (
            _parse_target_str(args.target) if args.target else []
        )
    except (ValueError, argparse.ArgumentTypeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    targets = _merge_targets(file_targets, cli_targets)
    n       = len(targets)

    needs_full = any(x is not None for x in [args.viz_steps, args.viz_target, args.viz3d])
    verbose    = not args.quiet

    # Validate viz indices
    for flag_name, flag_val in [
        ("--vizSteps", args.viz_steps),
        ("--vizTarget", args.viz_target),
        ("--viz3d", args.viz3d),
    ]:
        if flag_val:
            bad = [i for i in flag_val if not (0 <= i < n)]
            if bad:
                label = "indices" if len(bad) > 1 else "index"
                print(
                    f"error: {flag_name} {label} {bad} out of range for "
                    f"{n} target{'s' if n > 1 else ''} (valid: 0–{n - 1})",
                    file=sys.stderr,
                )
                return 1

    save_dir: Path | None = None
    if args.save_fig is not None:
        save_dir = Path(args.save_fig)
        save_dir.mkdir(parents=True, exist_ok=True)

    _pipeline_kwargs = dict(
        threshold=args.threshold,
        is_talairach=args.space == "talairach",
        is_native=args.space == "native",
        surface_thickness_mm=args.surface_thickness,
        k_local=args.k_local,
        verbose=verbose,
    )

    try:
        if needs_full:
            states = run_full_multi(args.nii, targets, **_pipeline_kwargs)
            results = [s.to_result() for s in states]
        else:
            states = []
            results = run_multi(args.nii, targets, **_pipeline_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Format and write result(s)
    formatters = {"txt": format_text, "json": format_json, "csv": format_csv}

    if args.output is not None:
        out_path = Path(args.output)
        fmt = _EXT_FORMAT[out_path.suffix.lower()]
        out_path = _unique_path(out_path)
        out_path.write_text(formatters[fmt](results), encoding="utf-8")
        if verbose:
            print(f"  Result saved to: {out_path}")

    if needs_full:
        show = not args.no_plot

        if args.viz_steps is not None:
            try:
                from ..viz.slice2d_view import applied_steps
            except ImportError:
                print(
                    "error: matplotlib is required for --vizSteps.\n"
                    "Install it with:  pip install brain2scalp[viz]",
                    file=sys.stderr,
                )
                return 1
            selected = [states[i] for i in _resolve_indices(args.viz_steps, n)]
            applied_steps(
                selected,
                slice_axis=args.step_axis,
                save_fig=str(_unique_path(save_dir / "steps.png")) if save_dir else None,
                show=show,
            )

        if args.viz_target is not None:
            try:
                from ..viz.slice2d_view import visualize_target
            except ImportError:
                print(
                    "error: matplotlib is required for --vizTarget.\n"
                    "Install it with:  pip install brain2scalp[viz]",
                    file=sys.stderr,
                )
                return 1
            selected = [states[i] for i in _resolve_indices(args.viz_target, n)]
            for j, state in enumerate(selected):
                if save_dir:
                    suffix = f"_{j}" if len(selected) > 1 else ""
                    fname = str(_unique_path(save_dir / f"target{suffix}.png"))
                else:
                    fname = None
                visualize_target(
                    [state],
                    save_fig=fname,
                    show=show,
                )

        if args.viz3d is not None:
            try:
                from ..viz.view_3d import visualize_volume
            except ImportError:
                print(
                    "error: plotly is required for --viz3d.\n"
                    "Install it with:  pip install brain2scalp[viz]",
                    file=sys.stderr,
                )
                return 1
            selected = [states[i] for i in _resolve_indices(args.viz3d, n)]
            visualize_volume(
                selected,
                save_html=str(_unique_path(save_dir / "viz3d.html")) if save_dir else None,
                show=show,
                stride=args.volume_stride,
            )

    return 0


def entry_point() -> None:
    """Setuptools console-script entry point."""
    sys.exit(main())


if __name__ == "__main__":
    entry_point()