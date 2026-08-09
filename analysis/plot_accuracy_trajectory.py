"""Accuracy over training steps, read straight from a trajectory folder.

Point ``--dir`` at a folder of per-checkpoint recall parquets -- each one a row
per question with a ``correct`` flag -- and it plots accuracy against the
training step, with a Wilson 95% band for the sampling noise at each point.

    python analysis/plot_accuracy_trajectory.py \
        --dir "data/artifacts/expermints/15shot_trajectory_top10_internal_with_hidden/recall" \
        --out analysis/accuracy_trajectory.html

Rows flagged ``excluded`` are dropped by default (pass ``--keep-excluded`` to
score them as wrong instead). The step comes from the ``step`` column, falling
back to ``step<N>`` in the filename.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Categorical slot 1 of the validated reference palette (light mode), the same
# chrome report_token_strain.py uses.
BLUE = "#2a78d6"
BAND = "rgba(42, 120, 214, 0.14)"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
SURFACE = "#fcfcfb"


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion -- stays inside [0,1] when hits is small."""
    if n == 0:
        return (0.0, 0.0)
    p, d = hits / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def step_of(df: pd.DataFrame, path: Path) -> int:
    if "step" in df.columns and df["step"].nunique() == 1:
        return int(df["step"].iloc[0])
    m = re.search(r"step[_-]?(\d+)", path.name)
    if not m:
        raise ValueError(f"no step in {path.name} and no usable 'step' column")
    return int(m.group(1))


def collect(folder: Path, keep_excluded: bool) -> pd.DataFrame:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no .parquet files under {folder}")

    rows = []
    for path in files:
        df = pd.read_parquet(path, columns=None)
        if "correct" not in df.columns:
            raise SystemExit(f"{path.name} has no 'correct' column")
        if "excluded" in df.columns and not keep_excluded:
            df = df.loc[~df["excluded"].astype(bool)]
        hits, n = int(df["correct"].astype(bool).sum()), len(df)
        lo, hi = wilson(hits, n)
        rows.append(dict(step=step_of(df, path), n=n, hits=hits,
                         acc=hits / n if n else float("nan"), lo=lo, hi=hi,
                         file=path.name))

    out = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    dupes = out.loc[out["step"].duplicated(keep=False), "file"].tolist()
    if dupes:
        print(f"warning: repeated steps across {dupes}")
    return out


def figure(traj: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    # Band first so the line and its markers sit on top of it.
    fig.add_trace(go.Scatter(
        x=list(traj["step"]) + list(traj["step"][::-1]),
        y=list(traj["hi"]) + list(traj["lo"][::-1]),
        fill="toself", fillcolor=BAND, line=dict(width=0),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=traj["step"], y=traj["acc"], mode="lines+markers",
        line=dict(color=BLUE, width=2),
        marker=dict(size=9, color=BLUE, line=dict(width=2, color=SURFACE)),
        customdata=traj[["hits", "n", "lo", "hi"]],
        hovertemplate=("step %{x:,}<br>accuracy %{y:.1%}"
                       "<br>%{customdata[0]:,} / %{customdata[1]:,} correct"
                       "<br>95%% CI %{customdata[2]:.1%}–%{customdata[3]:.1%}"
                       "<extra></extra>"),
        showlegend=False))

    # Only the endpoints get a printed number; the axis carries the rest. They
    # hang off the band edge rather than the marker, so neither can collide --
    # the first below its band, the last above it, both pulled in from the edge.
    first, last = traj.iloc[0], traj.iloc[-1]
    for row, y, shift, anchor in ((first, "lo", -16, "left"),
                                  (last, "hi", 16, "right")):
        fig.add_annotation(x=row["step"], y=row[y], yshift=shift,
                           xanchor=anchor, text=f"{row['acc']:.1%}",
                           showarrow=False,
                           font=dict(color=MUTED, size=11))

    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                tickfont=dict(color=MUTED))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif",
                  size=13, color=INK2),
        margin=dict(l=72, r=28, t=58, b=56),
        title=dict(text=title, font=dict(size=15, color=INK), x=0,
                   xanchor="left"),
        xaxis=dict(axis, title="training step", tickformat=","),
        yaxis=dict(axis, title="accuracy", tickformat=".0%", rangemode="tozero"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID,
                        font=dict(color=INK, size=12)))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path,
                    help="folder of per-checkpoint recall parquets")
    ap.add_argument("--out", type=Path, default=None,
                    help="HTML to write (default: <dir>/accuracy_trajectory.html)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--keep-excluded", action="store_true",
                    help="score excluded rows as wrong instead of dropping them")
    ap.add_argument("--show", action="store_true", help="open in a browser")
    args = ap.parse_args()

    traj = collect(args.dir, args.keep_excluded)
    print(traj[["step", "hits", "n", "acc"]].to_string(index=False,
                                                       float_format="%.4f"))

    title = args.title or f"Accuracy across checkpoints — {args.dir.name}"
    fig = figure(traj, title)
    out = args.out or args.dir / "accuracy_trajectory.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="inline",
                   config=dict(displaylogo=False))
    print(f"wrote {out}")
    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
