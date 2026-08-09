"""Plot the numbers in analysis/FINDINGS_token_strain.md. Data only, no argument.

Every figure is a direct rendering of one block of the findings doc: axes, values,
uncertainty, and sample sizes. No captions, no interpretation, no verdict — read
the doc for what the numbers mean.

Everything is recomputed from the panel and the permutation JSON, so pointing
``--panel`` at a different panel regenerates the whole page against that data.

    python analysis/report_token_strain.py \
        --panel data/artifacts/features/strain_panel_15shot_top10.parquet \
        --null-json data/artifacts/features/null_control_15shot_top10_namelen.json \
        --out analysis/report_token_strain.html
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from analysis.strain_models import (  # noqa: E402
    BASE, F1, F2_W, MODELS, NAMELEN, STRAIN, STRAIN_W, LR, fit, prepare)

# Categorical slots 1-3 of the validated reference palette (light mode):
# worst adjacent CVD dE 9.2, normal-vision dE 27.6, all six checks pass.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
SURFACE = "#fcfcfb"

PLOT_CONFIG = {
    "displayModeBar": "hover", "displaylogo": False,
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d",
                               "zoomIn2d", "zoomOut2d", "autoScale2d",
                               "resetScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2, "width": 1100,
                             "height": 620},
}


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #
_AXIS = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
             tickfont=dict(color=MUTED))
_CATAXIS = dict(gridcolor="rgba(0,0,0,0)", linecolor=GRID,
                tickfont=dict(color=INK2))


def axis(title=None, cat=False, **over):
    ax = dict(_CATAXIS if cat else _AXIS)
    if title:
        ax["title"] = title
    ax.update(over)
    return ax


def layout(title, **over):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif",
                  size=13, color=INK2),
        margin=dict(l=76, r=28, t=56, b=54),
        title=dict(text=title, font=dict(size=14.5, color=INK), x=0,
                   xanchor="left"),
        xaxis=axis(), yaxis=axis(),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID,
                        font=dict(color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=INK2), bgcolor="rgba(0,0,0,0)"),
        barcornerradius=4, bargap=0.35,
    )
    base.update(over)
    return base


def marker(color, size=9):
    return dict(size=size, color=color, line=dict(width=2, color=SURFACE))


def bar_labels(fmt, values):
    return dict(text=[fmt.format(v) for v in values], textposition="outside",
                textfont=dict(color=MUTED, size=11))


def headroom(values, pad=1.18):
    return [0, float(max(values)) * pad]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
COEF_TERMS = ["l_subj_strain", "l_obj_strain", "l_s_ntok", "l_o_ntok",
              "l_o_pop", "l_s_pop", "l_evidence_static", "l_evidence_w"]

FAMILIES = {
    "strain (static)": STRAIN,
    "strain (windowed)": STRAIN_W,
    "F1 ownership/rivals": F1,
    "F2 alt-answer ratio": F2_W,
}


def load(panel_path: str, null_path: str | None):
    raw = pd.read_parquet(panel_path)
    p = prepare(raw).dropna(subset=["ratio_static", "ratio_w",
                                    "l_s_own_rare", "l_o_own_rare"])
    d = {"raw": raw, "panel": p}

    fits = {n: fit(f, p) for n, f in MODELS.items()}
    d["fits"] = fits
    d["base"], d["base_len"] = fits["exposure only"], fits["+ NAME LENGTH"]

    # Each family scored against both reference models.
    d["families"] = {
        name: (LR.between(fit(BASE + NAMELEN + terms, p), d["base"]),
               LR.between(fit(BASE + NAMELEN + terms, p), d["base_len"]))
        for name, terms in FAMILIES.items()
    }
    d["namelen_lr"] = LR.between(d["base_len"], d["base"])

    # Coefficients, clustered, from BASE + NAMELEN + STRAIN.
    mc = fit(BASE + NAMELEN + STRAIN, p, cluster=True)
    tb = mc.summary2().tables[1]
    rows = []
    for t in COEF_TERMS:
        if t not in tb.index:
            continue
        c, se = tb.loc[t, "Coef."], tb.loc[t, "Std.Err."]
        sd = p[t].std()
        rows.append(dict(term=t, coef=c, se=se, sd=sd, std=c * sd,
                         std_se=se * sd, p=tb.loc[t, "P>|z|"]))
    d["coefs"] = pd.DataFrame(rows).sort_values("std")

    # Windowed vs static, both directions.
    both = fit(BASE + NAMELEN + STRAIN + STRAIN_W, p)
    d["w_on_s"] = LR.between(both, fits["+ strain STATIC | name length"])
    d["s_on_w"] = LR.between(both, fits["+ strain WINDOWED | name length"])

    # Subject strain per relation, each on its own subsample.
    per_rel = []
    for rel, sub in [("all", p)] + [(r, p[p["prop"] == r])
                                    for r in sorted(p["prop"].unique())]:
        f = BASE + NAMELEN + STRAIN
        if sub["prop"].nunique() == 1:
            f = f.replace(" + C(prop)", "")
        try:
            m = fit(f, sub, cluster=True).summary2().tables[1]
            per_rel.append(dict(relation=rel, rows=len(sub),
                                events=int(sub["forgot"].sum()),
                                coef=m.loc["l_subj_strain", "Coef."],
                                se=m.loc["l_subj_strain", "Std.Err."],
                                p=m.loc["l_subj_strain", "P>|z|"]))
        except Exception:                                   # noqa: BLE001
            continue
    d["per_relation"] = pd.DataFrame(per_rel)

    d["null"] = json.loads(Path(null_path).read_text()) if (
        null_path and Path(null_path).exists()) else None
    return d


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_panel_shape(d):
    w = (d["raw"].groupby(["window", "step_lo", "step_hi"])["forgot"]
         .agg(["size", "sum"]).reset_index())
    x = [f"{lo//1000}K→{hi//1000}K" for lo, hi in zip(w["step_lo"], w["step_hi"])]
    fig = go.Figure()
    fig.add_bar(x=x, y=w["size"], marker_color=BLUE, name="at risk",
                hovertemplate="<b>%{x}</b><br>at risk %{y}<extra></extra>")
    fig.add_bar(x=x, y=w["sum"], marker_color=ORANGE, name="forgot",
                hovertemplate="<b>%{x}</b><br>forgot %{y}<extra></extra>")
    fig.update_layout(**layout("Rows at risk and forgetting events, per window",
                               yaxis=axis("facts"), barmode="group"))
    return "Panel shape", fig


def fig_relation_mix(d):
    g = (d["panel"].groupby("prop")["forgot"].agg(["size", "sum"])
         .sort_values("size"))
    fig = go.Figure()
    fig.add_bar(y=g.index, x=g["size"], orientation="h", marker_color=BLUE,
                name="rows", **bar_labels("{}", g["size"]),
                hovertemplate="<b>%{y}</b><br>rows %{x}<extra></extra>")
    fig.add_bar(y=g.index, x=g["sum"], orientation="h", marker_color=ORANGE,
                name="events", **bar_labels("{}", g["sum"]),
                hovertemplate="<b>%{y}</b><br>events %{x}<extra></extra>")
    fig.update_layout(**layout(
        "Rows and forgetting events per relation",
        xaxis=axis("count", range=headroom(g["size"])), yaxis=axis(cat=True),
        barmode="group", margin=dict(l=110, r=40, t=56, b=54)))
    return "Relation mix", fig


def fig_model_comparison(d):
    names = list(d["families"])
    fig = go.Figure()
    fig.add_bar(y=names, x=[d["families"][n][0].chi2 for n in names],
                orientation="h", marker_color=ORANGE, name="vs BASE",
                customdata=[d["families"][n][0].df for n in names],
                **bar_labels("{:.1f}", [d["families"][n][0].chi2 for n in names]),
                hovertemplate="<b>%{y}</b><br>χ²=%{x:.1f}"
                              "<br>df=%{customdata}<extra>vs BASE</extra>")
    fig.add_bar(y=names, x=[d["families"][n][1].chi2 for n in names],
                orientation="h", marker_color=BLUE, name="vs BASE + name length",
                customdata=[d["families"][n][1].df for n in names],
                **bar_labels("{:.1f}", [d["families"][n][1].chi2 for n in names]),
                hovertemplate="<b>%{y}</b><br>χ²=%{x:.1f}"
                              "<br>df=%{customdata}<extra>vs BASE+NAMELEN</extra>")
    allv = [v.chi2 for pair in d["families"].values() for v in pair]
    fig.update_layout(**layout(
        "Likelihood-ratio χ² added by each feature family",
        xaxis=axis("LR χ²", range=headroom(allv)), yaxis=axis(cat=True),
        barmode="group", margin=dict(l=150, r=50, t=56, b=54)))
    return "Model comparison", fig


def fig_pseudo_r2(d):
    order = ["exposure only", "+ NAME LENGTH", "+ strain STATIC | name length",
             "+ F1 ownership/rivals | len", "+ F2 ratio WINDOWED | len",
             "+ everything | name length"]
    order = [n for n in order if n in d["fits"]]
    vals = [d["fits"][n].prsquared for n in order]
    fig = go.Figure()
    fig.add_bar(y=order, x=vals, orientation="h", marker_color=BLUE,
                showlegend=False, **bar_labels("{:.4f}", vals),
                hovertemplate="<b>%{y}</b><br>pseudo-R² %{x:.4f}<extra></extra>")
    fig.update_layout(**layout(
        "McFadden pseudo-R² by model",
        xaxis=axis("pseudo-R²", range=[min(vals) * 0.985, max(vals) * 1.006]),
        yaxis=axis(cat=True), margin=dict(l=230, r=60, t=56, b=54)))
    return "Model fit", fig


def fig_coefficients(d, standardised=True):
    c = d["coefs"]
    key, se_key = ("std", "std_se") if standardised else ("coef", "se")
    title = ("Coefficients, per 1 SD of the predictor (coef × SD)"
             if standardised else "Coefficients, raw (per log-unit)")
    fig = go.Figure()
    fig.add_scatter(
        x=c[key], y=c["term"], mode="markers", showlegend=False,
        marker=marker(BLUE, 11),
        error_x=dict(type="data", array=1.96 * c[se_key], color=BLUE,
                     thickness=2, width=0),
        customdata=np.stack([c["p"], c["sd"]], -1),
        hovertemplate="<b>%{y}</b><br>%{x:+.3f}"
                      "<br>p=%{customdata[0]:.4f}"
                      "<br>SD(x)=%{customdata[1]:.3f}<extra></extra>")
    fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(**layout(
        title, xaxis=axis("log-odds of forgetting  (95% CI)"),
        yaxis=axis(cat=True), margin=dict(l=150, r=40, t=56, b=54)))
    return ("Effect sizes (standardised)" if standardised
            else "Effect sizes (raw)"), fig


def fig_quartiles(d):
    p = d["panel"].copy()
    fig = go.Figure()
    for lbl, col, key in (("subject strain", BLUE, "subj_strain"),
                          ("object strain", ORANGE, "obj_strain")):
        p["q"] = pd.qcut(p[key].rank(method="first"), 4,
                         labels=["Q1", "Q2", "Q3", "Q4"])
        g = p.groupby("q", observed=True)["forgot"].agg(["mean", "size"])
        fig.add_scatter(x=list(g.index), y=g["mean"], mode="lines+markers",
                        line=dict(color=col, width=2), marker=marker(col),
                        name=lbl, customdata=g["size"],
                        hovertemplate="<b>%{x}</b><br>rate %{y:.3f}"
                                      "<br>n=%{customdata}<extra>" + lbl + "</extra>")
    fig.update_layout(**layout(
        "Raw forgetting rate by strain quartile (no model, no controls)",
        xaxis=axis("quartile (low → high)"), yaxis=axis("forgetting rate")))
    return "Raw dose-response", fig


def fig_permutation(d):
    n = d["null"]
    if not n:
        return None
    obs, null = n["observed"], np.asarray(n["null"])
    fig = go.Figure()
    fig.add_histogram(x=null, nbinsx=24, marker_color=ORANGE, opacity=0.85,
                      name=f"{len(null)} permuted indices",
                      hovertemplate="χ² %{x}<br>%{y} permutations<extra></extra>")
    fig.add_vline(x=obs, line=dict(color=BLUE, width=2))
    fig.add_annotation(x=obs, y=1.0, yref="y domain", xanchor="left", xshift=6,
                       text=f"<b>real χ²={obs:.1f}</b>", showarrow=False,
                       font=dict(color=BLUE, size=12))
    fig.update_layout(**layout(
        f"Permutation null: {int((null >= obs).sum())} of {len(null)} ≥ real, "
        f"p={n['p_empirical']:.3f}",
        xaxis=axis("LR χ² over BASE + name length"), yaxis=axis("permutations"),
        margin=dict(l=76, r=28, t=70, b=54)))
    return "Permutation test", fig


def fig_per_relation(d):
    r = d["per_relation"].iloc[::-1]
    colors = [ORANGE if t == "all" else BLUE for t in r["relation"]]
    fig = go.Figure()
    fig.add_scatter(
        x=r["coef"], y=r["relation"], mode="markers", showlegend=False,
        marker=dict(size=11, color=colors, line=dict(width=2, color=SURFACE)),
        error_x=dict(type="data", array=1.96 * r["se"], color=MUTED,
                     thickness=2, width=0),
        customdata=np.stack([r["p"], r["rows"], r["events"]], -1),
        hovertemplate="<b>%{y}</b><br>coef %{x:+.3f}"
                      "<br>p=%{customdata[0]:.3f}"
                      "<br>%{customdata[1]} rows, %{customdata[2]} events"
                      "<extra></extra>")
    fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(**layout(
        "l_subj_strain fitted separately within each relation (95% CI)",
        xaxis=axis("log-odds of forgetting"), yaxis=axis(cat=True),
        margin=dict(l=110, r=40, t=56, b=54)))
    return "Per relation", fig


def fig_windowed(d):
    p = d["panel"]
    s = p.sample(min(2500, len(p)), random_state=0)
    lim = float(max(p["subj_strain"].max(), p["subj_strain_w"].max()))
    r = float(np.corrcoef(p["subj_strain"], p["subj_strain_w"])[0, 1])
    fig = go.Figure()
    fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", name="y = x",
                    line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip")
    fig.add_scatter(x=s["subj_strain"], y=s["subj_strain_w"], mode="markers",
                    marker=dict(size=5, color=BLUE, opacity=0.35,
                                line=dict(width=0)),
                    name="fact × window",
                    hovertemplate="static %{x:.0f} → windowed %{y:.0f}<extra></extra>")
    fig.update_layout(**layout(
        f"Subject strain: windowed vs static (Pearson r = {r:.3f})",
        xaxis=axis("static"), yaxis=axis("windowed")))
    return "Windowed vs static", fig


FIGURES = [fig_panel_shape, fig_relation_mix, fig_model_comparison, fig_pseudo_r2,
           lambda d: fig_coefficients(d, True), lambda d: fig_coefficients(d, False),
           fig_quartiles, fig_permutation, fig_per_relation, fig_windowed]


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
CSS = f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:{SURFACE}; color:{INK};
  font:15px/1.6 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
.wrap {{ max-width:880px; margin:0 auto; padding:40px 24px 80px; }}
h1 {{ font-size:22px; margin:0 0 4px; letter-spacing:-.02em; }}
.sub {{ color:{MUTED}; margin:0 0 6px; font-size:13.5px; }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px; margin:16px 0 8px; }}
.pill {{ background:#fff; border:1px solid {GRID}; border-radius:999px;
  padding:4px 12px; font-size:12.5px; color:{INK2}; }}
.pill b {{ color:{INK}; }}
section {{ margin:0 0 34px; padding-top:22px; border-top:1px solid {GRID}; }}
h2 {{ font-size:13px; margin:0 0 10px; color:{MUTED}; font-weight:600;
  text-transform:uppercase; letter-spacing:.05em; }}
.chart {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; margin-top:10px; font-size:13px; }}
th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid {GRID}; }}
th {{ color:{MUTED}; font-weight:600; font-size:11.5px; text-transform:uppercase;
  letter-spacing:.04em; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
code {{ background:#f2f1ed; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
"""


def coef_table(d) -> str:
    c = d["coefs"].sort_values("std", key=abs, ascending=False)
    body = "".join(
        f"<tr><td><code>{r.term}</code></td><td class='num'>{r.coef:+.4f}</td>"
        f"<td class='num'>{r.se:.4f}</td><td class='num'>{r.sd:.3f}</td>"
        f"<td class='num'>{r.std:+.3f}</td><td class='num'>{np.exp(r.std):.2f}</td>"
        f"<td class='num'>{r.p:.4f}</td></tr>" for r in c.itertuples())
    return ("<table><thead><tr><th>term</th><th class='num'>coef</th>"
            "<th class='num'>std.err</th><th class='num'>SD(x)</th>"
            "<th class='num'>coef×SD</th><th class='num'>OR/SD</th>"
            f"<th class='num'>p</th></tr></thead><tbody>{body}</tbody></table>")


def lr_table(d) -> str:
    rows = [("BASE + NAMELEN", d["namelen_lr"], None)]
    rows += [(n, a, b) for n, (a, b) in d["families"].items()]
    body = ""
    for name, a, b in rows:
        vs_len = (f"<td class='num'>χ²({b.df})={b.chi2:.1f}</td>"
                  f"<td class='num'>{b.p:.2e}</td>") if b else \
                 "<td class='num'>—</td><td class='num'>—</td>"
        body += (f"<tr><td>{name}</td><td class='num'>χ²({a.df})={a.chi2:.1f}</td>"
                 f"<td class='num'>{a.p:.2e}</td>{vs_len}</tr>")
    return ("<table><thead><tr><th>added terms</th>"
            "<th class='num'>vs BASE</th><th class='num'>p</th>"
            "<th class='num'>vs BASE+NAMELEN</th><th class='num'>p</th>"
            f"</tr></thead><tbody>{body}</tbody></table>")


def render(d, panel_path: str) -> str:
    p, sections = d["panel"], []
    for i, make in enumerate(FIGURES):
        out = make(d)
        if out is None:
            continue
        title, fig = out
        chart = pio.to_html(fig, include_plotlyjs="inline" if not sections else False,
                            full_html=False, config=PLOT_CONFIG)
        extra = ""
        if title.startswith("Effect sizes (standardised)"):
            extra = coef_table(d)
        elif title == "Model comparison":
            extra = lr_table(d)
        elif title == "Windowed vs static":
            extra = ("<table><tbody>"
                     f"<tr><td>windowed on top of static</td>"
                     f"<td class='num'>χ²({d['w_on_s'].df})={d['w_on_s'].chi2:.1f}</td>"
                     f"<td class='num'>p={d['w_on_s'].p:.3f}</td></tr>"
                     f"<tr><td>static on top of windowed</td>"
                     f"<td class='num'>χ²({d['s_on_w'].df})={d['s_on_w'].chi2:.1f}</td>"
                     f"<td class='num'>p={d['s_on_w'].p:.3f}</td></tr>"
                     "</tbody></table>")
        sections.append(f"<section><h2>{title}</h2>"
                        f"<div class='chart'>{chart}</div>{extra}</section>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token strain — data</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Token strain — data</h1>
<p class="sub">Figures for <code>analysis/FINDINGS_token_strain.md</code>.
Panel: <code>{Path(panel_path).name}</code>.</p>
<div class="meta">
  <span class="pill"><b>{len(p):,}</b> rows</span>
  <span class="pill"><b>{int(p['forgot'].sum()):,}</b> forgetting events</span>
  <span class="pill"><b>{p['id'].nunique():,}</b> facts</span>
  <span class="pill"><b>{p['window'].nunique()}</b> windows</span>
  <span class="pill"><b>{p['prop'].nunique()}</b> relations</span>
</div>
{''.join(sections)}
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel",
                    default="data/artifacts/features/strain_panel_15shot_top10.parquet")
    ap.add_argument("--null-json",
                    default="data/artifacts/features/null_control_15shot_top10_namelen.json")
    ap.add_argument("--out", default="analysis/report_token_strain.html")
    a = ap.parse_args()

    d = load(a.panel, a.null_json)
    Path(a.out).write_text(render(d, a.panel), encoding="utf-8")
    print(f"wrote {a.out}  ({len(d['panel']):,} rows, "
          f"{int(d['panel']['forgot'].sum()):,} events)")


if __name__ == "__main__":
    main()
