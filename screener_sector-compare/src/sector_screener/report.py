from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
from jinja2 import Template

REPORT_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{{ industry }} sector screen</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1220; --card:#131d30; --ink:#edf3ff; --muted:#9fb0ca; --accent:#67e8f9; --warn:#fbbf24; --good:#6ee7b7; --bad:#fb7185; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,sans-serif; }
    main { max-width:1180px; margin:auto; padding:32px 20px 60px; }
    h1,h2 { margin:0 0 12px; } h1 { font-size:30px; } h2 { margin-top:28px; font-size:19px; }
    .meta { color:var(--muted); margin-bottom:24px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }
    .card { background:var(--card); border:1px solid #263653; border-radius:12px; padding:16px; }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    .value { font-size:24px; font-weight:700; margin-top:4px; }
    .alert { border-left:5px solid {{ '#6ee7b7' if alert.triggered else '#fbbf24' }}; }
    table { width:100%; border-collapse:collapse; background:var(--card); border-radius:12px; overflow:hidden; }
    th,td { padding:9px 10px; border-bottom:1px solid #263653; text-align:right; white-space:nowrap; }
    th:first-child,td:first-child { text-align:left; } th { color:var(--muted); font-size:12px; }
    .note { color:var(--muted); font-size:13px; }
    code { color:var(--accent); }
  </style>
</head>
<body><main>
  <h1>{{ industry }} correction & rebound screen</h1>
  <div class="meta">{{ stage|upper }} · data through {{ as_of }} · {{ ticker_count }} usable / {{ selected_ticker_count }} selected tickers · {{ universe_source }} · research screen, not an investment recommendation</div>
  <section class="grid">
    <div class="card"><div class="label">Trend</div><div class="value">{{ trend }}</div></div>
    <div class="card"><div class="label">60d drawdown</div><div class="value">{{ drawdown }}</div></div>
    <div class="card"><div class="label">Median correlation</div><div class="value">{{ correlation }}</div></div>
    <div class="card"><div class="label">PC1 explained</div><div class="value">{{ pc1 }}</div></div>
    <div class="card"><div class="label">Rebound probability</div><div class="value">{{ probability }}</div></div>
  </section>
  <h2>Alert state</h2>
  <div class="card alert"><strong>{{ 'REBOUND TRIGGER' if alert.triggered else ('CORRECTION WATCH' if alert.watch else 'No active signal') }}</strong><br>
    {{ alert.reason }} <span class="note">Close-based signals are actionable no earlier than the next session.</span>
  </div>
  <h2>Relative-strength ranking</h2>
  {{ ranking_table }}
  <h2>Walk-forward validation</h2>
  <div class="card"><pre>{{ validation }}</pre></div>
  <h2>Interpretation guardrails</h2>
  <div class="card note">A high correlation/PC1 reading confirms a common sector move; it is not, by itself, a bottom. The alert requires a correction watch, breadth improvement, and the XGBoost probability threshold. Universe membership is pinned in the run manifest, but current-membership screens may still introduce survivorship bias in historical tests.</div>
</main></body></html>"""
)


def render_report(
    path: Path,
    industry: str,
    stage: str,
    latest: pd.Series,
    ranking: pd.DataFrame,
    validation: dict,
    alert: dict,
    selected_ticker_count: int,
    universe_source: str,
) -> None:
    columns = [
        "fall_resistance_rank",
        "rise_strength_rank",
        "downside_capture",
        "upside_capture",
        "momentum_20d",
        "sector_correlation",
    ]
    table = ranking[columns].round(3).reset_index().to_html(index=False, border=0)
    trend_score = float(latest.get("mid_trend_score", 0))
    trend = "strong up" if trend_score >= 1 else "strong down" if trend_score <= -1 else "mixed"
    document = REPORT_TEMPLATE.render(
        industry=html.escape(industry),
        stage=stage,
        as_of=latest.name.date().isoformat(),
        ticker_count=len(ranking),
        selected_ticker_count=selected_ticker_count,
        universe_source=html.escape(universe_source),
        trend=trend,
        drawdown=f"{latest.get('drawdown_60d', float('nan')):.1%}",
        correlation=f"{latest.get('median_pairwise_correlation', float('nan')):.2f}",
        pc1=f"{latest.get('pc1_explained_variance', float('nan')):.1%}",
        probability=f"{latest.get('rebound_probability', float('nan')):.1%}",
        ranking_table=table,
        validation=html.escape(str(validation)),
        alert=alert,
    )
    path.write_text(document, encoding="utf-8")
