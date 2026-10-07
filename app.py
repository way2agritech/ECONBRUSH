"""
Brush Cutter Selection Advisor - Streamlit prototype.

Run:  streamlit run app.py
Data: brushcutter_data.xlsx next to this file (override with BRUSHCUTTER_DATA=/path/to/file.xlsx,
      or upload a workbook from the sidebar).

The page only draws what engine.py and decision.py return. Nothing about brush cutters is scored here.
"""
from __future__ import annotations

import hashlib
import html
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from decision import Business, Candidate, Decision, decide, load_business
from engine import (DataError, Dataset, ROLE_FILTER, ROLE_KEY, ROLE_LOOKUP, ROLE_RANGE,
                    load_dataset, norm)

DATA_FILE = Path(os.environ.get("BRUSHCUTTER_DATA", Path(__file__).with_name("brushcutter_data.xlsx")))

# --------------------------------------------------------------------------- #
# Display copy. Optional: anything not listed here falls back to the workbook text.
# --------------------------------------------------------------------------- #
INPUT_COPY = {
    "land_extent": ("Land holding", "Total area that will be cleared."),
    "crop": ("Crop", "Sets how many times a year the land is cleared."),
    "soil_feature": ("Ground condition", "The surface the machine has to work on."),
    "weed_height": ("Weed type", "What is growing in the area to be cleared."),
    "power_source": ("Power source", "Pick one to list only machines with that power source."),
}
VALUE_COPY = {
    "soil_feature": {"harsh": "Harsh - rough or stony ground", "even and tilled": "Even and tilled",
                     "slopy": "Sloping terrain"},
    "weed_height": {"soft weeds": "Soft weeds - grass, thin growth", "hard weeds": "Hard weeds - woody, thick stems"},
    "power_source": {"All": "No preference"},
}
LAYER_LABEL = {"technical": "Field fit", "economic": "Running cost", "commercial": "Dealer support"}
FACTOR_LABEL = {"stock_status": "Stock status", "service_support": "Brand service", "warranty": "Warranty"}
FARMER_SCORE_LABEL = {"technical": "Field fit", "economic": "Running cost", "commercial": "Dealer support"}


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def esc(x) -> str:
    return html.escape(str(x), quote=True)


def sentence(v: str) -> str:
    return v[:1].upper() + v[1:]


def nice(v: str) -> str:
    return sentence(v.replace("_and_", " & ").replace("_", " ").strip())


def show(key: str, value: str) -> str:
    return VALUE_COPY.get(key, {}).get(value) or sentence(value)


def inr(x: float) -> str:
    n = int(round(x))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head) + "," + tail
    return ("-" if n < 0 else "") + "\u20b9" + s


def fit(score: float) -> tuple[str, str]:
    if score >= 80:
        return "Best choice", "fit-strong"
    if score >= 60:
        return "Good choice", "fit-good"
    return "Check before buying", "fit-part"


def score_tone(score: float) -> str:
    if score >= 80:
        return "score-strong"
    if score >= 60:
        return "score-good"
    return "score-fair"


def meter(value: float, accent: bool = False) -> str:
    v = max(0.0, min(100.0, value))
    return f'<div class="meter{" accent" if accent else ""}"><span style="width:{v:.1f}%"></span></div>'


def grid(items: list[tuple[str, str]], cols: int | None = None) -> str:
    cols = cols or len(items)
    cells = "".join(f'<div class="cell"><div class="k">{esc(k)}</div><div class="v">{v}</div></div>' for k, v in items)
    return f'<div class="grid" style="grid-template-columns:repeat({cols},1fr)">{cells}</div>'


def table(headers: list[str], rows: list[list[str]], num: tuple[int, ...] = ()) -> str:
    head = "".join(f'<th class="{"n" if i in num else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"n" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows)
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def eyebrow(text: str) -> str:
    return f'<div class="eyebrow">{esc(text)}</div>'


def tag(text: str, warn: bool = False) -> str:
    return f'<span class="tag{" warn" if warn else ""}">{esc(text)}</span>'


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #
def onboarding_html(ds: Dataset, biz: Business) -> str:
    steps = [
        ("Farm size", "Land holding and crop decide how often the machine will be used in a year."),
        ("Field condition", "Ground and weed type decide how hard the machine must work."),
        ("Recommended products", f"The best {int(biz.settings.get('top_n', 5))} products are shown for the selected farm conditions."),
    ]
    items = "".join(f'<li><span class="step-n">{i}</span><div><b>{esc(t)}</b><p>{esc(d)}</p></div></li>'
                    for i, (t, d) in enumerate(steps, 1))
    return f'<div class="panel">{eyebrow("Procedure")}<ol class="steps">{items}</ol></div>'


def card_html(c: Candidate, d: Decision, position: int) -> str:
    r = c.result
    e = c.econ
    label, fcls = fit(c.overall)
    tags = []
    if c.source:
        tags.append(tag(nice(c.source)))
    if c.stock:
        tags.append(tag(nice(c.stock), warn=norm(c.stock) != "in_stock"))
    if c.warranty:
        tags.append(tag(f"{c.warranty} mo warranty"))
    if c.service:
        tags.append(tag(nice(c.service)))
    meta = "".join(tags)
    head = (
        '<div class="rec-card">'
        '<div class="card-head">'
        f'<div class="rank"><span>#{position}</span></div>'
        '</div>'
        '<div class="product-row">'
        f'<div class="ident"><div class="product-name">{esc(r.name)}</div><div class="sub">{esc(c.brand)}</div></div>'
        '<div class="product-aside">'
        f'<div class="price"><span>Price</span><b>{inr(e.price) if e else "Not listed"}</b></div>'
        f'<div class="fit {fcls}">{label}</div>'
        '</div>'
        '</div>'
        f'<div class="tag-row">{meta}</div>'
    )
    score_items = [("overall", "Overall score", c.overall)]
    score_items += [
        (k, FARMER_SCORE_LABEL.get(k, LAYER_LABEL.get(k, nice(k))), v)
        for k, v in (("technical", c.technical), ("economic", c.economic), ("commercial", c.commercial))
        if k in d.layer_weights
    ]
    layers = "".join(
        f'<div class="layer {k}"><div class="lab"><span>{esc(label_text)}</span>'
        f'<b>{v:.0f}</b></div>{meter(v)}</div>'
        for k, label_text, v in score_items)
    return (
        f'{head}<div class="layers">{layers}</div>'
        '</div>'
    )


def detail_html(c: Candidate, d: Decision) -> str:
    r, e = c.result, c.econ
    rows = []
    for ch in r.checks:
        if not ch.applicable:
            res = '<span class="r-na">Not applicable</span>'
        elif ch.is_match:
            res = '<span class="r-ok">Match</span>'
        elif ch.partial:
            res = '<span class="r-warn">Partly inside range</span>'
        else:
            res = '<span class="r-bad">Not matched</span>'
        rows.append([esc(ch.label), esc(", ".join(ch.suitable)), esc(", ".join(ch.offered) or "-"), res,
                     f"{ch.weight * 100:.1f}%", f"{ch.earned:.1f}"])
    rows.append(["<b>Field-fit score</b>", "", "", "", "<b>100%</b>", f"<b>{c.technical:.1f}</b>"])
    out = eyebrow("Field-fit check") + table(
        ["Need", "Best for this farm", "This product", "Result", "Share", "Score"], rows, num=(4, 5))
    if e:
        cost_rows = [
            ["Purchase price spread over life", f"{inr(e.price)} / {e.life:g} yr", inr(e.capital)],
            ["Maintenance", "fixed per year", inr(e.maintenance)],
            ["Energy", f"{e.hours:.0f} h x {e.energy_use / e.hours:.2f} {e.energy_unit}/h x {inr(e.energy_price)}/{e.energy_unit}"
             if e.energy_unit else "no energy price found", inr(e.energy_cost)],
            ["<b>Annual cost</b>", "", f"<b>{inr(e.annual)}</b>"],
        ]
        out += eyebrow("Cost build-up") + table(["Item", "Basis", "Per year"], cost_rows, num=(2,))
    com_rows = [[esc(FACTOR_LABEL.get(p.factor, nice(p.factor))), esc(nice(p.shown) if p.shown else "not listed"),
                 f"{p.points:.0f}", f"{p.weight * 100:.0f}%"] for p in c.parts]
    com_rows.append(["<b>Dealer-support score</b>", "", f"<b>{c.commercial:.1f}</b>", "<b>100%</b>"])
    out += eyebrow("Dealer support") + table(["Factor", "Value", "Points", "Share"], com_rows, num=(2, 3))
    return out


def compare_html(d: Decision) -> str:
    cs = d.shortlist
    heads = "".join(f'<th class="n">{i:02d}<br><span class="th-name">{esc(c.result.name)}</span></th>'
                    for i, c in enumerate(cs, 1))

    def row(label, values, best=None, fmt=str):
        raw = [v for v in values]
        target = None
        if best and any(v is not None for v in raw):
            nums = [v for v in raw if v is not None]
            target = max(nums) if best == "max" else min(nums)
        cells = "".join(
            f'<td class="n{" best" if (target is not None and v == target) else ""}">{fmt(v) if v is not None else "-"}</td>'
            for v in raw)
        return f"<tr><th>{esc(label)}</th>{cells}</tr>"

    body = "".join([
        row("Overall score", [round(c.overall, 1) for c in cs], "max", lambda v: f"{v:.1f}"),
        row("Field fit", [round(c.technical, 1) for c in cs], "max", lambda v: f"{v:.1f}"),
        row("Running cost", [round(c.economic, 1) for c in cs], "max", lambda v: f"{v:.1f}"),
        row("Dealer support", [round(c.commercial, 1) for c in cs], "max", lambda v: f"{v:.1f}"),
        row("Purchase price", [c.econ.price if c.econ else None for c in cs], "min", inr),
        row("Annual cost", [c.econ.annual if c.econ else None for c in cs], "min", inr),
        row("Cost per acre worked", [c.econ.per_pass if c.econ else None for c in cs], "min", inr),
        row("Working hours / yr", [c.econ.hours if c.econ else None for c in cs], "min", lambda v: f"{v:.0f} h"),
    ])
    for ch in cs[0].result.checks:
        vals = []
        for c in cs:
            m = next(x for x in c.result.checks if x.feature == ch.feature)
            cls = "ok" if m.is_match else ("miss" if m.applicable else "na")
            vals.append(f'<span class="dot {cls}"></span>{esc(", ".join(m.offered)) if m.offered else "n/a"}')
        body += f'<tr><th>{esc(ch.label)}</th>' + "".join(f'<td class="n spec-cell">{v}</td>' for v in vals) + "</tr>"
    body += "<tr><th>Stock</th>" + "".join(f'<td class="n">{esc(nice(c.stock)) or "-"}</td>' for c in cs) + "</tr>"
    body += "<tr><th>Warranty</th>" + "".join(f'<td class="n">{esc(c.warranty)} mo</td>' for c in cs) + "</tr>"
    body += "<tr><th>Brand service</th>" + "".join(f'<td class="n">{esc(nice(c.service))}</td>' for c in cs) + "</tr>"
    return (f'<div class="tbl-wrap"><table class="tbl compare"><thead><tr><th></th>{heads}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>'
            '<div class="legend">Bold cells are the best value in the row. Dots: '
            '<span class="sw ok"></span>meets requirement<span class="sw miss"></span>does not meet<span class="sw na"></span>not applicable.</div>')


def weights_html(d: Decision) -> str:
    rows = []
    for w in d.rec.weights:
        rows.append([esc(w.label), f"{w.rank:g}", f"{w.points:g}",
                     f'<div class="wcell">{meter(w.weight * 100, True)}<span class="mono">{w.weight * 100:.1f}%</span></div>',
                     esc(", ".join(w.suitable))])
    feat = table(["Need", "Importance rank", "Points", "Share of field-fit score", "Suited to this farm"], rows, num=(1, 2))
    lay = grid([(LAYER_LABEL.get(k, nice(k)), f'<span class="mono">{v * 100:.0f}%</span>') for k, v in d.layer_weights.items()])
    notes = "".join(f"<li>{esc(t)}</li>" for t in [
        "Experts rank each specification from 1 (most important) for every combination of ground, weeds and land size.",
        "Ranks are turned into points with the rank_points table, then into shares that add up to 100%.",
        "A product earns a feature's share when it offers a suitable value. A blank cell means not applicable, "
        "and the remaining shares are rescaled so that product is not penalised.",
        "Products with the same field-fit score are ordered by how many suitable options they offer.",
    ])
    return (f'{eyebrow("Rule row applied")}<p class="body">{esc(d.rec.rule_row)}</p>'
            f'{eyebrow("Importance of each specification for this farm")}{feat}'
            f'{eyebrow("Weight of each layer in the overall score")}{lay}'
            f'{eyebrow("How the field-fit score works")}<ul class="plain">{notes}</ul>')


def data_html(d: Decision, ds: Dataset) -> str:
    out = eyebrow("Assumptions") + "<ul class='plain'>" + "".join(f"<li>{esc(a)}</li>" for a in d.assumptions) + "</ul>"
    notes = ds.warnings + [f"{f}: {why}" for f, why in d.rec.unscored]
    out += eyebrow("Data checks on the workbook")
    out += ("<ul class='plain'>" + "".join(f"<li>{esc(w)}</li>" for w in notes) + "</ul>") if notes \
        else '<p class="body">No problems found.</p>'
    if d.rec.excluded:
        out += eyebrow("Left out by the power source choice") + table(
            ["Product", "Brand", "Reason"],
            [[esc(x.name), esc(next(iter(x.meta.values()), "")), esc("; ".join(x.reasons))] for x in d.rec.excluded])
    if d.capped:
        out += eyebrow(f"Held back by the limit of {d.max_per_brand} per brand") + table(
            ["Overall rank", "Product", "Brand", "Overall"],
            [[str(c.order), esc(c.result.name), esc(c.brand), f"{c.overall:.1f}"] for c in d.capped], num=(0, 3))
    if d.rest:
        out += eyebrow("Ranked below the shortlist") + table(
            ["Overall rank", "Product", "Brand", "Overall", "Field fit", "Running cost", "Support"],
            [[str(c.order), esc(c.result.name), esc(c.brand), f"{c.overall:.1f}", f"{c.technical:.1f}",
              f"{c.economic:.1f}", f"{c.commercial:.1f}"] for c in d.rest], num=(0, 3, 4, 5, 6))
    return out


def shortlist_csv(d: Decision) -> bytes:
    rows = []
    for i, c in enumerate(d.shortlist, 1):
        e = c.econ
        rows.append({"rank": i, "product": c.result.name, "brand": c.brand, "overall": round(c.overall, 1),
                     "field_fit": round(c.technical, 1), "running_cost": round(c.economic, 1),
                     "dealer_support": round(c.commercial, 1), "price": e.price if e else None,
                     "annual_cost": round(e.annual) if e else None, "hours_per_year": round(e.hours) if e else None})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
html{overflow-x:hidden;scroll-behavior:smooth}
:root{
--white-1:#FBFCFA;--white-2:#fafafa;--white-3:#f4f4f4;
--green-0:#f1fdf1;--green-1:#4CAF50;--green-2:#063024;--green-3:#165247;
--surface-1:#f0fdf4;--content-1:#1e8a23;--stroke-1:#4CAF50;
--surface-2:#eff6ff;--content-2:#1447e6;--stroke-2:#bedbff;
--surface-3:#fffbeb;--content-3:#bb4d00;--stroke-3:#fee685;
--surface-4:#ffebee;--content-4:#fb2c36;--stroke-4:#ffc9c9;
--header-height:80px;--muted:#5f6f68;--line:#dce8dc;--line-strong:#b7d8b8;
--sans:Inter,'Segoe UI',system-ui,sans-serif;--mono:'Segoe UI Mono',Consolas,monospace}
.stApp{background:linear-gradient(180deg,var(--white-1),var(--green-0));color:var(--green-2)}
.stApp,.stApp p,.stApp label,.stApp li,.stApp input,.stApp button,.stApp [data-baseweb="select"]{font-family:var(--sans)}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.block-container{padding-top:1rem;padding-bottom:3rem;max-width:1180px}
section[data-testid="stSidebar"]{background:var(--white-1);border-right:1px solid var(--line)}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem}
[data-testid="stForm"]{border:1px solid var(--line-strong);border-radius:8px;background:#fff;padding:14px 14px 10px;box-shadow:0 12px 35px rgba(6,48,36,.06)}
div[data-baseweb="select"]>div,div[data-baseweb="base-input"],[data-testid="stNumberInputContainer"]{border-radius:8px!important}
div[data-baseweb="select"]>div,[data-testid="stNumberInputContainer"]{background:var(--white-1);border-color:var(--line-strong)!important}
div[data-baseweb="base-input"] input,[data-testid="stNumberInputContainer"] input{background:#fff!important;color:var(--green-2)!important;-webkit-text-fill-color:var(--green-2)!important;font-weight:700}
[data-testid="stNumberInputContainer"] button{color:var(--green-2)!important}
[data-testid="stWidgetLabel"] p{font-size:12px;font-weight:700;color:var(--green-2)}
.stFormSubmitButton button,.stDownloadButton button{border-radius:8px;font-weight:700;padding:.62rem 1rem}
.stFormSubmitButton button{background:var(--green-1);color:#fff;border:1px solid var(--green-1)}
.stFormSubmitButton button:hover{background:var(--green-3);border-color:var(--green-3);color:#fff}
.stDownloadButton button{background:#fff;color:var(--green-3);border:1px solid var(--stroke-1)}
.stDownloadButton button:hover{background:var(--surface-1);border-color:var(--green-3);color:var(--green-2)}
[data-testid="stVerticalBlockBorderWrapper"]:has(.rec-card){background:transparent;border:0;border-radius:0;padding:0;box-shadow:none;overflow:visible;margin-bottom:0}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:8px;background:var(--white-2)}
[data-testid="stExpander"] summary p{font-size:12px;font-weight:700;color:var(--green-3)}
.eyebrow{font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--green-3);margin:16px 0 7px}
.panel{background:#fff;border:1px solid var(--line-strong);border-radius:8px;padding:6px 18px 16px;margin-bottom:14px;box-shadow:0 12px 30px rgba(6,48,36,.05)}
.panel.tight{padding-bottom:10px}
.body{font-size:14px;line-height:1.55;color:#31453e;margin:0 0 10px}
.foot{font-size:12px;color:var(--muted);margin-top:10px}
.steps{list-style:none;margin:0;padding:0}
.steps li{display:flex;gap:14px;padding:10px 0;border-bottom:1px solid var(--line)}
.steps li:last-child{border-bottom:0}
.steps p{margin:2px 0 0;color:var(--muted);font-size:13.5px}
.step-n{flex:none;width:26px;height:26px;border-radius:50%;background:var(--green-1);color:#fff;font:800 12px/26px var(--sans);text-align:center}
.grid{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.cell{background:#fff;padding:9px 11px}
.cell .k,.spec .k{font-size:11px;font-weight:800;color:var(--muted)}
.cell .v{font-size:14px;margin-top:3px}
.mono{font-family:var(--mono)}.strong{font-weight:800}
.section-title{font-size:18px;font-weight:850;color:var(--green-2);margin:0 0 10px}
.product-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
.rec-card{padding:11px;background:#fff;border:1px solid rgba(15,77,60,.14);border-radius:8px;box-shadow:0 10px 24px rgba(6,48,36,.07);min-width:0}
.card-head{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:8px}
.rank span{display:inline-flex;align-items:center;height:22px;padding:0 8px;border-radius:999px;
background:#f3f7f4;border:1px solid var(--line);font:800 10px var(--sans);color:#47655b;text-transform:uppercase;letter-spacing:.03em}
.product-row{display:block;padding-bottom:8px;border-bottom:1px solid var(--line)}
.product-name{font-size:16px;font-weight:850;word-break:break-word;color:#102f26;line-height:1.1}
.sub{color:#5b6d66;font-size:11px;margin-top:2px;font-weight:750}
.product-aside{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:8px}
.price{text-align:left;min-width:0}
.price span{display:block;font-size:9px;font-weight:850;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.price b{display:block;font:850 13px var(--mono);color:#102f26;margin-top:1px;white-space:nowrap}
.tag-row{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;max-height:42px;overflow:hidden}
.tag{display:inline-flex;align-items:center;border:1px solid var(--stroke-2);background:var(--surface-2);color:var(--content-2);border-radius:999px;padding:2px 6px;font-size:9.5px;font-weight:750;line-height:1.15}
.tag.warn{border-color:var(--stroke-3);color:var(--content-3);background:var(--surface-3)}
.fit{flex:none;font-size:9.5px;font-weight:850;text-transform:uppercase;letter-spacing:.04em;padding:5px 7px;border-radius:999px;background:#f5f8f6;white-space:nowrap}
.fit-strong{color:#137a45}.fit-good{color:#1f64c8}.fit-part{color:#b75c00}
.layers{display:flex;flex-direction:column;gap:6px;padding:9px 0 0}
.layer{border:1px solid var(--stroke-2);border-radius:7px;padding:7px;background:var(--surface-2)}
.layer.overall{background:#f3f8f4;border-color:#b7d8b8;padding:8px}.layer.technical{background:var(--surface-1);border-color:var(--stroke-1)}.layer.economic{background:var(--surface-3);border-color:var(--stroke-3)}.layer.commercial{background:var(--surface-2);border-color:var(--stroke-2)}
.lab{display:flex;justify-content:space-between;gap:8px;font-size:10.5px;font-weight:800;color:var(--green-2)}
.lab b{font:800 12px var(--mono);color:var(--green-2)}
.meter{height:5px;background:rgba(6,48,36,.12);border-radius:999px;margin-top:5px;overflow:hidden}
.layer.overall .meter{height:7px}
.meter span{display:block;height:100%;background:var(--content-2);border-radius:999px}
.layer.overall .meter span,.layer.technical .meter span,.meter.accent span{background:var(--content-1)}.layer.economic .meter span{background:var(--content-3)}.layer.commercial .meter span{background:var(--content-2)}
.specs{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.spec{background:#fff;padding:8px 10px;border-left:4px solid var(--line)}
.spec.ok{border-left-color:var(--content-1);background:var(--surface-1)}.spec.miss{border-left-color:var(--content-3);background:var(--surface-3)}
.spec .v{font-size:13px;margin-top:2px;color:var(--green-2)}
.legend{font-size:12px;color:var(--muted);margin:7px 0 2px}
.sw,.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 5px 0 12px;background:var(--line);vertical-align:baseline}
.legend .sw:first-child{margin-left:0}
.sw.ok,.dot.ok{background:var(--content-1)}.sw.miss,.dot.miss{background:var(--content-3)}.dot{margin:0 6px 0 0}
.note{font-size:13px;margin:12px 0 0;padding:9px 11px;border:1px solid var(--line);border-left:4px solid var(--line-strong);border-radius:8px;background:var(--white-2)}
.note.ok{border-left-color:var(--content-1);background:var(--surface-1)}.note.warn{border-left-color:var(--content-3);background:var(--surface-3)}
.tbl-wrap{overflow-x:auto;margin-bottom:6px;border-radius:8px}
.tbl{width:100%;border-collapse:collapse;font-size:13px;background:#fff}
.tbl th{font-size:11px;font-weight:800;color:var(--muted);text-align:left;border-bottom:1px solid var(--line-strong);padding:8px 9px;vertical-align:bottom}
.tbl td{padding:8px 9px;border-bottom:1px solid var(--line);vertical-align:top}
.tbl .n{text-align:right;font-family:var(--mono)}
.tbl.compare th:first-child{min-width:150px}
.tbl.compare tbody th{font-size:12px;font-weight:800;text-align:left;color:var(--green-2);border-bottom:1px solid var(--line);background:var(--white-2)}
.tbl.compare td.best{font-weight:800;background:var(--surface-1)}
.tbl .spec-cell{font-family:var(--sans);font-size:12.5px}
.th-name{font-weight:800;font-size:12px;color:var(--green-2);text-transform:none;letter-spacing:0}
.r-ok{color:var(--content-1);font-weight:800}.r-bad{color:var(--content-4);font-weight:800}.r-warn{color:var(--content-3);font-weight:800}.r-na{color:var(--muted)}
.wcell{display:flex;align-items:center;gap:10px;justify-content:flex-end}.wcell .meter{width:110px;margin:0}
.plain{margin:0 0 8px;padding-left:18px;font-size:13.5px;line-height:1.6;color:#31453e}
.scrollbar-thin,.scrollbar-thin-x,.scrollbar-thin-y{scrollbar-width:thin;scrollbar-color:#cbd5e1 transparent}
.scrollbar-thin::-webkit-scrollbar-track,.scrollbar-thin-x::-webkit-scrollbar-track,.scrollbar-thin-y::-webkit-scrollbar-track{background:transparent}
.scrollbar-thin::-webkit-scrollbar-thumb,.scrollbar-thin-x::-webkit-scrollbar-thumb,.scrollbar-thin-y::-webkit-scrollbar-thumb{background-color:#cbd5e1;border-radius:9999px}
.scrollbar-thin::-webkit-scrollbar-thumb:hover,.scrollbar-thin-x::-webkit-scrollbar-thumb:hover,.scrollbar-thin-y::-webkit-scrollbar-thumb:hover{background-color:#94a3b8}
.scrollbar-thin::-webkit-scrollbar,.scrollbar-thin-x::-webkit-scrollbar{height:5px;width:0}
.scrollbar-thin::-webkit-scrollbar,.scrollbar-thin-y::-webkit-scrollbar{width:5px;height:0}
.scrollbar-thin::-webkit-scrollbar{width:5px;height:5px}
@media(max-width:1100px){.product-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:900px){:root{--header-height:50px}.specs{grid-template-columns:repeat(2,1fr)}.product-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
.card-head,.product-row{align-items:flex-start}.price{text-align:left}
.grid{grid-template-columns:repeat(2,1fr)!important}}
@media(max-width:560px){.product-grid{grid-template-columns:1fr}}
"""


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_workbook_data(blob: bytes, name: str):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / name
        path.write_bytes(blob)
        ds = load_dataset(path)
        biz = load_business(path)
    return ds, biz


def current_source() -> tuple[bytes, str]:
    uploaded = st.session_state.get("data_upload")
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name
    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}.")
        st.stop()
    return DATA_FILE.read_bytes(), DATA_FILE.name


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def input_widget(q):
    label, help_text = INPUT_COPY.get(q.key, (q.label, q.help))
    if q.role == ROLE_RANGE:
        unit = f" ({q.unit})" if q.unit else ""
        return st.number_input(label + unit, min_value=0.1, max_value=1000.0, value=2.0, step=0.5,
                               format="%.1f", help=help_text or None, key=f"in_{q.key}")
    index = None
    if q.role == ROLE_FILTER:
        index = next((i for i, o in enumerate(q.options) if q.no_filter and norm(o) == norm(q.no_filter)), 0)
    return st.selectbox(label, list(q.options), index=index, placeholder="Select",
                        format_func=lambda v, k=q.key: show(k, v), help=help_text or None, key=f"in_{q.key}")


def main() -> None:
    st.set_page_config(page_title="Brush Cutter Finder", layout="wide", initial_sidebar_state="expanded")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

    blob, name = current_source()
    try:
        ds, biz = load_workbook_data(blob, name)
    except DataError as err:
        st.error("The workbook cannot be used yet.")
        for p in err.problems:
            st.markdown(f"- {p}")
        st.stop()
    sig = hashlib.md5(blob).hexdigest()

    with st.sidebar:
        groups = [
            ("Farm", [q for q in ds.questions if q.role in (ROLE_RANGE, ROLE_LOOKUP)]),
            ("Field conditions", [q for q in ds.questions if q.role == ROLE_KEY]),
            ("Preference", [q for q in ds.questions if q.role == ROLE_FILTER]),
        ]
        answers: dict = {}
        with st.form("farm_inputs"):
            for title, qs in groups:
                if qs:
                    if title != "Farm":
                        st.markdown(eyebrow(title), unsafe_allow_html=True)
                    for q in qs:
                        answers[q.key] = input_widget(q)
            submitted = st.form_submit_button("Get recommendations", use_container_width=True)

        if submitted:
            missing = [INPUT_COPY.get(q.key, (q.label,))[0] for q in ds.questions if answers.get(q.key) is None]
            if missing:
                st.error("Please complete: " + ", ".join(missing) + ".")
            else:
                try:
                    st.session_state["run"] = {"sig": sig, "answers": dict(answers),
                                               "decision": decide(ds, biz, answers)}
                except (LookupError, ValueError, DataError) as err:
                    st.session_state.pop("run", None)
                    st.error(str(err))

    run = st.session_state.get("run")
    if not run or run["sig"] != sig:
        st.markdown(onboarding_html(ds, biz), unsafe_allow_html=True)
        return

    d: Decision = run["decision"]

    st.markdown('<div class="section-title">Top 5 products for your farm</div>', unsafe_allow_html=True)
    if not d.shortlist:
        st.warning("No product is available with these choices. Try 'No preference' for power source.")
    elif len(d.shortlist) < d.top_n:
        st.info(f"Only {len(d.shortlist)} matching products are available for these choices.")
    cards = "".join(card_html(c, d, i) for i, c in enumerate(d.shortlist, 1))
    st.markdown(f'<div class="product-grid">{cards}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
