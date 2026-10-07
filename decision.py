"""
Decision layer on top of the technical engine.

engine.py answers "does this machine suit the field?". This module adds the other
two layers the workbook describes and combines them into one ranked shortlist:

  technical   score from engine.py                                   (0..100)
  economic    annual cost of ownership, cheapest model = 100         (0..100)
  commercial  stock, warranty and service support from the sheets    (0..100)

  overall = technical x w1 + economic x w2 + commercial x w3        (weights sheet, normalised)

Economic model (every number comes from the workbook)
  area worked per year = land (acres) x operations per year (crop_operations)
  hours per year       = area worked per year / coverage_area_acres_per_hr
  energy per year      = hours x fuel_consumption_per_hr x price
                         (petrol: litres x petrol price; battery / electric: kWh x electricity price)
  annual cost          = price / life_years + maintenance_per_year + energy per year
  economic score       = lowest annual cost among the candidate models / this model's annual cost x 100

The few workbook names this layer depends on are listed in the constants below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from engine import (
    DataError, Dataset, ModelResult, Recommendation, ROLE_LOOKUP, ROLE_RANGE,
    _read, _text, norm, recommend,
)

FEATURE_POWER = "power_source"          # models column that says how the machine is powered
BRAND_COLUMN = "brand"                  # models column that names the brand (links to brand_service)
ENERGY_OF = {"petrol": "petrol", "battery": "electricity", "electric": "electricity"}
FACTOR_POINTS = {"stock_status": "stock_status", "service_support": "service_support", "warranty": "warranty_months"}
DEFAULT_TOP_N, DEFAULT_MAX_PER_BRAND, DEFAULT_TIE_MARGIN = 5, 2, 2.0


def _num(value) -> float | None:
    try:
        return float(_text(value))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Business data
# --------------------------------------------------------------------------- #
@dataclass
class Business:
    crops: dict[str, tuple[str, float]]               # norm(crop) -> (crop, operations per year)
    economic: dict[str, dict[str, float]]             # norm(model) -> numbers
    commercial: dict[str, dict[str, str]]             # norm(model) -> stock_status, warranty_months
    service: dict[str, str]                           # norm(brand) -> service_support
    points: dict[tuple[str, str], float]              # (factor, norm(value)) -> points
    layer_weights: dict[str, float]                   # technical / economic / commercial, as written
    factor_weights: dict[str, float]                  # commercial factor -> weight, as written
    energy_prices: dict[str, tuple[float, str]]       # energy -> (price, unit)
    settings: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def load_business(path: str | Path) -> Business:
    sheets = _read(path)
    problems: list[str] = []
    required = {
        "crop_operations": ("crop", "operations_per_year"),
        "economic": ("model", "price", "life_years", "fuel_consumption_per_hr",
                     "coverage_area_acres_per_hr", "maintenance_per_year"),
        "commercial": ("model", "stock_status", "warranty_months"),
        "brand_service": ("brand", "service_support"),
        "commercial_points": ("factor", "value", "points"),
        "weights": ("layer", "weight", "commercial_factor"),
        "settings": ("key", "value"),
    }
    for sheet, cols in required.items():
        if sheet not in sheets:
            problems.append(f"Sheet '{sheet}' is missing.")
            continue
        for c in cols:
            if c not in sheets[sheet].columns:
                problems.append(f"Sheet '{sheet}' needs a '{c}' column.")
    if problems:
        raise DataError(problems)

    warnings: list[str] = []

    crops = {}
    for _, r in sheets["crop_operations"].iterrows():
        n = _num(r["operations_per_year"])
        if _text(r["crop"]) and n is not None and n > 0:
            crops[norm(r["crop"])] = (_text(r["crop"]), n)
        elif _text(r["crop"]):
            warnings.append(f"crop_operations: '{_text(r['crop'])}' has no usable operations_per_year.")

    economic = {}
    for _, r in sheets["economic"].iterrows():
        vals = {c: _num(r[c]) for c in required["economic"][1:]}
        if _text(r["model"]):
            if any(v is None for v in vals.values()) or vals["coverage_area_acres_per_hr"] <= 0 or vals["life_years"] <= 0:
                warnings.append(f"economic: '{_text(r['model'])}' has a missing or invalid number, so its cost cannot be estimated.")
            else:
                economic[norm(r["model"])] = vals

    commercial = {norm(r["model"]): {"stock_status": _text(r["stock_status"]), "warranty_months": _text(r["warranty_months"])}
                  for _, r in sheets["commercial"].iterrows() if _text(r["model"])}
    service = {norm(r["brand"]): _text(r["service_support"]) for _, r in sheets["brand_service"].iterrows() if _text(r["brand"])}
    points = {(norm(r["factor"]), norm(r["value"])): _num(r["points"]) or 0.0
              for _, r in sheets["commercial_points"].iterrows() if _text(r["factor"])}

    w = sheets["weights"]
    cols = list(w.columns)
    fcol = cols[cols.index("commercial_factor") + 1] if cols.index("commercial_factor") + 1 < len(cols) else None
    layer_weights = {norm(r["layer"]): _num(r["weight"]) or 0.0 for _, r in w.iterrows() if _text(r["layer"])}
    factor_weights = {norm(r["commercial_factor"]): _num(r[fcol]) or 0.0
                      for _, r in w.iterrows() if fcol and _text(r["commercial_factor"])}
    if sum(layer_weights.values()) <= 0:
        raise DataError(["weights: the layer weights add up to zero."])
    if sum(factor_weights.values()) <= 0:
        raise DataError(["weights: the commercial factor weights add up to zero."])

    settings = {norm(r["key"]): _num(r["value"]) for _, r in sheets["settings"].iterrows() if _text(r["key"])}
    settings = {k: v for k, v in settings.items() if v is not None}
    energy_prices = {}
    for k, v in settings.items():
        m = re.fullmatch(r"(.+)_price_per_(.+)", k)
        if m:
            energy_prices[m.group(1)] = (v, {"litre": "L", "liter": "L", "kwh": "kWh"}.get(m.group(2), m.group(2)))

    return Business(crops, economic, commercial, service, points, layer_weights, factor_weights,
                    energy_prices, settings, warnings)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Economics:
    hours: float
    energy_use: float
    energy_unit: str
    energy_price: float
    energy_cost: float
    capital: float
    maintenance: float
    annual: float
    per_pass: float          # annual cost per acre worked
    price: float
    life: float


@dataclass
class CommercialPart:
    factor: str
    shown: str
    points: float
    weight: float            # share of the commercial score (0..1)


@dataclass
class Candidate:
    result: ModelResult
    brand: str
    source: str
    technical: float
    economic: float = 0.0
    commercial: float = 0.0
    overall: float = 0.0
    econ: Economics | None = None
    parts: list[CommercialPart] = field(default_factory=list)
    stock: str = ""
    warranty: str = ""
    service: str = ""
    order: int = 0           # position in the overall ranking, 1-based
    gap_to_next: float | None = None
    close_to_next: bool = False


@dataclass
class Decision:
    rec: Recommendation
    shortlist: list[Candidate]
    rest: list[Candidate]            # ranked below the shortlist
    capped: list[Candidate]          # skipped by the per-brand limit
    acres: float
    crop: str
    ops: float
    passes: float                    # acres x operations per year
    layer_weights: dict[str, float]  # normalised to 1
    top_n: int
    max_per_brand: int
    tie_margin: float
    assumptions: list[str]


def _economics(biz: Business, model: str, source: str, passes: float) -> Economics | None:
    e = biz.economic.get(norm(model))
    if e is None:
        return None
    hours = passes / e["coverage_area_acres_per_hr"]
    energy = biz.energy_prices.get(ENERGY_OF.get(norm(source), ""))
    price, unit = energy if energy else (0.0, "")
    use = hours * e["fuel_consumption_per_hr"]
    energy_cost = use * price
    capital = e["price"] / e["life_years"]
    annual = capital + e["maintenance_per_year"] + energy_cost
    return Economics(hours, use, unit, price, energy_cost, capital, e["maintenance_per_year"], annual,
                     annual / passes if passes > 0 else 0.0, e["price"], e["life_years"])


def _points_for(biz: Business, factor: str, value: str) -> float:
    exact = biz.points.get((factor, norm(value)))
    if exact is not None:
        return exact
    number = _num(value)
    if number is not None:       # numeric tiers (warranty): use the highest tier the model reaches
        tiers = [(_num(v), p) for (f, v), p in biz.points.items() if f == factor and _num(v) is not None]
        reached = [t for t in tiers if t[0] <= number]
        if reached:
            return max(reached)[1]
    return 0.0


def _commercial(biz: Business, model: str, brand: str) -> tuple[float, list[CommercialPart], str, str, str]:
    c = biz.commercial.get(norm(model), {})
    values = {
        "stock_status": c.get("stock_status", ""),
        "service_support": biz.service.get(norm(brand), ""),
        "warranty": c.get("warranty_months", ""),
    }
    total_w = sum(biz.factor_weights.values())
    parts, score = [], 0.0
    for factor, fw in biz.factor_weights.items():
        shown = values.get(factor, "")
        pts = _points_for(biz, FACTOR_POINTS.get(factor, factor), shown) if shown else 0.0
        parts.append(CommercialPart(factor, shown, pts, fw / total_w))
        score += pts * fw / total_w
    return score, parts, values["stock_status"], values["warranty"], values["service_support"]


def decide(ds: Dataset, biz: Business, answers: dict) -> Decision:
    rec = recommend(ds, answers)

    rq, lq = ds.range_questions, ds.lookup_questions
    if not rq or not lq:
        raise DataError(["inputs: a 'range_key' input (land) and a 'lookup' input (crop) are needed for the cost layer."])
    acres = float(_text(answers[rq[0].key]))
    crop_name = _text(answers[lq[0].key])
    hit = biz.crops.get(norm(crop_name))
    if hit is None:
        raise ValueError(f"'{crop_name}' has no operations_per_year in the crop_operations sheet.")
    crop, ops = hit
    passes = acres * ops

    cands: list[Candidate] = []
    for r in rec.results:
        brand = r.meta.get(BRAND_COLUMN, "")
        source = next((c.offered[0] for c in r.checks if c.feature == FEATURE_POWER and c.offered), "")
        com, parts, stock, warranty, service = _commercial(biz, r.name, brand)
        cands.append(Candidate(result=r, brand=brand, source=source, technical=r.score,
                               econ=_economics(biz, r.name, source, passes),
                               commercial=com, parts=parts, stock=stock, warranty=warranty, service=service))

    costs = [c.econ.annual for c in cands if c.econ and c.econ.annual > 0]
    cheapest = min(costs) if costs else None
    lw_total = sum(biz.layer_weights.values())
    lw = {k: v / lw_total for k, v in biz.layer_weights.items()}
    for c in cands:
        c.economic = cheapest / c.econ.annual * 100 if cheapest and c.econ and c.econ.annual > 0 else 0.0
        c.overall = (c.technical * lw.get("technical", 0) + c.economic * lw.get("economic", 0)
                     + c.commercial * lw.get("commercial", 0))

    original = {id(c): i for i, c in enumerate(cands)}
    cands.sort(key=lambda c: (-round(c.overall, 9), -round(c.technical, 9), -round(c.result.coverage, 9), original[id(c)]))
    for i, c in enumerate(cands, 1):
        c.order = i

    top_n = int(biz.settings.get("top_n", DEFAULT_TOP_N))
    cap = int(biz.settings.get("max_per_brand", DEFAULT_MAX_PER_BRAND))
    tie = float(biz.settings.get("tie_margin", DEFAULT_TIE_MARGIN))
    shortlist: list[Candidate] = []
    capped: list[Candidate] = []
    rest: list[Candidate] = []
    per_brand: dict[str, int] = {}
    for c in cands:
        if len(shortlist) >= top_n:
            rest.append(c)
        elif per_brand.get(norm(c.brand), 0) >= cap:
            capped.append(c)
        else:
            shortlist.append(c)
            per_brand[norm(c.brand)] = per_brand.get(norm(c.brand), 0) + 1
    for a, b in zip(shortlist, shortlist[1:]):
        a.gap_to_next = a.overall - b.overall
        a.close_to_next = a.gap_to_next <= tie

    assumptions = [
        "Energy use per hour is taken from fuel_consumption_per_hr: litres for petrol models, kWh for battery and electric models.",
        "Annual cost = price / life_years + maintenance_per_year + energy cost. Maintenance is a fixed yearly figure and does not change with hours of use.",
        "Cost efficiency is relative: the cheapest eligible model scores 100 and the others score cheapest / own cost x 100.",
        "Commercial score uses stock status, warranty and brand service support with the weights on the weights sheet.",
        f"Operations per year for {crop} is {ops:g}, from the crop_operations sheet. Each operation covers the whole holding once.",
    ]
    if any(c.econ is None for c in cands):
        assumptions.append("Models with no usable row on the economic sheet get a cost score of 0.")
    assumptions.extend(biz.warnings)

    return Decision(rec=rec, shortlist=shortlist, rest=rest, capped=capped, acres=acres, crop=crop, ops=ops,
                    passes=passes, layer_weights=lw, top_n=top_n, max_per_brand=cap, tie_margin=tie,
                    assumptions=assumptions)
