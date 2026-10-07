"""
Technical scoring engine for the brush cutter advisor.

Everything here is driven by the workbook. No feature, input, value or
model name is written in this file.

Workbook contract
-----------------
inputs       input | allowed_values | role | no_filter_value  (+ optional label | help | unit)
               role = rule_key  -> picks the row in `rules` and `feature_rank` (exact match)
               role = range_key -> a number the farmer types. `rules` and `feature_rank` carry
                                   <input>_min and <input>_max columns; a row applies when
                                   min <= answer < max. Blank max = no upper limit.
               role = filter    -> only removes models that do not offer the answer
               role = lookup    -> a pick that is not scored here; it is carried through for the
                                   decision layer (crop -> operations per year)
rules        <rule_key columns> + <range_key _min/_max columns> + <feature columns>
               list feature : one column, suitable values split by |
               range feature: two columns X_min and X_max (numbers). Feature name is X.
feature_rank same key columns as rules + <feature names>   cells hold a rank (1 = most important)
rank_points  rank | points                                lookup table
models       <id columns> + <feature names>               cells hold what the model offers, split by |
               range features hold numbers or ranges, e.g. "12-14 inch|16.5 inch"
units        (optional) feature | unit                    unit the rule ranges are written in

Method
------
weight(feature)   = points(rank(feature)) / sum of points over every scored feature in that row
applicable        = the model lists at least one value for the feature. A blank cell means
                    "not applicable" (e.g. stroke on a battery model): the feature is left
                    out for that model and the remaining weights are rescaled to add to 100%
match(list)       = 1 when the model offers at least one value the rules list as suitable
match(range)      = 1 when at least one value the model offers lies entirely inside [min, max]
                    (a model range such as 12-14 must have both ends inside the rule range)
score             = sum(weight x match) x 100, over applicable features
ties              = broken by how many of the suitable options a model offers (coverage);
                    a range feature has no option list, so it counts 1 when matched, else 0
unit check        = if the rule unit and the unit written in the models differ (HP vs cc),
                    the feature cannot be compared. It is left out of scoring for everyone
                    and reported, never guessed.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SEPARATOR = "|"
SHEETS = ("rules", "feature_rank", "models", "rank_points", "inputs")
ROLE_KEY, ROLE_FILTER, ROLE_RANGE, ROLE_LOOKUP = "rule_key", "filter", "range_key", "lookup"
ROLES = (ROLE_KEY, ROLE_FILTER, ROLE_RANGE, ROLE_LOOKUP)


class DataError(Exception):
    """The workbook cannot be used. `problems` lists everything that is wrong."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


# --------------------------------------------------------------------------- #
# Cell helpers
# --------------------------------------------------------------------------- #
def _text(value) -> str:
    """Excel cell -> clean string. Blank cells become ''. 2.0 becomes '2'."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return re.sub(r"\s+", " ", str(value)).strip()


def norm(value) -> str:
    """Comparison key: case and spacing never decide a match."""
    return _text(value).casefold()


def split_values(cell) -> tuple[str, ...]:
    """'a | b|c' -> ('a', 'b', 'c'); duplicates dropped; blank -> ()."""
    seen, out = set(), []
    for part in _text(cell).split(SEPARATOR):
        part = _text(part)
        if part and norm(part) not in seen:
            seen.add(norm(part))
            out.append(part)
    return tuple(out)


def label_of(column: str) -> str:
    text = column.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Question:
    key: str
    label: str
    role: str
    options: tuple[str, ...]
    no_filter: str | None
    help: str = ""
    unit: str = ""


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float
    unit: str = ""

    def label(self) -> str:
        core = f"{self.lo:g}" if self.lo == self.hi else f"{self.lo:g} to {self.hi:g}"
        return f"{core} {self.unit}".strip()


@dataclass(frozen=True)
class Span:
    lo: float
    hi: float
    unit: str
    raw: str


_SPAN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:[-\u2013\u2014]|\bto\b)?\s*(\d+(?:\.\d+)?)?\s*([^\d]*)$", re.I)


def parse_span(token: str) -> Span | None:
    """'12-14 inch' -> Span(12, 14, 'inch'); '16.5 inch' -> Span(16.5, 16.5, 'inch'); junk -> None."""
    m = _SPAN.match(_text(token))
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    return Span(lo, hi, _text(m.group(3)), _text(token)) if hi >= lo else None


EPS = 1e-9
Band = tuple  # (lo, hi) with hi None = no upper limit
RowKey = tuple  # (texts, bands)


@dataclass
class Dataset:
    questions: list[Question]
    features: list[str]
    id_columns: list[str]
    rules: dict[RowKey, dict[str, object]]   # tuple of values, or Range for range features
    ranks: dict[RowKey, dict[str, float]]
    points: dict[float, float]
    models: list[dict]            # {"id": {col: text}, "values": {feature: tuple}}
    warnings: list[str] = field(default_factory=list)
    source: str = ""
    range_features: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    unscored: dict[str, str] = field(default_factory=dict)   # feature -> reason it cannot be scored

    @property
    def key_questions(self) -> list[Question]:
        return [q for q in self.questions if q.role == ROLE_KEY]

    @property
    def range_questions(self) -> list[Question]:
        return [q for q in self.questions if q.role == ROLE_RANGE]

    @property
    def filter_questions(self) -> list[Question]:
        return [q for q in self.questions if q.role == ROLE_FILTER]

    @property
    def lookup_questions(self) -> list[Question]:
        return [q for q in self.questions if q.role == ROLE_LOOKUP]


@dataclass
class FeatureWeight:
    feature: str
    label: str
    rank: float
    points: float
    weight: float                 # 0..1, sums to 1 over the row
    suitable: tuple[str, ...]     # display values; a range feature has one entry such as '12 to 16.5 inch'
    rng: Range | None = None


@dataclass
class Check:
    feature: str
    label: str
    weight: float                 # effective weight for this model (0 when not applicable)
    base_weight: float            # weight before leaving out non-applicable features
    rank: float
    points: float
    suitable: tuple[str, ...]
    offered: tuple[str, ...]
    matched: tuple[str, ...]      # offered values that are suitable
    extra: tuple[str, ...]        # offered values that are not suitable
    missing: tuple[str, ...]      # suitable values the model does not offer
    applicable: bool
    is_match: bool
    earned: float                 # score points gained (0..100 scale)
    lost: float
    coverage: float               # share of suitable options the model offers
    is_range: bool = False
    partial: tuple[str, ...] = ()  # range features: offered values that overlap the range but are not fully inside


@dataclass
class ModelResult:
    name: str
    meta: dict[str, str]
    checks: list[Check]
    score: float                  # 0..100
    matched_points: float
    coverage: float               # 0..1 tie-break
    position: int = 0             # 1-based order in the list
    rank: int = 0                 # competition rank (ties share a number)
    tied_with_previous: bool = False
    tied_with_next: bool = False
    matched_count: int = 0
    total_count: int = 0          # applicable features
    na_count: int = 0
    gaps_vs_top: list[Check] = field(default_factory=list)   # top matches, this misses
    leads_vs_top: list[Check] = field(default_factory=list)  # this matches, top misses


@dataclass
class Excluded:
    name: str
    meta: dict[str, str]
    reasons: list[str]


@dataclass
class Recommendation:
    answers: dict[str, str]
    weights: list[FeatureWeight]   # sorted by weight, high to low
    total_points: float
    results: list[ModelResult]
    excluded: list[Excluded]
    active_filters: list[tuple[str, str]]
    unscored: list[tuple[str, str]] = field(default_factory=list)   # (feature label, reason)
    rule_row: str = ""             # plain-language description of the rule row that was used


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def _read(path: str | Path) -> dict[str, pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name=None, dtype=object, engine="openpyxl")
    sheets = {}
    for name, frame in raw.items():
        frame = frame.copy()
        frame.columns = [_text(c) for c in frame.columns]
        frame = frame.loc[:, [c for c in frame.columns if c and not c.startswith("Unnamed")]]
        frame = frame.dropna(how="all")
        sheets[_text(name).casefold()] = frame.reset_index(drop=True)
    return sheets


def _overlap(a: Band, b: Band) -> bool:
    inf = float("inf")
    return a[0] < (b[1] if b[1] is not None else inf) and b[0] < (a[1] if a[1] is not None else inf)


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    problems: list[str] = []
    warnings: list[str] = []
    sheets = _read(path)

    for name in SHEETS:
        if name not in sheets:
            problems.append(f"Sheet '{name}' is missing.")
    if problems:
        raise DataError(problems)

    inputs, rules, ranks_df, models, rp = (sheets[s] for s in ("inputs", "rules", "feature_rank", "models", "rank_points"))

    units: dict[str, str] = {}
    if "units" in sheets:
        u = sheets["units"]
        if {"feature", "unit"} <= set(u.columns):
            units = {_text(r["feature"]): _text(r["unit"]) for _, r in u.iterrows() if _text(r["feature"])}
        else:
            warnings.append("Sheet 'units' needs 'feature' and 'unit' columns, so it was ignored.")

    # ---- inputs -----------------------------------------------------------
    for col in ("input", "allowed_values", "role"):
        if col not in inputs.columns:
            problems.append(f"Sheet 'inputs' needs a '{col}' column.")
    if problems:
        raise DataError(problems)

    questions: list[Question] = []
    for i, row in inputs.iterrows():
        key, role = _text(row["input"]), _text(row["role"]).casefold()
        if not key:
            continue
        options = split_values(row["allowed_values"])
        no_filter = _text(row["no_filter_value"]) if "no_filter_value" in inputs.columns else ""
        if role not in ROLES:
            problems.append(f"inputs: '{key}' has role '{role}'. Use one of: {', '.join(ROLES)}.")
        if role != ROLE_RANGE and not options:
            problems.append(f"inputs: '{key}' has no allowed_values.")
        if no_filter and norm(no_filter) not in {norm(o) for o in options}:
            problems.append(f"inputs: no_filter_value '{no_filter}' is not one of the allowed values of '{key}'.")
        questions.append(Question(
            key=key,
            label=_text(row["label"]) if "label" in inputs.columns and _text(row["label"]) else label_of(key),
            role=role,
            options=options,
            no_filter=no_filter or None,
            help=_text(row["help"]) if "help" in inputs.columns else "",
            unit=_text(row["unit"]) if "unit" in inputs.columns else "",
        ))
    if len({q.key for q in questions}) != len(questions):
        problems.append("inputs: an input name appears more than once.")
    key_inputs = [q for q in questions if q.role == ROLE_KEY]
    range_inputs = [q for q in questions if q.role == ROLE_RANGE]
    if not key_inputs and not range_inputs:
        problems.append("inputs: at least one input needs role 'rule_key' or 'range_key'.")

    # ---- column structure ---------------------------------------------------
    key_cols = [q.key for q in key_inputs]
    band_cols = [c for q in range_inputs for c in (q.key + "_min", q.key + "_max")]
    all_keys = key_cols + band_cols
    for sheet_name, frame in (("rules", rules), ("feature_rank", ranks_df)):
        for k in all_keys:
            if k not in frame.columns:
                problems.append(f"Sheet '{sheet_name}' has no '{k}' column.")
    if problems:
        raise DataError(problems)

    rule_cols = [c for c in rules.columns if c not in all_keys]
    range_feats = [c[:-4] for c in rule_cols if c.endswith("_min") and c[:-4] + "_max" in rule_cols]
    paired = {f + suffix for f in range_feats for suffix in ("_min", "_max")}
    features: list[str] = []
    for c in rule_cols:
        base = c[:-4] if c in paired else c
        if base not in features:
            features.append(base)
    for f in range_feats:
        if units and f not in units:
            warnings.append(f"Sheet 'units' has no unit for '{f}'.")
    if not features:
        problems.append("Sheet 'rules' has no feature columns.")
    if set(features) != {c for c in ranks_df.columns if c not in all_keys}:
        problems.append("Sheets 'rules' and 'feature_rank' must have the same feature columns.")
    missing = [f for f in features if f not in models.columns]
    if missing:
        problems.append("Sheet 'models' is missing feature columns: " + ", ".join(missing) + ".")
    for q in questions:
        if q.role == ROLE_FILTER and q.key not in models.columns:
            problems.append(f"Sheet 'models' has no '{q.key}' column to filter on.")
    if problems:
        raise DataError(problems)

    id_columns = [c for c in models.columns if c not in features]
    if not id_columns:
        problems.append("Sheet 'models' needs a column that names each model.")

    # ---- rank_points --------------------------------------------------------
    points: dict[float, float] = {}
    if not {"rank", "points"} <= set(rp.columns):
        problems.append("Sheet 'rank_points' needs 'rank' and 'points' columns.")
    else:
        for _, r in rp.iterrows():
            try:
                rank, pts = float(r["rank"]), float(r["points"])
            except (TypeError, ValueError):
                problems.append(f"rank_points: '{_text(r['rank'])}' / '{_text(r['points'])}' is not a number.")
                continue
            if rank in points:
                problems.append(f"rank_points: rank {rank:g} appears twice.")
            if pts < 0:
                problems.append(f"rank_points: rank {rank:g} has negative points.")
            points[rank] = pts
    if problems:
        raise DataError(problems)

    # ---- rules + feature_rank ----------------------------------------------
    def row_key(r: pd.Series, sheet: str, row_no: int) -> RowKey | None:
        texts = tuple(norm(r[c]) for c in key_cols)
        bands = []
        for q in range_inputs:
            lo_t, hi_t = _text(r[q.key + "_min"]), _text(r[q.key + "_max"])
            try:
                lo = float(lo_t) if lo_t else 0.0
                hi = float(hi_t) if hi_t else None
            except ValueError:
                problems.append(f"{sheet}: row {row_no} has a {q.key} bound that is not a number.")
                return None
            if hi is not None and hi <= lo:
                problems.append(f"{sheet}: row {row_no} has {q.key}_max ({hi:g}) not above {q.key}_min ({lo:g}).")
                return None
            bands.append((lo, hi))
        return texts, tuple(bands)

    def describe(k: RowKey) -> str:
        texts, bands = k
        parts = [f"{c} = {t}" for c, t in zip(key_cols, texts)]
        for q, (lo, hi) in zip(range_inputs, bands):
            parts.append(f"{q.key} {lo:g} to {hi:g}" if hi is not None else f"{q.key} {lo:g} and above")
        return " / ".join(parts)

    def keyed(frame: pd.DataFrame, sheet: str) -> dict[RowKey, pd.Series]:
        out: dict[RowKey, pd.Series] = {}
        for i, r in frame.iterrows():
            k = row_key(r, sheet, i + 2)
            if k is None:
                continue
            if k in out:
                problems.append(f"{sheet}: the row {describe(k)} appears twice.")
            out[k] = r
        return out

    rule_rows, rank_rows = keyed(rules, "rules"), keyed(ranks_df, "feature_rank")
    for k in [k for k in rule_rows if k not in rank_rows]:
        problems.append(f"Row {describe(k)} exists only in rules.")
    for k in [k for k in rank_rows if k not in rule_rows]:
        problems.append(f"Row {describe(k)} exists only in feature_rank.")

    rules_map: dict[RowKey, dict[str, object]] = {}
    ranks_map: dict[RowKey, dict[str, float]] = {}
    for k in [k for k in rule_rows if k in rank_rows]:
        label = describe(k)
        rules_map[k], ranks_map[k] = {}, {}
        for f in features:
            if f in range_feats:
                try:
                    lo, hi = float(_text(rule_rows[k][f + "_min"])), float(_text(rule_rows[k][f + "_max"]))
                except ValueError:
                    problems.append(f"rules [{label}]: '{f}_min' and '{f}_max' must both be numbers.")
                    rules_map[k][f] = None
                else:
                    if lo > hi:
                        problems.append(f"rules [{label}]: '{f}_min' ({lo:g}) is bigger than '{f}_max' ({hi:g}).")
                    rules_map[k][f] = Range(lo, hi, units.get(f, ""))
            else:
                vals = split_values(rule_rows[k][f])
                if not vals:
                    problems.append(f"rules [{label}]: '{f}' is blank.")
                rules_map[k][f] = vals
            try:
                rk = float(rank_rows[k][f])
            except (TypeError, ValueError):
                problems.append(f"feature_rank [{label}]: '{f}' is not a number.")
                continue
            if rk not in points:
                problems.append(f"feature_rank [{label}]: rank {rk:g} for '{f}' is not in rank_points.")
            ranks_map[k][f] = rk
        if not problems and sum(points[ranks_map[k][f]] for f in features) <= 0:
            problems.append(f"feature_rank [{label}]: points add up to zero.")

    # bands for the same rule_key values must not overlap
    groups: dict[tuple, list[tuple]] = {}
    for texts, bands in rules_map:
        groups.setdefault(texts, []).append(bands)
    for texts, group in groups.items():
        for a, b in itertools.combinations(group, 2):
            if all(_overlap(x, y) for x, y in zip(a, b)):
                problems.append("rules: overlapping bands for " + describe((texts, a)) + " and " + describe((texts, b)) + ".")

    # input values vs rules coverage
    for q in key_inputs:
        seen = {norm(r[q.key]) for r in rule_rows.values()}
        for o in q.options:
            if norm(o) not in seen:
                warnings.append(f"'{o}' is offered for {q.label} but no rule uses it.")
        for s in seen - {norm(o) for o in q.options}:
            warnings.append(f"Rules use {q.label} value '{s}' that is not in the inputs sheet.")
    for combo in itertools.product(*[[norm(o) for o in q.options] for q in key_inputs]):
        bands_here = groups.get(combo)
        if not bands_here:
            warnings.append("No rule row for " + (" / ".join(combo) or "any combination") + ".")
            continue
        for j, q in enumerate(range_inputs):
            ordered = sorted((b[j] for b in bands_here), key=lambda b: b[0])
            if ordered[0][0] > 0:
                warnings.append(f"{label_of(q.key)} below {ordered[0][0]:g} has no rule for {' / '.join(combo)}.")
            for (a_lo, a_hi), (b_lo, _) in zip(ordered, ordered[1:]):
                if a_hi is not None and a_hi < b_lo:
                    warnings.append(f"{label_of(q.key)} between {a_hi:g} and {b_lo:g} has no rule for {' / '.join(combo)}.")
            if ordered[-1][1] is not None:
                warnings.append(f"{label_of(q.key)} above {ordered[-1][1]:g} has no rule for {' / '.join(combo)}.")

    # ---- models -------------------------------------------------------------
    model_list: list[dict] = []
    names: set[str] = set()
    model_units: dict[str, set[str]] = {f: set() for f in range_feats}
    for i, r in models.iterrows():
        ident = {c: _text(r[c]) for c in id_columns}
        name = ident[id_columns[0]]
        if not name:
            problems.append(f"models: row {i + 2} has no {id_columns[0]}.")
            continue
        if norm(name) in names:
            warnings.append(f"Model '{name}' appears more than once.")
        names.add(norm(name))
        values, spans = {}, {}
        for f in features:
            values[f] = split_values(r[f])
            if f in range_feats:
                parsed = []
                for tok in values[f]:
                    sp = parse_span(tok)
                    if sp is None:
                        problems.append(f"models: {name} {f} value '{tok}' is not a number or a range like 12-14.")
                    else:
                        parsed.append(sp)
                        if sp.unit:
                            model_units[f].add(norm(sp.unit))
                spans[f] = tuple(parsed)
        model_list.append({"id": ident, "values": values, "spans": spans})

    unscored: dict[str, str] = {}
    for f in range_feats:
        declared = units.get(f, "")
        if declared and model_units[f] and model_units[f] != {norm(declared)}:
            listed = ", ".join(sorted(model_units[f]))
            unscored[f] = f"the rules are written in {declared} but the models list {listed}"
            warnings.append(f"{label_of(f)} is not scored: {unscored[f]}. "
                            f"Enter the models' {label_of(f).lower()} in {declared} to switch it on.")
    # models and rules must speak the same language: flag values that can never match or never get used
    for f in features:
        if f in range_feats:
            continue
        rule_vocab = {norm(v): v for k in rules_map for v in (rules_map[k].get(f) or ())}
        model_vocab = {norm(v): v for m in model_list for v in m["values"][f]}
        for key, v in model_vocab.items():
            if key not in rule_vocab:
                warnings.append(f"{label_of(f)}: models offer '{v}' but no rule ever lists it, so it can never score.")
        for key, v in rule_vocab.items():
            if key not in model_vocab:
                warnings.append(f"{label_of(f)}: rules list '{v}' but no model offers it.")

    if not model_list:
        problems.append("Sheet 'models' has no models.")

    if problems:
        raise DataError(problems)

    return Dataset(
        questions=questions, features=features, id_columns=id_columns,
        rules=rules_map, ranks=ranks_map, points=points,
        models=model_list, warnings=warnings, source=path.name,
        range_features=range_feats, units=units, unscored=unscored,
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def rule_key(ds: Dataset, answers: dict) -> RowKey:
    """The rule row that applies to these answers (exact match on rule_key inputs, band on range_key inputs)."""
    texts = tuple(norm(answers[q.key]) for q in ds.key_questions)
    values = [float(_text(answers[q.key])) for q in ds.range_questions]
    for k in ds.ranks:
        if k[0] == texts and all(lo <= v and (hi is None or v < hi) for v, (lo, hi) in zip(values, k[1])):
            return k
    shown = ", ".join([str(answers[q.key]) for q in ds.key_questions] +
                      [f"{q.label} {answers[q.key]}" for q in ds.range_questions])
    raise LookupError("No expert guidance for this combination: " + shown)


def describe_row(ds: Dataset, key: RowKey) -> str:
    texts, bands = key
    parts = list(texts)
    for q, (lo, hi) in zip(ds.range_questions, bands):
        unit = f" {q.unit}" if q.unit else ""
        parts.append(f"{q.label.lower()} {lo:g} to {hi:g}{unit}" if hi is not None else f"{q.label.lower()} {lo:g}{unit} and above")
    return " / ".join(parts)


def _is_range_rule(value) -> bool:
    return all(hasattr(value, name) for name in ("lo", "hi", "label"))


def _coerce_spans(spans, offered: tuple[str, ...]) -> tuple[Span, ...]:
    try:
        parsed = tuple(sp for sp in spans if all(hasattr(sp, name) for name in ("lo", "hi", "raw")))
    except TypeError:
        parsed = ()
    if parsed:
        return parsed
    return tuple(sp for token in offered if (sp := parse_span(token)) is not None)


def derive_weights(ds: Dataset, answers: dict) -> tuple[list[FeatureWeight], float, RowKey]:
    """rank -> points (lookup) -> points / row total -> weight."""
    key = rule_key(ds, answers)
    scored = [f for f in ds.features if f not in ds.unscored]
    pts = {f: ds.points[ds.ranks[key][f]] for f in scored}
    total = sum(pts.values())
    rows = []
    for f in scored:
        rule = ds.rules[key][f]
        rng = rule if _is_range_rule(rule) else None
        rows.append(FeatureWeight(f, label_of(f), ds.ranks[key][f], pts[f], pts[f] / total,
                                  (rule.label(),) if rng else rule, rng))
    rows.sort(key=lambda w: (-w.weight, ds.features.index(w.feature)))
    return rows, total, key


def _check(w: FeatureWeight, offered: tuple[str, ...], applicable_points: float,
           spans: tuple[Span, ...] = ()) -> Check:
    applicable = bool(offered)
    partial: tuple[str, ...] = ()
    rng = w.rng if w.rng is not None else (w.suitable if _is_range_rule(w.suitable) else None)
    suitable = (rng.label(),) if rng is not None else w.suitable
    if rng is not None:
        spans = _coerce_spans(spans, offered)
        within = tuple(sp.raw for sp in spans if sp.lo >= rng.lo - EPS and sp.hi <= rng.hi + EPS)
        partial = tuple(sp.raw for sp in spans if sp.raw not in within
                        and sp.lo <= rng.hi + EPS and sp.hi >= rng.lo - EPS)
        matched, extra, missing = within, tuple(v for v in offered if v not in within), ()
        coverage = 1.0 if applicable and matched else 0.0
    else:
        suitable_keys = {norm(v) for v in suitable}
        offered_keys = {norm(v) for v in offered}
        matched = tuple(v for v in offered if norm(v) in suitable_keys)
        extra = tuple(v for v in offered if norm(v) not in suitable_keys)
        missing = tuple(v for v in suitable if norm(v) not in offered_keys)
        coverage = len(matched) / len(suitable) if suitable and applicable else 0.0
    is_match = applicable and bool(matched)
    eff = w.points / applicable_points if applicable and applicable_points > 0 else 0.0
    return Check(
        feature=w.feature, label=w.label, weight=eff, base_weight=w.weight, rank=w.rank, points=w.points,
        suitable=suitable, offered=offered, matched=matched, extra=extra, missing=missing,
        applicable=applicable, is_match=is_match,
        earned=eff * 100 if is_match else 0.0,
        lost=eff * 100 if applicable and not is_match else 0.0,
        coverage=coverage, is_range=rng is not None, partial=partial,
    )


def recommend(ds: Dataset, answers: dict) -> Recommendation:
    for q in ds.questions:
        raw = answers.get(q.key)
        if q.role == ROLE_RANGE:
            try:
                value = float(_text(raw))
            except ValueError:
                raise ValueError(f"{q.label} must be a number.") from None
            if value < 0:
                raise ValueError(f"{q.label} cannot be negative.")
            continue
        if not _text(raw):
            raise ValueError(f"Answer missing for {q.label}.")
        if norm(raw) not in {norm(o) for o in q.options}:
            raise ValueError(f"'{raw}' is not an allowed answer for {q.label}.")

    weights, total, key = derive_weights(ds, answers)

    active = [
        (q, answers[q.key]) for q in ds.filter_questions
        if not (q.no_filter and norm(answers[q.key]) == norm(q.no_filter))
    ]

    kept, excluded = [], []
    for m in ds.models:
        reasons = []
        for q, ans in active:
            offered = m["values"].get(q.key) if q.key in m["values"] else split_values(m["id"].get(q.key, ""))
            if norm(ans) not in {norm(v) for v in offered}:
                shown = ", ".join(offered) if offered else "not listed"
                reasons.append(f"{q.label}: {shown}")
        (excluded if reasons else kept).append((m, reasons))

    def meta(m):
        return {c: v for c, v in m["id"].items() if c != ds.id_columns[0] and v}

    results: list[ModelResult] = []
    for m, _ in kept:
        applicable_points = sum(w.points for w in weights if m["values"][w.feature])
        checks = [_check(w, m["values"][w.feature], applicable_points, m["spans"].get(w.feature, ())) for w in weights]
        matched_points = sum(c.points for c in checks if c.is_match)
        cov_points = sum(c.points * c.coverage for c in checks)
        applicable_n = sum(c.applicable for c in checks)
        results.append(ModelResult(
            name=m["id"][ds.id_columns[0]], meta=meta(m), checks=checks,
            score=matched_points / applicable_points * 100 if applicable_points else 0.0,
            matched_points=matched_points,
            coverage=cov_points / applicable_points if applicable_points else 0.0,
            matched_count=sum(c.is_match for c in checks), total_count=applicable_n,
            na_count=len(checks) - applicable_n,
        ))

    order = {id(r): i for i, r in enumerate(results)}
    results.sort(key=lambda r: (-round(r.score, 9), -round(r.coverage, 9), order[id(r)]))

    prev = None
    for i, r in enumerate(results, 1):
        r.position = i
        same = prev is not None and round(prev.score, 9) == round(r.score, 9) \
            and round(prev.coverage, 9) == round(r.coverage, 9)
        r.tied_with_previous = same
        r.rank = prev.rank if same else i
        if same:
            prev.tied_with_next = True
        prev = r

    if results:
        top = results[0]
        for r in results[1:]:
            r.gaps_vs_top = [b for a, b in zip(top.checks, r.checks) if a.is_match and b.applicable and not b.is_match]
            r.leads_vs_top = [b for a, b in zip(top.checks, r.checks) if b.is_match and a.applicable and not a.is_match]

    return Recommendation(
        answers=dict(answers), weights=weights, total_points=total, results=results,
        excluded=[Excluded(m["id"][ds.id_columns[0]], meta(m), rs) for m, rs in excluded],
        active_filters=[(q.label, a) for q, a in active],
        unscored=[(label_of(f), why) for f, why in ds.unscored.items()],
        rule_row=describe_row(ds, key),
    )
