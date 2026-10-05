"""The Q26 feature coverage report: aggregate only (brief 08 E.3).

What share of the universe rows, per year, has each characteristic before the
0-fill; how often each missing-value flag is set; how often a row has no
sector; and the data-layer counts behind them (the keyset-8 firm-quarters,
link ambiguities, the period-end mapping check, how quarters became
available). It never contains a model result or a return statistic, and
:func:`assert_aggregate_only` refuses to write anything that names a
security, so the report can live in ``results/feature_coverage/``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "PerSecurityValues",
    "feature_coverage",
    "assert_aggregate_only",
    "write_feature_coverage",
    "universe_firm_quarters",
    "rdq_coverage_by_fiscal_year",
    "early_years_table",
]

#: Keys a per-security record would carry; refused anywhere in a report.
IDENTIFIER_KEYS: frozenset[str] = frozenset({
    "permno", "lpermno", "permco", "gvkey", "kygvkey", "ticker", "cusip", "entity",
    "entity_labels", "security", "issuer",
})


class PerSecurityValues(ValueError):
    """A report holds something that identifies a security."""


def _walk(obj: Any, path: str = "") -> Iterable[tuple[str, Any, bool]]:
    """(path, item, is_key) for every key and leaf of nested dicts/lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}/{k}", k, True
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(obj, list | tuple):
        yield path, obj, False
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj, False


def assert_aggregate_only(
    report: dict[str, Any],
    forbidden: Iterable[str] = (),
    *,
    max_list_len: int = 200,
) -> None:
    """Raise :class:`PerSecurityValues` unless ``report`` is aggregate only.

    Refused: a key named like an identifier (:data:`IDENTIFIER_KEYS`), any
    key or string equal to one of ``forbidden`` (the panel's PERMNOs, the
    extract's GVKEYs), and any list longer than ``max_list_len`` (a per-row
    or per-security series, not a summary).
    """
    ids = {str(x) for x in forbidden}
    for path, item, is_key in _walk(report):
        if isinstance(item, list | tuple) and len(item) > max_list_len:
            raise PerSecurityValues(f"{path}: a list of {len(item)} values is not a summary")
        if is_key and str(item).lower() in IDENTIFIER_KEYS:
            raise PerSecurityValues(f"{path}: key {item!r} names a security identifier")
        if (is_key or isinstance(item, str)) and str(item) in ids:
            raise PerSecurityValues(f"{path}: {item!r} identifies a security")


def feature_coverage(panel_metadata: dict[str, Any], compustat_info: dict[str, Any]) -> dict:
    """The coverage report from a built Q26 panel's metadata and the
    Compustat extract's ``extract.json``. Aggregate numbers only."""
    features = panel_metadata.get("features")
    if not features or panel_metadata.get("feature_set") != "q26":
        raise ValueError("the panel carries no Q26 feature report: build it with feature_set='q26'")
    build = panel_metadata.get("build", {})
    q26 = build.get("q26_inputs", {})
    creport = compustat_info.get("report", {})
    return {
        "window": build.get("window"),
        "crsp_release": panel_metadata.get("crsp_release"),
        "crsp_stock_file": panel_metadata.get("crsp_stock_file"),
        "compustat_release": panel_metadata.get("compustat_release"),
        "sector_source": panel_metadata.get("sector_source"),
        "rows": features.get("rows"),
        "present_share": features.get("present_share"),
        "flag_price_rate": features.get("flag_price_rate"),
        "flag_fund_rate": features.get("flag_fund_rate"),
        "sector_missing_share": features.get("sector_missing_share"),
        "within_sector_fallback_share": features.get("within_sector_fallback_share"),
        "seq_window_incomplete_rows": features.get("seq_window_incomplete_rows"),
        # "year_2019", not "2019": a bare number can equal an identifier
        "by_year": {f"year_{y}": v for y, v in (features.get("by_year") or {}).items()},
        "attach_counts": {k: v for k, v in q26.items() if k != "quarters"},
        "quarters": q26.get("quarters"),
        "compustat": {
            "history_start": compustat_info.get("history_start"),
            "restated_keyset": creport.get("restated_keyset"),
            "restated_keyset_firm_quarters": creport.get("restated_keyset_firm_quarters"),
            "keyset_firm_quarters": {
                f"keyset_{k}": v for k, v in (creport.get("keyset_firm_quarters") or {}).items()
            },
            "link_ambiguities": creport.get("link_ambiguities"),
            "link_ambiguous_permnos": creport.get("link_ambiguous_permnos"),
            "gics_lpermno_disagreements": creport.get("gics_lpermno_disagreements"),
            "period_end_mapping": creport.get("period_end_mapping"),
            "filing_srctypes": creport.get("filing_srctypes"),
        },
    }


def universe_firm_quarters(
    quarters: pd.DataFrame, links: pd.DataFrame, spells: pd.DataFrame, cspec: Any
) -> pd.DataFrame:
    """The firm-quarters of index members: one row per (``KYGVKEY``,
    ``FYYYYQ``) whose GVKEY is linked (under ``cspec``'s link filter) on the
    quarter's period end to a PERMNO that is an index member that day.

    ``quarters`` is :func:`nec_moe.compustat.quarterly_fundamentals` output
    (``KYGVKEY``, ``FYYYYQ``, ``FYEARQ``, ``datadate``, ``rdq``,
    ``first_filing``); ``links`` the extract's raw link rows; ``spells`` the
    membership spells (``PERMNO``, ``MbrStartDt``, ``MbrEndDt``, both bounds
    inclusive). A quarter without a period end cannot be placed and is left
    out (it is counted in the period-end mapping check instead).
    """
    from .compustat import valid_links

    cols = ["KYGVKEY", "FYYYYQ", "FYEARQ", "datadate", "rdq", "first_filing"]
    q = quarters[cols].dropna(subset=["datadate"])
    v = valid_links(links, cspec)[["KYGVKEY", "LPERMNO", "LINKDT", "LINKENDDT"]]
    m = q.merge(v, on="KYGVKEY", how="inner")
    m = m[(m["LINKDT"] <= m["datadate"]) & (m["datadate"] <= m["LINKENDDT"])]
    s = spells[["PERMNO", "MbrStartDt", "MbrEndDt"]].astype({"PERMNO": "int64"})
    m = m.merge(s, left_on="LPERMNO", right_on="PERMNO", how="inner")
    m = m[(m["MbrStartDt"] <= m["datadate"]) & (m["datadate"] <= m["MbrEndDt"])]
    return m[cols].drop_duplicates(["KYGVKEY", "FYYYYQ"]).reset_index(drop=True)


def rdq_coverage_by_fiscal_year(universe_quarters: pd.DataFrame) -> dict[str, dict]:
    """Per fiscal year (``FYEARQ``): the members' firm-quarters, and the share
    with a report date (RDQ) and with a 10-Q/10-K filing date. Keys are
    ``"fiscal_<year>"`` (a bare number could equal an identifier)."""
    out: dict[str, dict] = {}
    for year, g in universe_quarters.groupby("FYEARQ"):
        n = len(g)
        out[f"fiscal_{int(year)}"] = {
            "firm_quarters": int(n),
            "rdq_share": float(g["rdq"].notna().mean()) if n else 0.0,
            "filing_share": float(g["first_filing"].notna().mean()) if n else 0.0,
        }
    return out


def early_years_table(
    report: dict[str, Any], rdq_by_year: dict[str, dict], first: int, last: int
) -> pd.DataFrame:
    """One row per year ``first..last``: the panel's rows and flag rates by
    **calendar** year (from the coverage report's ``by_year``) beside the
    members' RDQ and filing-date coverage by **fiscal** year. Aggregate only."""
    if last < first:
        raise ValueError(f"early-years range {first}..{last} is empty")
    by_year = report.get("by_year") or {}
    rows = {}
    for year in range(first, last + 1):
        cal = by_year.get(f"year_{year}", {})
        fis = rdq_by_year.get(f"fiscal_{year}", {})
        rows[f"year_{year}"] = {
            "rows": cal.get("rows"),
            "flag_price_rate": cal.get("flag_price_rate"),
            "flag_fund_rate": cal.get("flag_fund_rate"),
            "sector_missing_share": cal.get("sector_missing_share"),
            "member_firm_quarters": fis.get("firm_quarters"),
            "rdq_share": fis.get("rdq_share"),
            "filing_share": fis.get("filing_share"),
        }
    return pd.DataFrame(rows).T


def _by_year_frames(report: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_year = report.get("by_year") or {}
    present = pd.DataFrame({y: v["present_share"] for y, v in by_year.items()}).T
    rates = pd.DataFrame({
        y: {"rows": v["rows"], "flag_price_rate": v["flag_price_rate"],
            "flag_fund_rate": v["flag_fund_rate"],
            "sector_missing_share": v["sector_missing_share"]}
        for y, v in by_year.items()
    }).T
    return present, rates


def write_feature_coverage(
    report: dict[str, Any],
    out_dir: str | Path,
    forbidden: Iterable[str],
    tables: dict[str, pd.DataFrame] | None = None,
) -> list[Path]:
    """Check, then write ``coverage.json``, ``present_share_by_year.csv`` and
    ``rates_by_year.csv`` into ``out_dir``, plus ``<name>.csv`` for every
    extra table in ``tables`` (e.g. the early-years table of brief 09 A.3).
    Nothing is written if any check fails."""
    forbidden = list(forbidden)
    assert_aggregate_only(report, forbidden)
    present, rates = _by_year_frames(report)
    frames = {"present_share_by_year": present, "rates_by_year": rates, **(tables or {})}
    for frame in frames.values():
        assert_aggregate_only({"index": [str(i) for i in frame.index],
                               "columns": [str(c) for c in frame.columns]}, forbidden)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "coverage.json"]
    paths[0].write_text(json.dumps(report, indent=2, default=str))
    for name, frame in frames.items():
        paths.append(out / f"{name}.csv")
        frame.to_csv(paths[-1])
    return paths
