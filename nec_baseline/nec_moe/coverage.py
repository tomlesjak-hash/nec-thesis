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
    report: dict[str, Any], out_dir: str | Path, forbidden: Iterable[str]
) -> list[Path]:
    """Check, then write ``coverage.json``, ``present_share_by_year.csv`` and
    ``rates_by_year.csv`` into ``out_dir``. Nothing is written if the check
    fails."""
    forbidden = list(forbidden)
    assert_aggregate_only(report, forbidden)
    present, rates = _by_year_frames(report)
    for frame in (present, rates):
        assert_aggregate_only({"index": [str(i) for i in frame.index],
                               "columns": [str(c) for c in frame.columns]}, forbidden)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "coverage.json", out / "present_share_by_year.csv", out / "rates_by_year.csv"]
    paths[0].write_text(json.dumps(report, indent=2, default=str))
    present.to_csv(paths[1])
    rates.to_csv(paths[2])
    return paths
