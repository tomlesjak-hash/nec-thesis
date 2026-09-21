"""Trexsim seasonality variables, computed locally — zero download.

Reproduces 17 of the platform's Seasonality variables from a trading calendar:
15 from GROUP_days_to_calendar_events + 2 from GROUP_fomc_dates.

All are 1D (identical across stocks on a given day), so on their own they
neutralize to zero. Their only use is as MULTIPLICATIVE GATES on a 2D signal.
Returned as Series indexed by date; broadcast across tickers when you use them.

Trexsim marks every one of these Delay 0 = No -> lag by one day in local fitness.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["calendar_features", "FOMC_DATES"]

#: FOMC meeting end-dates (scheduled 8x/year). Extend as needed; only the
#: *dates* matter, not the decisions.
FOMC_DATES = pd.to_datetime([
    "2011-01-26","2011-03-15","2011-04-27","2011-06-22","2011-08-09","2011-09-21","2011-11-02","2011-12-13",
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20","2012-08-01","2012-09-13","2012-10-24","2012-12-12",
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19","2013-07-31","2013-09-18","2013-10-30","2013-12-18",
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18","2014-07-30","2014-09-17","2014-10-29","2014-12-17",
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-03-15","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10",
    "2026-01-28","2026-03-18","2026-04-29","2026-06-17","2026-07-29","2026-09-16","2026-11-04","2026-12-16",
])


def _third_friday(year: int, month: int) -> pd.Timestamp:
    fridays = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    fridays = fridays[(fridays.month == month) & (fridays.dayofweek == 4)]
    return fridays[2]


def _since_until(dates: pd.DatetimeIndex, events: pd.DatetimeIndex
                 ) -> tuple[pd.Series, pd.Series]:
    """Trading days since the last event / until the next, per date."""
    pos = pd.Series(range(len(dates)), index=dates)
    ev = pos.reindex(events.intersection(dates)).dropna().astype(int).values
    since, until = [], []
    for i in range(len(dates)):
        prev = ev[ev <= i]
        nxt = ev[ev >= i]
        since.append(i - prev[-1] if len(prev) else float("nan"))
        until.append(nxt[0] - i if len(nxt) else float("nan"))
    return (pd.Series(since, index=dates, dtype="float32"),
            pd.Series(until, index=dates, dtype="float32"))


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """-> DataFrame indexed by date, one column per Trexsim seasonality var."""
    dates = pd.DatetimeIndex(dates).sort_values()
    d = pd.DataFrame(index=dates)

    d["day_of_the_week"] = dates.dayofweek + 1          # 1=Mon .. 5=Fri
    d["month_of_the_year"] = dates.month
    d["quarter_of_year"] = dates.quarter

    # position within month / quarter / year
    for name, keys in (("month", [dates.year, dates.month]),
                       ("quarter", [dates.year, dates.quarter]),
                       ("year", [dates.year])):
        g = pd.Series(range(len(dates)), index=dates).groupby(keys)
        d[f"days_since_first_trading_day_of_{name}"] = (g.cumcount()).values
        d[f"days_until_last_trading_day_of_{name}"] = (
            g.transform("size").values - g.cumcount().values - 1
        )

    # monthly + quarterly options expiration (3rd Friday)
    months = sorted({(t.year, t.month) for t in dates})
    opex = pd.DatetimeIndex([_third_friday(y, m) for y, m in months])
    qopex = pd.DatetimeIndex([_third_friday(y, m) for y, m in months if m in (3, 6, 9, 12)])
    for tag, ev in (("monthly_options_expiration", opex),
                    ("quarterly_options_expiration", qopex)):
        s, u = _since_until(dates, ev)
        d[f"days_since_last_{tag}"] = s
        d[f"days_until_next_{tag}"] = u

    # trading holidays = weekday gaps in the calendar
    allbd = pd.date_range(dates[0], dates[-1], freq="B")
    hol = allbd.difference(dates)
    prev_hol, next_hol = [], []
    for t in dates:
        p = hol[hol < t]
        n = hol[hol > t]
        prev_hol.append((t - p[-1]).days if len(p) else float("nan"))
        next_hol.append((n[0] - t).days if len(n) else float("nan"))
    d["days_since_last_trading_holiday"] = pd.Series(prev_hol, index=dates, dtype="float32")
    d["days_until_next_trading_holiday"] = pd.Series(next_hol, index=dates, dtype="float32")

    # FOMC
    s, u = _since_until(dates, FOMC_DATES)
    d["days_since_last_fomc_meeting"] = s
    d["days_until_next_fomc_meeting"] = u

    return d.astype("float32")
