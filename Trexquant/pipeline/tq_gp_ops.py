"""GP primitive set — every operator maps 1:1 to a Trexsim builtin.

GOVERNING RULE: if the GP can build an expression Trexsim cannot evaluate, the
discovered alpha is unsubmittable and the search time was wasted. So the
primitive set is a strict subset of Trexsim's 32 builtins, and `to_trexsim()`
converts any discovered formula into paste-ready platform syntax.

Compared with your QIP set (add sub mul div neg rank diff5 ts_max5 ts_min5
ts_ma5 — 10 primitives, all hard-wired to a 5-day window), this adds:

  * VARIABLE WINDOWS {5, 10, 20, 60} via strongly-typed GP, instead of only 5.
    Alpha101 uses windows from 2 to 250; a 5-only search cannot express
    momentum or slow reversal at all.
  * ts_corr  — the single most-used operator in Alpha101 (44 occurrences
    across alpha101 + gtja191). Price-volume correlation is that library's
    core idiom and your GP had no way to express it.
  * ts_delta, ts_std, ts_zscore, ts_sum, ts_median, ts_skew, ts_decay
  * cs_zscore alongside cs_rank
  * at_signlog / at_signsqrt for tail compression
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

#: Rolling windows offered to the GP. Kept small and log-spaced — more values
#: multiply the search space without adding expressive power.
WINDOWS = (5, 10, 20, 60)

_EPS = 1e-12


def _mp(w: int) -> int:
    return max(2, w // 2)


def _safe(x):
    return x.replace(0, np.nan) if hasattr(x, "replace") else x


def w_id(w):
    """Identity on a window.

    Required by DEAP's strongly-typed GP: `genFull` must be able to place a
    PRIMITIVE of the required type at any depth below the target height. The
    window type otherwise has terminals only, so generation dies with
    "tried to add a primitive of type Win, but there is none available".
    Module-level (not a lambda) so it survives pickling for multiprocessing.
    `to_trexsim` unwraps it.
    """
    return w


# ------------------------------------------------------------ arithmetic --
def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def div(a, b): return a / _safe(b)
def neg(a): return -a


# ---------------------------------------------------------- element-wise --
def signlog(a):
    return np.sign(a) * np.log1p(np.abs(a))


def signsqrt(a):
    return np.sign(a) * np.sqrt(np.abs(a) + 1.0)


# -------------------------------------------------------- cross-sectional --
def cs_rank(a):
    """Percentile rank in [0,1]. NOTE Trexsim's cs_rank returns [1,2] —
    `to_trexsim()` emits `(cs_rank(x) - 1)` to preserve these semantics."""
    return a.rank(axis=1, pct=True)


def cs_zscore(a):
    return a.sub(a.mean(axis=1), axis=0).div(_safe(a.std(axis=1)), axis=0)


# ------------------------------------------------------------ time-series --
def ts_mean(a, w): return a.rolling(w, min_periods=_mp(w)).mean()
def ts_std(a, w): return a.rolling(w, min_periods=_mp(w)).std()
def ts_sum(a, w): return a.rolling(w, min_periods=_mp(w)).sum()
def ts_median(a, w): return a.rolling(w, min_periods=_mp(w)).median()
def ts_max(a, w): return a.rolling(w, min_periods=_mp(w)).max()
def ts_min(a, w): return a.rolling(w, min_periods=_mp(w)).min()
def ts_skew(a, w): return a.rolling(w, min_periods=_mp(w)).skew()
def ts_delay(a, w): return a.shift(w)
def ts_delta(a, w): return a - a.shift(w)


def ts_zscore(a, w):
    m = a.rolling(w, min_periods=_mp(w)).mean()
    s = a.rolling(w, min_periods=_mp(w)).std()
    return (a - m) / _safe(s)


def ts_decay(a, w):
    """Exponentially weighted mean -> Trexsim ts_mean_exp. The turnover lever:
    'hold with exponentially decaying size'."""
    return a.ewm(span=w, min_periods=_mp(w)).mean()


def ts_corr(a, b, w):
    """Rolling per-stock correlation -> Trexsim ts_corr_binary."""
    return a.rolling(w, min_periods=_mp(w)).corr(b)


# ------------------------------------------------------------ translation --
_UNARY = {
    "neg": ("-({0})", None),
    "signlog": ("at_signlog({0})", None),
    "signsqrt": ("at_signsqrt({0})", None),
    "cs_rank": ("(cs_rank({0}) - 1)", None),      # [1,2] -> [0,1]
    "cs_zscore": ("cs_zscore({0})", None),
}
_BINARY_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_TS = {
    "ts_mean": "ts_mean", "ts_std": "ts_std", "ts_sum": "ts_sum",
    "ts_median": "ts_median", "ts_max": "ts_max", "ts_min": "ts_min",
    "ts_skew": "ts_skew", "ts_delay": "ts_delay", "ts_zscore": "ts_zscore",
}


def to_trexsim(formula: str, terminals: dict[str, str] | None = None) -> str:
    """DEAP prefix string -> paste-ready Trexsim expression.

    Handles three things that would otherwise silently change the alpha:
      * infix arithmetic (`add(a,b)` -> `(a + b)`)
      * `cs_rank` rescaling from [1,2] to [0,1]
      * `div` -> `/ at_zero2nan(...)` so a zero denominator gives NaN, not inf
    """
    terminals = terminals or {}

    def walk(node) -> str:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Name):
            # window terminals are named w5/w10/w20/w60 in the GP -> emit the int
            if len(node.id) > 1 and node.id[0] == "w" and node.id[1:].isdigit():
                return node.id[1:]
            return terminals.get(node.id, node.id)
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"-{walk(node.operand)}"
        if isinstance(node, ast.Call):
            fn = node.func.id
            args = [walk(a) for a in node.args]
            if fn in _BINARY_OP:
                if fn == "div":
                    return f"({args[0]} / at_zero2nan({args[1]}))"
                return f"({args[0]} {_BINARY_OP[fn]} {args[1]})"
            if fn in _UNARY:
                return _UNARY[fn][0].format(args[0])
            if fn in _TS:
                return f"{_TS[fn]}({args[0]}, {args[1]})"
            if fn == "ts_delta":
                return f"({args[0]} - ts_delay({args[0]}, {args[1]}))"
            if fn == "ts_decay":
                return f"ts_mean_exp({args[0]}, {args[1]}, 0.5)"
            if fn == "ts_corr":
                return f"ts_corr_binary({args[0]}, {args[1]}, {args[2]})"
            if fn == "w_id":
                return args[0]           # unwrap the DEAP typing helper
            return f"{fn}({', '.join(args)})"
        raise ValueError(f"cannot translate node {ast.dump(node)}")

    return walk(ast.parse(formula.replace("\n", "").replace("\t", ""),
                          mode="eval"))


def recompute(formula: str, terminals: dict[str, pd.DataFrame]):
    """Re-evaluate a discovered formula against a DIFFERENT window's terminals.

    Needed for the held-out check: the training alpha cannot simply be
    reindexed onto the validation dates, because every rolling operator has to
    be recomputed from that window's own history.
    """
    env = {
        "add": add, "sub": sub, "mul": mul, "div": div, "neg": neg,
        "signlog": signlog, "signsqrt": signsqrt,
        "cs_rank": cs_rank, "cs_zscore": cs_zscore,
        "ts_mean": ts_mean, "ts_std": ts_std, "ts_sum": ts_sum,
        "ts_median": ts_median, "ts_max": ts_max, "ts_min": ts_min,
        "ts_skew": ts_skew, "ts_delay": ts_delay, "ts_delta": ts_delta,
        "ts_zscore": ts_zscore, "ts_decay": ts_decay, "ts_corr": ts_corr,
        "w_id": w_id,
    }
    env.update({f"w{w}": w for w in WINDOWS})
    env.update(terminals)
    return eval(formula.replace("\n", "").replace("\t", ""),  # noqa: S307
                {"__builtins__": {}}, env)


#: How each GP terminal is written in Trexsim. `to_trexsim` substitutes these
#: so the output is directly paste-able into the platform editor.
TERMINAL_TREXSIM = {
    "c_ma20": "(close / at_zero2nan(ts_mean(close, 20)))",
    "c_ma60": "(close / at_zero2nan(ts_mean(close, 60)))",
    "hl": "(high / at_zero2nan(low))",
    "co": "((close - open) / at_zero2nan(open))",
    "gap": "((open - ts_delay(close, 1)) / at_zero2nan(ts_delay(close, 1)))",
    "stoch": ("((close - ts_min(low, 20)) / "
              "at_zero2nan(ts_max(high, 20) - ts_min(low, 20)))"),
    "ret1": "ret1",
    "ret5": "ret5",
    "ret20": "ret20",
    "v_ma20": "(volume / at_zero2nan(ts_mean(volume, 20)))",
    "dv_adv": ("((close * volume) / at_zero2nan("
               "ts_mean(close * volume, 60)))"),
    "vol20": "ts_std(ret1, 20)",
    "size": "(cs_rank(ts_mean(close * volume, 60)) - 1)",
    "beta60": "ts_corr_binary(ret1, ret1_spx, 60)",
}
