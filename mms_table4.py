"""
Martinez, Meier & Sprenger (2023, JEEA) Table 4 with aggregate screening counts.
Heidhues-Strack (2021) estimator: delta = 1, naive beta, everyone screened by T.

Screening in period t pays -c + eta_t, eta = eps(1) - eps(0) ~ F (mean 0).
Naive perceived continuation value (time-consistent future selves), W = U + c:
    W(0) = 0,   W(r) = W(r-1) + E[(eta - W(r-1))^+]      (r = periods left)
Self t screens iff eta_t >= (1 - beta) c + beta W(T - t - 1):
    p_t = 1 - F((1 - beta) c + beta W(T - t - 1)),   p_T = 1.
AMONG_SCREEN_t is the number still unscreened after t, so
    logL = sum_{t<T} CNT_INDI_t log p_t + AMONG_SCREEN_t log(1 - p_t).

Usage: python mms_table4.py monthly.csv weekly.csv
Monthly columns: YYYYMM, CNT_INDI, YEAR, AMONG_SCREEN. Weekly: woy, CNT_INDI, AMONG_SCREEN (YEAR optional).
"""
import sys

import numpy as np
import pandas as pd
from scipy import optimize, special

# (distribution, shock variance in units of pi^2/3)
SPECS = [("logistic", 1), ("logistic", 5), ("logistic", 25),
         ("normal", 1), ("normal", 5), ("normal", 25)]


def load(path):
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path, sep=None, engine="python")
    df["t"] = df["YYYYMM"] % 100 if "YYYYMM" in df else df["woy"].clip(lower=1)  # week 0 -> week 1
    if "YEAR" not in df:
        df["YEAR"] = "all"
    df = df.groupby(["YEAR", "t"], as_index=False).agg(CNT_INDI=("CNT_INDI", "sum"),
                                                       AMONG_SCREEN=("AMONG_SCREEN", "min"))
    df["r"] = df.groupby("YEAR")["t"].transform("max") - df["t"]  # periods left until T
    return df


def shock(dist, m):
    """log(1 - F) and E[(eta - a)^+] for eta with variance m * pi^2 / 3."""
    if dist == "logistic":
        s = np.sqrt(m)
        return lambda x: special.log_expit(-x / s), lambda a: s * np.logaddexp(0, -a / s)
    s = np.sqrt(m * np.pi ** 2 / 3)
    return (lambda x: special.log_ndtr(-x / s),
            lambda a: s * (np.exp(-(a / s) ** 2 / 2) / np.sqrt(2 * np.pi) - a / s * special.ndtr(-a / s)))


def hessian(f, x, h=1e-4):
    E = np.eye(len(x)) * h
    return np.array([[(f(x + i + j) - f(x + i - j) - f(x - i + j) + f(x - i - j)) / (4 * h * h)
                      for j in E] for i in E])


def fit(df, dist, m):
    logsf, excess = shock(dist, m)
    d = df[df.r > 0].groupby("r", as_index=False)[["CNT_INDI", "AMONG_SCREEN"]].sum()  # pool years
    W = [0.0]
    for _ in range(d.r.max() - 1):
        W.append(W[-1] + excess(W[-1]))
    w, n, s = np.array(W)[d.r - 1], d.CNT_INDI.values, d.AMONG_SCREEN.values

    # theta = (beta, a), a = (1 - beta) c; p_t = 1 - F(a + beta w)
    ll = lambda th: np.sum(n * logsf(th[1] + th[0] * w) + s * logsf(-th[1] - th[0] * w))
    ok = (n > 0) & (s > 0)  # start: logistic approximation, log-odds of waiting = a + beta w
    th0 = np.linalg.lstsq(np.c_[w, np.ones(len(w))][ok], np.sqrt(m) * np.log(s / n)[ok], rcond=None)[0]
    N = df.CNT_INDI.sum()
    th = optimize.minimize(lambda th: -ll(th) / N, th0, method="BFGS", options={"gtol": 1e-10}).x

    V = np.linalg.inv(-hessian(ll, th))
    beta, a = th
    g = np.array([a, 1 - beta]) / (1 - beta) ** 2  # gradient of c = a / (1 - beta)
    return {"beta": beta, "se(beta)": np.sqrt(V[0, 0]), "delta": 1,
            "c": a / (1 - beta), "se(c)": np.sqrt(g @ V @ g),
            "Shock variance (x pi^2/3)": m, "Observations": df.t.nunique(),
            "Individuals": N, "Log-likelihood": ll(th)}


def table4(df):
    return pd.DataFrame({f"({k}) {dist}": fit(df, dist, m) for k, (dist, m) in enumerate(SPECS, 1)})


if __name__ == "__main__":
    pd.set_option("display.float_format", "{:,.3f}".format, "display.width", 250, "display.max_columns", None)
    for path in sys.argv[1:]:
        df = load(path)
        groups = {"pooled": df} | ({y: g for y, g in df.groupby("YEAR")} if df.YEAR.nunique() > 1 else {})
        out = pd.concat({g: table4(d) for g, d in groups.items()}, names=["group", ""])
        print(f"\n=== {path} ===\n{out}")
        out.to_csv(path.rsplit(".", 1)[0] + "_table4.csv")
