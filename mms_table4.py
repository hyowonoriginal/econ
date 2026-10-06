"""
Martinez, Meier & Sprenger (2023, JEEA) Table 4 with aggregate screening counts.
Heidhues-Strack (2021) estimator: delta = 1, naive beta.

Screening in period t pays -c + eta_t, eta = eps(1) - eps(0) ~ F (mean 0).
Naive perceived continuation value (time-consistent future selves), W = U + c:
    W(0) = 0,   W(r) = W(r-1) + E[(eta - W(r-1))^+]      (r = periods left)
Self t screens iff eta_t >= (1 - beta) c + beta W(T - t - 1):
    p_t = 1 - F((1 - beta) c + beta W(T - t - 1)),   p_T = 1.
AMONG_SCREEN_t is the number still unscreened after t, so
    logL = sum_{t<T} CNT_INDI_t log p_t + AMONG_SCREEN_t log(1 - p_t).

Never screened (MMS Online Appendix A.6.1): with participation rate q, NEVER = screened (1/q - 1)
people per year are added to AMONG_SCREEN in every period, including T. Never screening pays
ybar (HS penalty), so W(0) = ybar + c at T and
    p_t = 1 - F((1 - beta) c + beta W(T - t)),   t = 1..T.
ybar is fixed at YBARS or estimated ("ybar free"); c and ybar are in units of sqrt(m).

Usage: see main.py. run(data, participation) takes an Excel file.
Monthly columns: YYYYMM, CNT_INDI, YEAR, AMONG_SCREEN. Weekly: woy, CNT_INDI, AMONG_SCREEN (YEAR optional).
"""
import numpy as np
import pandas as pd
from scipy import optimize, special

# (distribution, shock variance in units of pi^2/3)
SPECS = [("logistic", 1), ("logistic", 5), ("logistic", 25),
         ("normal", 1), ("normal", 5), ("normal", 25)]
YBARS = [0, -2, -5, -8]  # fixed values of never screening


def load(data, participation=1.0):
    df = pd.read_excel(data)
    df["t"] = df["YYYYMM"] % 100 if "YYYYMM" in df else df["woy"].clip(lower=1)  # week 0 -> week 1
    if "YEAR" not in df:
        df["YEAR"] = "all"
    df = df.groupby(["YEAR", "t"], as_index=False).agg(CNT_INDI=("CNT_INDI", "sum"),
                                                       AMONG_SCREEN=("AMONG_SCREEN", "min"))
    df["r"] = df.groupby("YEAR")["t"].transform("max") - df["t"]  # periods left until T
    df["NEVER"] = df.groupby("YEAR")["CNT_INDI"].transform("sum") * (1 / participation - 1)
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


def fit(df, dist, m, ybar=None, starts=None):
    """ybar None: p_T = 1. A number: fixed ybar (x sqrt(m)). "free": estimated, from `starts`."""
    logsf, excess = shock(dist, m)
    mand, k = ybar is None, np.sqrt(m)  # theta holds a, c, ybar in units of k
    d = (df[df.r > 0] if mand else df).groupby("r", as_index=False)[["CNT_INDI", "AMONG_SCREEN", "NEVER"]].sum()
    n, s = d.CNT_INDI.values, d.AMONG_SCREEN.values + (0 if mand else d.NEVER.values)
    R = d.r.values - mand  # index of W for each period
    N = df.CNT_INDI.sum() + (0 if mand else df.groupby("YEAR").NEVER.first().sum())

    def W(w0):
        W = [w0]
        for _ in range(R.max()):
            W.append(W[-1] + excess(W[-1]))
        return np.array(W)[R]

    def cutoff(th):  # p_t = 1 - F(cutoff)
        if mand:  # theta = (beta, a / k), a = (1 - beta) c
            return k * th[1] + th[0] * W(0.0)
        b, c = th[0], k * th[1]
        return (1 - b) * c + b * W(k * (th[2] if ybar == "free" else ybar) + c)

    ll = lambda th: np.sum(n * logsf(x := cutoff(th)) + s * logsf(-x))
    if mand:  # start: logistic approximation, log-odds of waiting = a + beta w
        ok, w = (n > 0) & (s > 0), W(0.0) / k
        starts = [np.linalg.lstsq(np.c_[w, np.ones(len(w))][ok], np.log(s / n)[ok], rcond=None)[0]]
    elif starts is None:
        starts = [[b, c] for b in (0.3, 0.6, 1, 1.5, 2) for c in (0, 3, 8)]
    res = [optimize.minimize(lambda th: -ll(th) / N, x0, method="BFGS", options={"gtol": 1e-10}) for x0 in starts]
    th = min(res, key=lambda r: np.nan_to_num(r.fun, nan=np.inf)).x

    V = np.linalg.inv(-hessian(ll, th))
    se = np.sqrt(np.diag(V)) * np.r_[1, k, k][:len(th)]
    out = {"beta": th[0], "se(beta)": se[0], "delta": 1}
    if mand:
        b, a = th
        g = k * np.array([a, 1 - b]) / (1 - b) ** 2  # gradient of c = k a / (1 - beta)
        out |= {"c": k * a / (1 - b), "se(c)": np.sqrt(g @ V @ g)}
    else:
        free = ybar == "free"
        out |= {"c": k * th[1], "se(c)": se[1], "ybar": k * (th[2] if free else ybar),
                "se(ybar)": se[2] if free else np.nan}
    return out | {"Shock variance (x pi^2/3)": m, "Observations": df.t.nunique(), "Individuals": N,
                  "Never screened": 1 - df.CNT_INDI.sum() / N, "Log-likelihood": ll(th)}


def table4(df):
    cols = {}
    for j, (dist, m) in enumerate(SPECS, 1):
        res = {"p_T = 1": fit(df, dist, m)}
        if df.NEVER.sum() > 0:  # fixed ybar first; their estimates start the free fit
            fixed = {f"ybar = {y}": fit(df, dist, m, y) for y in YBARS}
            k = np.sqrt(m)
            res["ybar free"] = fit(df, dist, m, "free", [[r["beta"], r["c"] / k, r["ybar"] / k] for r in fixed.values()])
            res |= fixed
        cols[f"({j}) {dist}"] = pd.concat({v: pd.Series(r) for v, r in res.items()})
    return pd.DataFrame(cols)


def run(data, participation=0.52, exclude=2020):
    """Table 4 for pooled data and, as a robustness check, without the year `exclude`."""
    df = load(data, participation)
    groups = {"pooled": df} | ({f"excl. {exclude}": df[df.YEAR != exclude]} if exclude in set(df.YEAR) else {})
    return pd.concat({g: table4(d) for g, d in groups.items()}, names=["group", "model", ""])