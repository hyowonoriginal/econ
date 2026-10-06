"""
Martinez, Meier & Sprenger (2023, JEEA) Table 4 with aggregate weekly screening counts.
Heidhues-Strack (2021) estimator: delta = 1, naive beta, participation rate q < 1.

Screening in week t pays -c + eta_t, eta = eps(1) - eps(0) ~ F (mean 0); never screening pays ybar.
Naive perceived continuation value (time-consistent future selves), W = U + c:
    W(0) = ybar + c,   W(r) = W(r-1) + E[(eta - W(r-1))^+]      (r = weeks left)
Self t screens iff eta_t >= (1 - beta) c + beta W(T - t):
    p_t = 1 - F((1 - beta) c + beta W(T - t)).
NEVER = screened (1/q - 1) people per year never screen, so
    logL = sum_t CNT_INDI_t log p_t + (AMONG_SCREEN_t + NEVER) log(1 - p_t).
Models: "p_T = 1" (MMS Table 4: no never-screened, everyone screened by T, last week dropped,
W(0) = 0 at T - 1), "MMS A.6.1" (c = C_FIX, ybar set so the last week's hazard is matched, only beta estimated),
"ybar free" (beta, c, ybar estimated), "ybar = y" (ybar fixed at YBARS).
c, ybar and C_FIX are in units of sqrt(m), so columns with the same distribution give the same beta.

Usage: see main.py. run(data) takes an Excel file with columns woy, CNT_INDI, AMONG_SCREEN (YEAR optional).
"""
import numpy as np
import pandas as pd
from scipy import optimize, special

# (distribution, shock variance in units of pi^2/3)
SPECS = [("logistic", 1), ("logistic", 5), ("logistic", 25),
         ("normal", 1), ("normal", 5), ("normal", 25)]
YBARS = [0, -2, -5, -8]  # fixed values of never screening
C_FIX = 1.34  # cost for MMS A.6.1


def load(data, q=0.52):
    df = pd.read_excel(data)
    df["t"] = df["woy"].clip(lower=1)  # week 0 -> week 1
    if "YEAR" not in df:
        df["YEAR"] = "all"
    df = df.groupby(["YEAR", "t"], as_index=False).agg(CNT_INDI=("CNT_INDI", "sum"),
                                                       AMONG_SCREEN=("AMONG_SCREEN", "min"))
    df["r"] = df.groupby("YEAR")["t"].transform("max") - df["t"]  # weeks left until T
    df["NEVER"] = df.groupby("YEAR")["CNT_INDI"].transform("sum") * (1 / q - 1)
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


def fit(df, dist, m, ybar, starts=None):
    """ybar: None (p_T = 1), a number (fixed), "free" (estimated from `starts`) or "mms" (MMS A.6.1)."""
    logsf, excess = shock(dist, m)
    k = np.sqrt(m)  # theta = (beta, c / k, ybar / k), as many as are estimated
    mand = ybar is None
    d = (df[df.r > 0] if mand else df).groupby("r", as_index=False)[["CNT_INDI", "AMONG_SCREEN", "NEVER"]].sum()
    n, s, r = d.CNT_INDI.values, (d.AMONG_SCREEN + (0 if mand else d.NEVER)).values, d.r.values - mand
    N = df.CNT_INDI.sum() + (0 if mand else df.groupby("YEAR").NEVER.first().sum())
    xT = optimize.brentq(lambda x: logsf(x) - np.log(n[0] / (n[0] + s[0])), -1e4, 1e4)  # last-week cutoff

    def W(w0):
        W = [w0]
        for _ in range(r.max()):
            W.append(W[-1] + excess(W[-1]))
        return np.array(W)[r]

    def params(th):  # (beta, c, ybar); MMS: c + beta ybar = xT; p_T = 1: ybar = -c gives W(0) = 0
        b, c = th[0], k * (C_FIX if ybar == "mms" else th[1])
        if mand:
            return b, c, -c
        return b, c, (xT - c) / b if ybar == "mms" else k * (th[2] if ybar == "free" else ybar)

    def cutoff(th):  # p_t = 1 - F(cutoff), log(1 - p_t) = logsf(-cutoff)
        b, c, y = params(th)
        return (1 - b) * c + b * W(y + c)

    ll = lambda th: np.sum(n * logsf(x := cutoff(th)) + s * logsf(-x))

    starts = starts or ([[b] for b in (0.3, 0.6, 1, 1.5, 2)] if ybar == "mms" else
                        [[b, c] for b in (0.3, 0.6, 1, 1.5, 2) for c in (0, 3, 8)])
    res = [optimize.minimize(lambda th: -ll(th) / N, x0, method="BFGS", options={"gtol": 1e-10}) for x0 in starts]
    th = min(res, key=lambda o: np.nan_to_num(o.fun, nan=np.inf)).x
    se = np.sqrt(np.diag(np.linalg.pinv(-hessian(ll, th)))) * np.r_[1, k, k][:len(th)]
    se = np.r_[se, [np.nan] * (3 - len(th))]
    b, c, y = params(th)
    return {"beta": b, "se(beta)": se[0], "delta": 1, "c": c, "se(c)": se[1], "ybar": np.nan if mand else y, "se(ybar)": se[2],
            "Shock variance (x pi^2/3)": m, "Observations": df.t.nunique(), "Individuals": N,
            "Never screened": 1 - df.CNT_INDI.sum() / N,
            "Never screened (model)": 0.0 if mand else np.exp(logsf(-cutoff(th)).sum()),  # prod_t (1 - p_t)
            "Log-likelihood": ll(th)}


def table4(df):
    cols = {}
    for j, (dist, m) in enumerate(SPECS, 1):
        k = np.sqrt(m)
        res = {"p_T = 1": fit(df, dist, m, None), "MMS A.6.1": fit(df, dist, m, "mms")}
        res |= {f"ybar = {y}": fit(df, dist, m, y) for y in YBARS}
        starts = [[o["beta"], o["c"] / k, o["ybar"] / k] for v, o in res.items() if v != "p_T = 1"]  # for free fit
        res["ybar free"] = fit(df, dist, m, "free", starts)
        cols[f"({j}) {dist}"] = pd.concat({v: pd.Series(o) for v, o in res.items()})
    return pd.DataFrame(cols)


def run(data, q=0.52, exclude=2020):
    """Table 4 for pooled data and, as a robustness check, without the year `exclude`."""
    df = load(data, q)
    groups = {"pooled": df} | ({f"excl. {exclude}": df[df.YEAR != exclude]} if exclude in set(df.YEAR) else {})
    return pd.concat({g: table4(d) for g, d in groups.items()}, names=["group", "model", ""])
