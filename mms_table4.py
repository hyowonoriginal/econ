"""
Replicate Martinez, Meier & Sprenger (2023, JEEA) Table 4 with aggregate monthly data.

Setting (MMS Table 4 / Heidhues & Strack 2021, Lemma 1)
-------------------------------------------------------
- delta = 1, naive quasi-hyperbolic agent (beta_hat = 1).
- Payoff of completing the task in period t: y_t = -c + eta_t, eta_t iid with CDF F
  (eta = eps(1) - eps(0), mean 0). With delta = 1 the benefit b_i (and its timing k_i)
  drops out, so only beta and c are estimated.
- Everyone completes by the deadline T (p_T = 1).

Naive beliefs (perceived continuation value, exponential self):
    U_T = E[y] = -c,   U_t = E[max(y, U_{t+1})].
Writing W_t = U_t + c gives a recursion that does not depend on (beta, c):
    W_T = 0,   W_t = W_{t+1} + E[(eta - W_{t+1})^+].
Self t completes iff y_t >= beta * U_{t+1}, i.e. eta_t >= (1 - beta) c + beta W_{t+1}:
    p_t = 1 - F((1 - beta) c + beta W_{t+1}),  t < T;   p_T = 1.
With logistic scale 1, W_{t+1} = ln(T - t), so
    p_t = 1 / (1 + exp((1 - beta) c) * (T - t)^beta),
which is MMS eq. (3) with delta = 1.

Data
----
Long CSV with columns `group, month, n`: number of people screened in each month,
by group (e.g. year). Months run 1..T (T = 12 by default).
"""

import argparse

import numpy as np
import pandas as pd
from scipy import optimize, special

LOGISTIC_VAR = np.pi ** 2 / 3  # variance of standard logistic

# Table 4 columns: (distribution, variance multiplier)
TABLE4_SPECS = [
    ("logistic", 1), ("logistic", 5), ("logistic", 25),
    ("normal", 1), ("normal", 5), ("normal", 25),
]


# ---------------------------------------------------------------------------
# Shock distribution
# ---------------------------------------------------------------------------
class Shock:
    """eta = eps(1) - eps(0), mean 0, variance var_mult * pi^2 / 3."""

    def __init__(self, dist, var_mult=1.0):
        self.dist = dist
        self.var = var_mult * LOGISTIC_VAR
        if dist == "logistic":
            self.scale = np.sqrt(var_mult)  # logistic scale s has variance s^2 pi^2 / 3
        elif dist == "normal":
            self.scale = np.sqrt(self.var)  # standard deviation
        else:
            raise ValueError(f"unknown distribution: {dist}")

    def sf(self, x):
        """Pr(eta >= x)."""
        z = x / self.scale
        return special.expit(-z) if self.dist == "logistic" else special.ndtr(-z)

    def logcdf(self, x):
        z = x / self.scale
        return special.log_expit(z) if self.dist == "logistic" else special.log_ndtr(z)

    def logsf(self, x):
        return self.logcdf(-x)  # both distributions are symmetric

    def expected_excess(self, a):
        """E[(eta - a)^+]."""
        s, z = self.scale, a / self.scale
        if self.dist == "logistic":
            return s * np.logaddexp(0.0, -z)
        return s * (np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) - z * special.ndtr(-z))


def naive_W(shock, T):
    """W_1..W_T from W_T = 0, W_t = W_{t+1} + E[(eta - W_{t+1})^+]. Index 0 = month 1."""
    W = np.zeros(T)
    for t in range(T - 2, -1, -1):
        W[t] = W[t + 1] + shock.expected_excess(W[t + 1])
    return W


def thresholds(beta, c, W):
    """Cutoff for eta in months 1..T-1: (1 - beta) c + beta W_{t+1}."""
    return (1.0 - beta) * c + beta * W[1:]


def hazard(beta, c, shock, T):
    """Conditional completion probability p_1..p_T (p_T = 1)."""
    W = naive_W(shock, T)
    return np.append(shock.sf(thresholds(beta, c, W)), 1.0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {"std_yyyy": "group", "screen_month": "month", "woy": "month", "n_screened": "n"}


def load_counts(data, T=None):
    """
    Long data (group, month, n) -> DataFrame indexed by group, columns 1..T.

    `month` is the period index (month 1..12, or week of year `woy` 1..52).
    T (deadline period) defaults to the last period in the data.

    `data` is a file path (.csv, tab-separated .txt/.tsv, or .xlsx) or a DataFrame.
    Columns std_yyyy / screen_month / n_screened are accepted as group / month / n.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif str(data).endswith((".xlsx", ".xls")):
        df = pd.read_excel(data)
    else:
        df = pd.read_csv(data, sep=None, engine="python")  # detects comma or tab
    df = df.rename(columns=COLUMN_ALIASES)
    if "group" not in df.columns:
        df["group"] = "all"
    wide = df.pivot_table(index="group", columns="month", values="n", aggfunc="sum")
    T = int(df["month"].max()) if T is None else T
    wide = wide.reindex(columns=range(1, T + 1), fill_value=0).fillna(0)

    if "N_eligible" in df.columns:
        # p_T = 1 assumes everyone in the sample is screened by month T.
        eligible = df.groupby("group")["N_eligible"].first()
        gap = eligible - wide.sum(axis=1)
        for g, k in gap[gap > 0].items():
            print(f"Warning: group {g} has {int(k):,} eligible people never screened; "
                  "they are dropped (p_T = 1 uses screened people only).")
    return wide


def risk_sets(counts):
    """counts: array (G, T) -> at-risk array (G, T): people not yet screened at start of t."""
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    total = counts.sum(axis=1, keepdims=True)
    done_before = np.cumsum(counts, axis=1) - counts
    return total - done_before


def empirical_hazard(counts):
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    R = risk_sets(counts).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return counts.sum(axis=0) / R


# ---------------------------------------------------------------------------
# Likelihood and estimation
# ---------------------------------------------------------------------------
class Table4Model:
    """
    Pooled MLE of (beta, c) with delta = 1. Groups share parameters, so the
    pooled log-likelihood is the sum over groups (same as MMS pooling 2005-2007).
    """

    def __init__(self, counts, dist="logistic", var_mult=1.0, beta_max=1.0):
        self.beta_max = beta_max
        counts = np.atleast_2d(np.asarray(counts, dtype=float))
        self.T = counts.shape[1]
        self.shock = Shock(dist, var_mult)
        self.W = naive_W(self.shock, self.T)
        R = risk_sets(counts)
        # Month T contributes log(1) = 0 because p_T = 1.
        self.n = counts[:, :-1].sum(axis=0)
        self.stay = (R[:, :-1] - counts[:, :-1]).sum(axis=0)
        self.n_people = counts.sum()

    def loglik(self, theta):
        beta, c = theta
        x = thresholds(beta, c, self.W)
        return np.sum(self.n * self.shock.logsf(x) + self.stay * self.shock.logcdf(x))

    def start_values(self):
        """
        OLS on the logistic closed form: log-odds of waiting = (1-beta) c + beta W_{t+1}.
        For other distributions, approximate by a logistic with the same variance.
        """
        n, stay = self.n, self.stay
        ok = (n > 0) & (stay > 0)
        s_eq = np.sqrt(self.shock.var / LOGISTIC_VAR)
        y = s_eq * np.log(stay[ok] / n[ok])
        X = np.column_stack([np.ones(ok.sum()), self.W[1:][ok]])
        a, b = np.linalg.lstsq(X, y, rcond=None)[0]
        beta = float(np.clip(b, 0.05, 1.5))
        c = a / (1 - beta) if abs(1 - beta) > 1e-3 else 0.0
        return np.array([beta, c])

    def loglik_a(self, beta, a):
        """Log-likelihood in terms of beta and the intercept a = (1 - beta) c."""
        x = a + beta * self.W[1:]
        return np.sum(self.n * self.shock.logsf(x) + self.stay * self.shock.logcdf(x))

    def fit(self, start=None):
        """
        MLE with beta restricted to (0, beta_max].

        Estimated as (beta, a) with a = (1 - beta) c, then c = a / (1 - beta).
        This keeps the problem well behaved near beta = 1: if beta hits the bound
        beta_max = 1, a is still estimated but c = a / 0 diverges (reported as NaN).
        """
        bmax = self.beta_max
        b0, c0 = self.start_values() if start is None else np.asarray(start, float)
        negll = lambda th: -self.loglik_a(*th) / self.n_people  # scaled for the optimizer
        bounds = [(1e-3, bmax), (None, None)]
        best = None
        for b, a in [(b0, (1 - b0) * c0), (0.8, 0.2), (0.5, 2.5)]:
            x0 = np.array([np.clip(b, 0.01, bmax - 0.01), a])
            res = optimize.minimize(negll, x0, method="Nelder-Mead", bounds=bounds,
                                    options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000})
            res = optimize.minimize(negll, res.x, method="L-BFGS-B", bounds=bounds,
                                    options={"ftol": 1e-15, "gtol": 1e-10})
            if best is None or res.fun < best.fun:
                best = res
        beta, a = best.x
        self.at_bound = bmax - beta < 1e-6
        full_negll = lambda th: -self.loglik_a(*th)
        if self.at_bound:
            # beta on the upper bound: no standard error for beta, only for a.
            beta = bmax
            h = numerical_hessian(lambda x: full_negll([beta, x[0]]), np.array([a]))
            se_b, se_a, cov_ba = np.nan, float(np.sqrt(1.0 / h[0, 0])), 0.0
        else:
            cov = np.linalg.inv(numerical_hessian(full_negll, np.array([beta, a])))
            se_b, se_a, cov_ba = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), cov[0, 1]
        self.ll = self.loglik_a(beta, a)
        self.a, self.se_a = a, se_a
        if abs(1 - beta) < 1e-6:
            c, se_c = np.copysign(np.inf, a), np.nan  # c = a / 0 diverges: not identified
        else:
            c = a / (1 - beta)
            # delta method: dc/dbeta = a / (1-beta)^2, dc/da = 1 / (1-beta)
            g = np.array([a / (1 - beta) ** 2, 1 / (1 - beta)])
            V = np.array([[0.0 if np.isnan(se_b) else se_b ** 2, cov_ba], [cov_ba, se_a ** 2]])
            se_c = float(np.sqrt(g @ V @ g))
        self.theta = np.array([beta, c])
        self.se = np.array([se_b, se_c])
        return self

    def predicted_hazard(self):
        beta = self.theta[0]
        return np.append(self.shock.sf(self.a + beta * self.W[1:]), 1.0)


def numerical_hessian(f, x, rel_step=1e-4):
    x = np.asarray(x, float)
    k = len(x)
    h = rel_step * np.maximum(np.abs(x), 1.0)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * h[i], np.eye(k)[j] * h[j]
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return (H + H.T) / 2


# ---------------------------------------------------------------------------
# Table 4
# ---------------------------------------------------------------------------
def fmt(x, paren=False):
    if np.isnan(x):
        return "-"
    if np.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"({x:.3f})" if paren else f"{x:.3f}"


def table4(counts, specs=TABLE4_SPECS, beta_max=1.0):
    cols = {}
    fits = {}
    for k, (dist, m) in enumerate(specs, start=1):
        mod = Table4Model(counts, dist, m, beta_max).fit()
        fits[k] = mod
        var_label = "pi^2/3" if m == 1 else f"{m} x pi^2/3"
        cols[f"({k})"] = {
            "Shock distribution": dist.capitalize(),
            "beta": fmt(mod.theta[0]),
            "  (se)": fmt(mod.se[0], paren=True),
            "delta": "1",
            "c": fmt(mod.theta[1]),
            "  (se) ": fmt(mod.se[1], paren=True),
            "(1-beta) c": fmt(mod.a),
            "  (se)  ": fmt(mod.se_a, paren=True),
            "Shock variance": var_label,
            "Observations (periods)": f"{mod.T}",
            "Individuals": f"{int(mod.n_people):,}",
            "Log-likelihood": f"{mod.ll:,.2f}",
        }
    return pd.DataFrame(cols), fits


def simulate_counts(beta, c, N=100_000, T=12, dist="logistic", var_mult=1.0, groups=3, seed=0):
    """Draw monthly counts from the model (for checking the estimator)."""
    rng = np.random.default_rng(seed)
    p = hazard(beta, c, Shock(dist, var_mult), T)
    out = np.zeros((groups, T), dtype=int)
    for g in range(groups):
        left = N
        for t in range(T):
            out[g, t] = rng.binomial(left, p[t])
            left -= out[g, t]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV with columns group, month, n")
    ap.add_argument("--T", type=int, help="deadline period (default: last period in data; 12 for simulation)")
    ap.add_argument("--beta-max", type=float, default=1.0, help="upper bound on beta (default 1)")
    ap.add_argument("--by-group", action="store_true", help="also estimate each group separately")
    ap.add_argument("--out", help="save Table 4 as CSV")
    args = ap.parse_args()

    if args.data:
        wide = load_counts(args.data, args.T)
    else:
        print("No --data given: running on simulated data (beta=0.75, c=2.5, logistic).\n")
        args.T = args.T or 12
        sim = simulate_counts(0.75, 2.5, T=args.T)
        wide = pd.DataFrame(sim, index=[f"sim{g}" for g in range(len(sim))],
                            columns=range(1, args.T + 1))

    print("Monthly counts:")
    print(wide.astype(int).to_string(), "\n")
    print("Empirical conditional hazard (pooled):")
    print(pd.Series(empirical_hazard(wide.values), index=wide.columns).round(4).to_string(), "\n")

    tab, fits = table4(wide.values, beta_max=args.beta_max)
    print("Table 4 (pooled):")
    print(tab.to_string())
    if args.out:
        tab.to_csv(args.out)

    if args.by_group:
        for g, row in wide.iterrows():
            print(f"\nTable 4 — group {g}:")
            print(table4(row.values[None, :], beta_max=args.beta_max)[0].to_string())


if __name__ == "__main__":
    main()
