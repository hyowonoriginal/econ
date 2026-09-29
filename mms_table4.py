"""
Replicate Martinez, Meier & Sprenger (2023, JEEA) Table 4 with aggregate monthly data.

Setting (MMS Table 4 / Heidhues & Strack 2021, Lemma 1)
-------------------------------------------------------
- delta = 1, naive quasi-hyperbolic agent (beta_hat = 1).
- Payoff of completing the task in period t: y_t = -c + eta_t, eta_t iid with CDF F
  (eta = eps(1) - eps(0), mean 0). With delta = 1 the benefit b_i (and its timing k_i)
  drops out, so only beta and c are estimated.
- Default (MMS Table 4): everyone completes by the deadline T (p_T = 1).
- Optional (MMS Online Appendix A.6.1 / HS penalty): with the number of eligible
  people given, some never complete. Not completing by T gives utility y_{T+1}
  (HS's penalty), either estimated as a third parameter or fixed by assumption
  (y_{T+1} = 0: an optional task, HS Section I).

Naive beliefs (perceived continuation value, exponential self):
    U_T = E[y] = -c,   U_t = E[max(y, U_{t+1})].
Writing W_t = U_t + c gives a recursion that does not depend on (beta, c):
    W_T = 0,   W_t = W_{t+1} + E[(eta - W_{t+1})^+].
Self t completes iff y_t >= beta * U_{t+1}, i.e. eta_t >= (1 - beta) c + beta W_{t+1}:
    p_t = 1 - F((1 - beta) c + beta W_{t+1}),  t < T;   p_T = 1.
With logistic scale 1, W_{t+1} = ln(T - t), so
    p_t = 1 / (1 + exp((1 - beta) c) * (T - t)^beta),
which is MMS eq. (3) with delta = 1.

With a finite penalty the same recursion starts one period later from
    W_{T+1} = d := y_{T+1} + c,
and p_T = 1 - F((1 - beta) c + beta d) < 1. The mandatory case is d = -inf.

Data
----
Long CSV with columns `group, month, n`: number of people screened in each period,
by group (e.g. year). Periods are months (1..12) or weeks (`woy`, may start at 0).
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


def naive_W(shock, T, d=None):
    """
    Naive continuation values W_t = U_t + c, index 0 = period 1.

    Mandatory task (d None): W_1..W_T with W_T = 0.
    Finite penalty: W_1..W_{T+1} with W_{T+1} = d = y_{T+1} + c.
    Recursion: W_t = W_{t+1} + E[(eta - W_{t+1})^+].
    """
    if d is None:
        W = np.zeros(T)
    else:
        W = np.empty(T + 1)
        W[T] = d
    for t in range(len(W) - 2, -1, -1):
        W[t] = W[t + 1] + shock.expected_excess(W[t + 1])
    return W


def thresholds(beta, c, W):
    """Cutoff for eta in months 1..T-1: (1 - beta) c + beta W_{t+1}."""
    return (1.0 - beta) * c + beta * W[1:]


def hazard(beta, c, shock, T, penalty=None):
    """Conditional completion probability p_1..p_T (p_T = 1 unless a finite penalty y_{T+1})."""
    if penalty is None:
        W = naive_W(shock, T)
        return np.append(shock.sf(thresholds(beta, c, W)), 1.0)
    return shock.sf(thresholds(beta, c, naive_W(shock, T, penalty + c)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {"std_yyyy": "group", "screen_month": "month", "woy": "month", "n_screened": "n"}


def load_counts(data, T=None):
    """
    Long data (group, month, n) -> DataFrame indexed by group, columns 1..T.

    `month` is the period index (month 1..12, or week of year `woy` 0..52).
    Periods run from the first period in the data to T (default: the last one).

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
    first = int(df["month"].min())
    T = int(df["month"].max()) if T is None else T
    wide = wide.reindex(columns=range(first, T + 1), fill_value=0).fillna(0)

    if "N_eligible" in df.columns:
        # p_T = 1 assumes everyone in the sample is screened by month T.
        eligible = df.groupby("group")["N_eligible"].first()
        gap = eligible - wide.sum(axis=1)
        for g, k in gap[gap > 0].items():
            print(f"Note: group {g} has {int(k):,} eligible people never screened. "
                  "Pass eligible=... (or participation=...) to table4 to include them.")
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
    Pooled MLE with delta = 1. Groups share parameters, so the pooled
    log-likelihood is the sum over groups (same as MMS pooling 2005-2007).

    eligible: None -> MMS Table 4 (p_T = 1, screened people only).
              array of eligible people per group -> never-screened people are
              included.
    penalty:  with eligible, None -> estimate y_{T+1}; a number -> fix y_{T+1}
              (0 = optional task). Fixing it identifies c even at beta = 1.
    """

    def __init__(self, counts, dist="logistic", var_mult=1.0, beta_max=1.0, eligible=None,
                 penalty=None):
        self.beta_max = beta_max
        self.fixed_penalty = penalty
        counts = np.atleast_2d(np.asarray(counts, dtype=float))
        self.T = counts.shape[1]
        self.shock = Shock(dist, var_mult)
        self.mandatory = eligible is None
        if self.mandatory:
            self.W = naive_W(self.shock, self.T)
            R = risk_sets(counts)
            # Period T contributes log(1) = 0 because p_T = 1.
            self.n = counts[:, :-1].sum(axis=0)
            self.stay = (R[:, :-1] - counts[:, :-1]).sum(axis=0)
            self.n_people = counts.sum()
            self.never = 0.0
        else:
            eligible = np.broadcast_to(np.asarray(eligible, dtype=float), counts.shape[:1])
            if np.any(eligible < counts.sum(axis=1)):
                raise ValueError("eligible is smaller than the number screened")
            R = eligible[:, None] - (np.cumsum(counts, axis=1) - counts)
            self.n = counts.sum(axis=0)
            self.stay = (R - counts).sum(axis=0)  # period T: never screened
            self.n_people = eligible.sum()
            self.never = self.stay[-1]

    def W_for(self, d=None):
        return self.W if self.mandatory else naive_W(self.shock, self.T, d)

    def loglik(self, theta):
        """theta = (beta, c) or (beta, c, y_{T+1})."""
        beta, c = theta[:2]
        d = None if self.mandatory else theta[2] + c
        return self.loglik_a(beta, (1 - beta) * c, d)

    def loglik_a(self, beta, a, d=None):
        """Log-likelihood in terms of beta, a = (1 - beta) c and d = y_{T+1} + c."""
        x = a + beta * self.W_for(d)[1:]
        return np.sum(self.n * self.shock.logsf(x) + self.stay * self.shock.logcdf(x))

    def start_values(self):
        """
        Starting (beta, a[, d]) list. (beta, a) from OLS on the logistic closed form
        (log-odds of waiting = a + beta W_{t+1}, mandatory W, periods before T),
        approximating other distributions by a logistic with the same variance.
        d is then set so that period T's hazard matches the data.
        """
        s_eq = np.sqrt(self.shock.var / LOGISTIC_VAR)
        W0 = naive_W(self.shock, self.T)
        n, stay = self.n[: self.T - 1], self.stay[: self.T - 1]
        ok = (n > 0) & (stay > 0)
        X = np.column_stack([np.ones(ok.sum()), W0[1:][ok]])
        a, b = np.linalg.lstsq(X, s_eq * np.log(stay[ok] / n[ok]), rcond=None)[0]
        pairs = [(float(np.clip(b, 0.05, 1.5)), a), (0.8, 0.2), (0.5, 2.5)]
        if self.mandatory:
            return [np.array(p) for p in pairs]
        if self.fixed_penalty is not None:
            return [np.array([b, c]) for b in (pairs[0][0], 0.8, 0.5) for c in (-2.0, 0.0, 2.0, 5.0)]
        logodds_T = s_eq * np.log(self.stay[-1] / self.n[-1])
        return [np.array([b, a, (logodds_T - a) / b]) for b, a in pairs]

    def unpack(self, th):
        """Native parameter vector -> (beta, a, d) with a = (1-beta) c, d = y_{T+1} + c."""
        if self.mandatory:
            return th[0], th[1], None
        if self.fixed_penalty is not None:  # th = (beta, c)
            beta, c = th
            return beta, (1 - beta) * c, self.fixed_penalty + c
        return th[0], th[1], th[2]

    def fit(self, start=None):
        """
        MLE with beta restricted to (0, beta_max].

        Native parameters: (beta, a) if mandatory, (beta, a, d) if y_{T+1} is
        estimated, (beta, c) if y_{T+1} is fixed; a = (1 - beta) c, d = y_{T+1} + c.
        With a free or no penalty, c = a / (1 - beta) diverges if beta hits the
        bound beta_max = 1 (reported as +/-inf); a fixed penalty avoids this.
        """
        bmax = self.beta_max
        starts = self.start_values() if start is None else [np.asarray(start, float)]
        k = len(starts[0])
        nll = lambda th: -self.loglik_a(*self.unpack(th))
        negll = lambda th: nll(th) / self.n_people  # scaled for the optimizer
        bounds = [(1e-3, bmax)] + [(None, None)] * (k - 1)
        best = None
        for x0 in starts:
            x0 = np.r_[np.clip(x0[0], 0.01, bmax - 0.01), x0[1:]]
            res = optimize.minimize(negll, x0, method="Nelder-Mead", bounds=bounds,
                                    options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 40000})
            res = optimize.minimize(negll, res.x, method="L-BFGS-B", bounds=bounds,
                                    options={"ftol": 1e-15, "gtol": 1e-10})
            if best is None or res.fun < best.fun:
                best = res
        x = best.x.copy()
        self.at_bound = bmax - x[0] < 1e-6
        V = np.zeros((k, k))  # covariance of the native parameters
        if self.at_bound:
            # beta on the upper bound: no standard error for beta.
            x[0] = bmax
            V[1:, 1:] = np.linalg.inv(numerical_hessian(lambda z: nll(np.r_[bmax, z]), x[1:]))
        else:
            V = np.linalg.inv(numerical_hessian(nll, x))
        sd = lambda g: float(np.sqrt(g @ V @ g))

        beta, a, d = self.unpack(x)
        self.est = x
        self.ll = self.loglik_a(beta, a, d)
        se_b = np.nan if self.at_bound else np.sqrt(V[0, 0])
        if self.fixed_penalty is not None:
            c = x[1]
            g_c = np.array([0.0, 1.0])
            g_a = np.array([-c, 1 - beta])  # a = (1 - beta) c
            self.a, self.se_a = a, sd(g_a)
            self.d, self.se_d = d, sd(g_c)
            self.penalty, self.se_penalty = self.fixed_penalty, np.nan
            self.theta, self.se = np.array([beta, c]), np.array([se_b, sd(g_c)])
            return self
        self.a, self.se_a = a, np.sqrt(V[1, 1])
        if abs(1 - beta) < 1e-6:
            c, se_c = np.copysign(np.inf, a), np.nan  # c = a / 0 diverges: not identified
        else:
            c = a / (1 - beta)
            g_c = np.r_[a / (1 - beta) ** 2, 1 / (1 - beta), np.zeros(k - 2)]  # delta method
            se_c = sd(g_c)
        self.theta = np.array([beta, c])
        self.se = np.array([se_b, se_c])
        if not self.mandatory:
            self.d, self.se_d = d, np.sqrt(V[2, 2])
            if np.isinf(c):
                self.penalty, self.se_penalty = -c, np.nan
            else:
                self.penalty, self.se_penalty = d - c, sd(np.r_[0, 0, 1] - g_c)
        return self

    def predicted_hazard(self):
        d = None if self.mandatory else self.d
        p = self.shock.sf(self.a + self.theta[0] * self.W_for(d)[1:])
        return np.append(p, 1.0) if self.mandatory else p


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


def table4(counts, specs=TABLE4_SPECS, beta_max=1.0, eligible=None, participation=None,
           penalty=None):
    """
    MMS Table 4 columns. By default p_T = 1 (screened people only).
    Give `eligible` (people eligible per group) or `participation` (screened /
    eligible, e.g. 0.52) to include never-screened people. Then y_{T+1} is
    estimated, or fixed at `penalty` (0 = optional task).
    """
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    if participation is not None:
        eligible = counts.sum(axis=1) / participation
    cols = {}
    fits = {}
    for k, (dist, m) in enumerate(specs, start=1):
        mod = Table4Model(counts, dist, m, beta_max, eligible, penalty).fit()
        fits[k] = mod
        var_label = "pi^2/3" if m == 1 else f"{m} x pi^2/3"
        col = {
            "Shock distribution": dist.capitalize(),
            "beta": fmt(mod.theta[0]),
            "  (se)": fmt(mod.se[0], paren=True),
            "delta": "1",
            "c": fmt(mod.theta[1]),
            "  (se) ": fmt(mod.se[1], paren=True),
            "(1-beta) c": fmt(mod.a),
            "  (se)  ": fmt(mod.se_a, paren=True),
        }
        if not mod.mandatory:
            col.update({
                "y_{T+1} (never screened)": fmt(mod.penalty),
                "  (se)   ": fmt(mod.se_penalty, paren=True),
                "y_{T+1} + c": fmt(mod.d),
                "  (se)    ": fmt(mod.se_d, paren=True),
            })
        col.update({
            "Shock variance": var_label,
            "Observations (periods)": f"{mod.T}",
            "Individuals": f"{int(round(mod.n_people)):,}",
            "Never screened": f"{mod.never / mod.n_people:.1%}",
            "Log-likelihood": f"{mod.ll:,.2f}",
        })
        cols[f"({k})"] = col
    return pd.DataFrame(cols), fits


def simulate_counts(beta, c, N=100_000, T=12, dist="logistic", var_mult=1.0, groups=3, seed=0,
                    penalty=None):
    """Draw counts per period from the model (for checking the estimator). N = eligible per group."""
    rng = np.random.default_rng(seed)
    p = hazard(beta, c, Shock(dist, var_mult), T, penalty)
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
    ap.add_argument("--participation", type=float,
                    help="screened / eligible (e.g. 0.52): include never-screened people")
    ap.add_argument("--penalty", type=float,
                    help="fix y_{T+1} (e.g. 0 for an optional task); default: estimate it")
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
        args.participation = None

    print("Counts per period:")
    print(wide.astype(int).to_string(), "\n")
    print("Empirical conditional hazard (pooled):")
    print(pd.Series(empirical_hazard(wide.values), index=wide.columns).round(4).to_string(), "\n")

    tab, fits = table4(wide.values, beta_max=args.beta_max, participation=args.participation,
                       penalty=args.penalty)
    print("Table 4 (pooled):")
    print(tab.to_string())
    if args.out:
        tab.to_csv(args.out)

    if args.by_group:
        for g, row in wide.iterrows():
            print(f"\nTable 4 — group {g}:")
            print(table4(row.values[None, :], beta_max=args.beta_max,
                         participation=args.participation, penalty=args.penalty)[0].to_string())


if __name__ == "__main__":
    main()
