from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

# t-критичні значення Стьюдента, двосторонній 95% (df = 1..30)
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
       2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
       2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
       2.048, 2.045, 2.042]


def mean_std(x: Sequence[float]) -> Tuple[float, float]:
    n = len(x)
    m = sum(x) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))


def ci95(x: Sequence[float]) -> Tuple[float, float]:
    """(середнє, напівширина 95% довірчого інтервалу)."""
    n = len(x)
    m, s = mean_std(x)
    if n < 2:
        return m, float("nan")
    t = T95[n - 2] if n - 1 <= 30 else 1.96
    return m, t * s / math.sqrt(n)


def n_required(sigma: float, eps: float) -> int:
    """Формула (3.5): n = 20 * sigma^2 / eps^2 (β = 0,95, нерівність Чебишева)."""
    return math.ceil(20.0 * sigma ** 2 / eps ** 2)



def gammainc_p(a: float, x: float) -> float:
    """Регуляризована нижня неповна гамма-функція P(a, x)."""
    if x <= 0:
        return 0.0
    if x < a + 1.0:                       # ряд
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))

    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, k: int) -> float:
    """P(χ²_k > x)."""
    if k <= 0:
        return float("nan")
    return 1.0 - gammainc_p(k / 2.0, x / 2.0)


def betainc_reg(a: float, b: float, x: float) -> float:
    """Регуляризована неповна бета-функція I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))

    def cf(a_, b_, x_):
        qab, qap, qam = a_ + b_, a_ + 1, a_ - 1
        c, d = 1.0, 1.0 - qab * x_ / qap
        d = 1e-300 if abs(d) < 1e-300 else d
        d = 1.0 / d
        h = d
        for m in range(1, 500):
            m2 = 2 * m
            aa = m * (b_ - m) * x_ / ((qam + m2) * (a_ + m2))
            d = 1.0 + aa * d
            d = 1e-300 if abs(d) < 1e-300 else d
            c = 1.0 + aa / c
            c = 1e-300 if abs(c) < 1e-300 else c
            d = 1.0 / d
            h *= d * c
            aa = -(a_ + m) * (qab + m) * x_ / ((a_ + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = 1e-300 if abs(d) < 1e-300 else d
            c = 1.0 + aa / c
            c = 1e-300 if abs(c) < 1e-300 else c
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < 1e-14:
                break
        return h

    if x < (a + 1.0) / (a + b + 2.0):
        return front * cf(a, b, x) / a
    return 1.0 - front * cf(b, a, 1.0 - x) / b


def f_sf(F: float, d1: int, d2: int) -> float:
    """P(F_{d1,d2} > F)."""
    if F <= 0:
        return 1.0
    return betainc_reg(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F))



def norm_cdf(x, m, s):
    return 0.5 * (1.0 + math.erf((x - m) / (s * math.sqrt(2.0))))


def lognorm_cdf(x, mu, sg):
    return 0.0 if x <= 0 else norm_cdf(math.log(x), mu, sg)


def gamma_cdf(x, k, theta):
    return 0.0 if x <= 0 else gammainc_p(k, x / theta)


def exp_shift_cdf(x, loc, mean_excess):
    return 0.0 if x <= loc else 1.0 - math.exp(-(x - loc) / mean_excess)



def fit_continuous(data: Sequence[float]) -> List[dict]:
    """Оцінка параметрів гіпотетичних законів методом моментів / МНП."""
    n = len(data)
    m, s = mean_std(data)
    out = []
    out.append({"name": "Нормальний",
                "params": f"m={m:.3f}, σ={s:.3f}", "k": 2,
                "cdf": lambda x, m=m, s=s: norm_cdf(x, m, s)})
    if min(data) > 0:
        lg = [math.log(v) for v in data]
        mu, sg = mean_std(lg)
        out.append({"name": "Логнормальний",
                    "params": f"μ={mu:.3f}, σ={sg:.3f}", "k": 2,
                    "cdf": lambda x, mu=mu, sg=sg: lognorm_cdf(x, mu, sg)})
        k_ = m * m / (s * s)
        th = s * s / m
        out.append({"name": "Гамма (Ерланга)",
                    "params": f"k={k_:.2f}, θ={th:.3f}", "k": 2,
                    "cdf": lambda x, k_=k_, th=th: gamma_cdf(x, k_, th)})
        loc = min(data) - (m - min(data)) / n      # незміщена оцінка зсуву
        ex = m - loc
        out.append({"name": "Зсунутий експоненціальний",
                    "params": f"x₀={loc:.3f}, середнє={ex:.3f}", "k": 2,
                    "cdf": lambda x, loc=loc, ex=ex: exp_shift_cdf(x, loc, ex)})
    return out


def chi2_test_continuous(data: Sequence[float], cdf: Callable, kparams: int,
                         bins: int = 12) -> dict:
    """χ²-критерій з інтервалами рівної емпіричної частоти."""
    x = sorted(data)
    n = len(x)
    edges = [x[min(n - 1, int(round(i * n / bins)))] for i in range(1, bins)]
    edges = sorted(set(edges))
    cuts = [-math.inf] + edges + [math.inf]
    obs = [0] * (len(cuts) - 1)
    for v in x:
        for j in range(len(cuts) - 1):
            if cuts[j] < v <= cuts[j + 1]:
                obs[j] += 1
                break
    stat = 0.0
    for j in range(len(obs)):
        lo = 0.0 if cuts[j] == -math.inf else cdf(cuts[j])
        hi = 1.0 if cuts[j + 1] == math.inf else cdf(cuts[j + 1])
        e = max(n * (hi - lo), 1e-9)
        stat += (obs[j] - e) ** 2 / e
    df = len(obs) - 1 - kparams
    return {"chi2": stat, "df": df,
            "p": chi2_sf(stat, df) if df > 0 else float("nan")}



def fit_discrete(data: Sequence[int]) -> List[dict]:
    m, s = mean_std(data)
    v = s * s
    out = []
    out.append({"name": "Пуассона", "params": f"λ={m:.3f}", "k": 1,
                "pmf": lambda k, m=m: math.exp(
                    -m + k * math.log(m) - math.lgamma(k + 1))})
    out.append({"name": "Нормальний (дискр.)",
                "params": f"m={m:.3f}, σ={s:.3f}", "k": 2,
                "pmf": lambda k, m=m, s=s: (
                    norm_cdf(k + .5, m, s) - (0.0 if k == 0 else
                                              norm_cdf(k - .5, m, s)))})
    if v > m:                          # від'ємний біноміальний (надмірна дисперсія)
        r = m * m / (v - m)
        p = m / v
        out.append({"name": "Від'ємний біноміальний",
                    "params": f"r={r:.2f}, p={p:.3f}", "k": 2,
                    "pmf": lambda k, r=r, p=p: math.exp(
                        math.lgamma(k + r) - math.lgamma(r)
                        - math.lgamma(k + 1) + r * math.log(p)
                        + k * math.log(1 - p))})
    k_ = m * m / v
    th = v / m
    out.append({"name": "Гамма (дискр.)",
                "params": f"k={k_:.2f}, θ={th:.3f}", "k": 2,
                "pmf": lambda k, k_=k_, th=th: (
                    gamma_cdf(k + .5, k_, th)
                    - (0.0 if k == 0 else gamma_cdf(k - .5, k_, th)))})
    return out


def chi2_test_discrete(data: Sequence[int], pmf: Callable, kparams: int,
                       min_expected: float = 5.0) -> dict:
    n = len(data)
    kmax = max(data)
    obs = [0] * (kmax + 1)
    for v in data:
        obs[v] += 1
    exp_ = [n * pmf(k) for k in range(kmax + 1)]
    tail = n - sum(exp_)                       # ймовірність «хвоста»
    exp_[-1] += max(tail, 0.0)
    # об'єднання сусідніх інтервалів до очікуваної частоти >= 5
    O, E, ao, ae = [], [], 0.0, 0.0
    for o, e in zip(obs, exp_):
        ao += o
        ae += e
        if ae >= min_expected:
            O.append(ao); E.append(ae); ao = ae = 0.0
    if ae > 0 and E:
        O[-1] += ao
        E[-1] += ae
    stat = sum((o - e) ** 2 / e for o, e in zip(O, E))
    df = len(O) - 1 - kparams
    return {"chi2": stat, "df": df,
            "p": chi2_sf(stat, df) if df > 0 else float("nan"),
            "bins": len(O)}



def anova_oneway(groups: Sequence[Sequence[float]]) -> dict:
    p = len(groups)
    q = len(groups[0])
    N = p * q
    gm = sum(sum(g) for g in groups) / N
    means = [sum(g) / q for g in groups]
    ssb = q * sum((m - gm) ** 2 for m in means)
    ssw = sum((v - means[j]) ** 2 for j, g in enumerate(groups) for v in g)
    d1, d2 = p - 1, N - p
    D_f = ssb / d1                         # факторна дисперсія
    D_r = ssw / d2                         # залишкова дисперсія
    F = D_f / D_r if D_r > 0 else float("inf")
    return {"F": F, "d1": d1, "d2": d2, "p": f_sf(F, d1, d2),
            "D_factor": D_f, "D_resid": D_r, "means": means, "grand": gm}



def regression(features: Sequence[Sequence[float]], y: Sequence[float]) -> dict:
    X = np.column_stack([np.ones(len(y))] + [np.asarray(f, float)
                                             for f in features])
    Y = np.asarray(y, float)
    b = np.linalg.solve(X.T @ X, X.T @ Y)        # b = (XᵀX)⁻¹ XᵀY
    yhat = X @ b
    ss_res = float(np.sum((Y - yhat) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    R = math.sqrt(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 0 else 1.0
    return {"b": b.tolist(), "R": R, "yhat": yhat.tolist(),
            "ss_res": ss_res}
