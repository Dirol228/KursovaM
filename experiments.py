
import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qn_model import build_d4
from stats_tools import (anova_oneway, chi2_test_continuous,
                         chi2_test_discrete, ci95, fit_continuous,
                         fit_discrete, mean_std, n_required, regression)

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                     "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SUMMARY = {}

def run_one(args):
    mean_ia, servers, T, warmup, seed, sample_dt, keep = args
    sim = build_d4(mean_ia=mean_ia, servers=servers, seed=seed,
                   warmup=warmup, sample_dt=sample_dt, keep_samples=keep)
    return sim.run(T)


def replicate(n, workers, mean_ia, T, warmup=0.0, servers=(1, 1, 1),
              sample_dt=None, seed0=1000, keep=False):
    jobs = [(mean_ia, servers, T, warmup, seed0 + i, sample_dt, keep)
            for i in range(n)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(run_one, jobs, chunksize=max(1, n // workers)))
    return [run_one(j) for j in jobs]


def save_csv(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> {path}")


def fmt(m, h):
    return f"{m:.2f} ± {h:.2f}"



def exp1(reps, workers, eps):
    print("\nE1. Перехідний період і тривалість експерименту")
    T, dt, runs4 = 20000.0, 100.0, 5
    conv = {}
    series = {}
    for ia in (8.0, 6.0, 5.0):
        res = replicate(runs4, workers, ia, T, sample_dt=dt, seed0=500)
        n = min(len(r["run_mean"]) for r in res)
        ts = [res[0]["run_mean"][k][0] for k in range(n)]
        ys = [[r["run_mean"][k][1] for k in range(n)] for r in res]
        spread = [max(y[k] for y in ys) - min(y[k] for y in ys)
                  for k in range(n)]
        t_conv = None
        for k in range(n):
            if all(s < eps for s in spread[k:]):
                t_conv = ts[k]
                break
        conv[ia] = {"t_conv": t_conv,
                    "spread_end": spread[-1],
                    "y_end": [y[-1] for y in ys]}
        series[ia] = (ts, ys)
        print(f"  інтервал {ia:g}: T за (3.1) при ε={eps} -> "
              f"{t_conv if t_conv is not None else 'не досягається'}; "
              f"розкид на T={T:g}: {spread[-1]:.3f}")


    N_ens = 400
    W_SM = 5
    warm = {}
    ens = {}
    for ia in (8.0, 6.0):
        res = replicate(N_ens, workers, ia, 1000.0, sample_dt=10.0,
                        seed0=7000)
        n = min(len(r["trajectory"]) for r in res)
        ts = [res[0]["trajectory"][k][0] for k in range(n)]
        raw = [sum(r["trajectory"][k][1] for r in res) / len(res)
               for k in range(n)]
        mean = [sum(raw[max(0, k - W_SM + 1):k + 1])
                / len(raw[max(0, k - W_SM + 1):k + 1]) for k in range(n)]
        i500 = ts.index(500.0)
        plateau = sum(raw[i500:]) / len(raw[i500:])
        t_w = None
        for k in range(n):
            if all(abs(m - plateau) <= 0.03 * plateau for m in mean[k:]):
                t_w = ts[k]
                break
        warm[ia] = {"plateau": plateau, "t_w": t_w}
        ens[ia] = (ts, raw, plateau)
        print(f"  інтервал {ia:g}: плато M ≈ {plateau:.3f}, "
              f"перехідний період (±3%) ≈ {t_w} хв")
    SUMMARY["E1"] = {"eps": eps, "T_trial": T, "runs": runs4,
                     "conv": {str(k): v for k, v in conv.items()},
                     "warm": {str(k): v for k, v in warm.items()}}

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for a, ia, ttl in ((ax[0], 8.0, "інтервал 8 хв (ρ<1)"),
                       (ax[1], 5.0, "інтервал 5 хв (ρ=1)")):
        ts, ys = series[ia]
        for i, y in enumerate(ys):
            a.plot(ts, y, label=f"запуск {i+1}")
        a.set_xlabel("модельний час t, хв")
        a.set_ylabel("середнє M̄(t) на [0; t], стор.")
        a.set_title(f"Збіжність середнього: {ttl}")
        a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E1_convergence.png"), dpi=200)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(5.4, 3.2))
    ts, mean, plateau = ens[8.0]
    a.plot(ts, mean, label=f"M(t), усереднено по {N_ens} запусках")
    a.axhline(plateau, ls="--", color="gray", label="плато")
    if warm[8.0]["t_w"] is not None:
        a.axvline(warm[8.0]["t_w"], ls=":", color="red",
                  label=f"перехідний період ≈ {warm[8.0]['t_w']:g} хв")
    a.set_xlim(0, 500)
    a.set_xlabel("модельний час t, хв")
    a.set_ylabel("зайнято сторінок ОЗП, M(t)")
    a.set_title("Перехідний період, інтервал 8 хв")
    a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E1_warmup.png"), dpi=200)
    plt.close(fig)


    rows = []
    Ts = [500, 1000, 2000, 5000, 10000, 20000]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for ia, color in ((5.0, "tab:red"), (8.0, "tab:blue")):
        ms, hs = [], []
        for T_ in Ts:
            res = replicate(reps, workers, ia, float(T_), seed0=3000)
            m, h = ci95([r["mean_M"] for r in res])
            ms.append(m); hs.append(h)
            rows.append([ia, T_, round(m, 3), round(h, 3)])
        ax.errorbar(Ts, ms, yerr=hs, marker="o", color=color, capsize=3,
                    label=f"середній інтервал {ia:g} хв")
    ax.set_xscale("log")
    ax.set_xlabel("тривалість моделювання T, хв")
    ax.set_ylabel("середня зайнята пам'ять M̄, стор.")
    ax.set_title("Залежність M̄ від T")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E1_mean_M_vs_T.png"), dpi=200)
    plt.close(fig)
    save_csv("E1_mean_M_vs_T.csv",
             ["середній інтервал, хв", "T, хв", "M̄", "напівширина ДІ"], rows)
    SUMMARY["E1"]["M_vs_T"] = rows
    return warm



def exp2(workers, T, warmup):
    print("\nE2. Кількість повторних замірів, формула (3.5)")
    rows, out = [], {}
    for ia, eps_abs in ((8.0, 0.02), (6.0, 0.05), (5.0, 2.0)):
        res = replicate(30, workers, ia, T, warmup, seed0=9000)
        vals = [r["mean_M"] for r in res]
        mx = [r["max_M"] for r in res]
        m, s = mean_std(vals)
        n_sig = n_required(s, s)          # ε = σ
        n_h = n_required(s, s / 2)        # ε = σ/2
        n_abs = n_required(s, eps_abs)
        rows.append([ia, round(m, 3), round(s, 4), n_sig, n_h, eps_abs,
                     n_abs, round(mean_std(mx)[1], 2)])
        out[str(ia)] = {"mean": m, "sigma": s, "n_sigma": n_sig,
                        "n_half": n_h, "eps": eps_abs, "n_eps": n_abs}
        print(f"  інтервал {ia:g}: M̄={m:.3f}, σ={s:.4f}; n(ε=σ)={n_sig}, "
              f"n(ε=σ/2)={n_h}, n(ε={eps_abs})={n_abs}")
    save_csv("E2_replications.csv",
             ["середній інтервал, хв", "M̄ (пробні)", "σ", "n для ε=σ",
              "n для ε=σ/2", "задана ε, стор.", "n для заданої ε",
              "σ(max M)"], rows)
    SUMMARY["E2"] = out


    N = 100
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    erows = []
    for ia, color in ((8.0, "tab:blue"), (5.0, "tab:red")):
        res = replicate(N, workers, ia, T, warmup, seed0=9500)
        vals = [r["mean_M"] for r in res]
        ns = [5, 10, 20, 30, 50, 100]
        rel = []
        for n in ns:
            m, h = ci95(vals[:n])
            rel.append(100 * h / m)
            erows.append([ia, n, round(m, 3), round(h, 3),
                          round(100 * h / m, 2)])
        ax.plot(ns, rel, marker="o", color=color,
                label=f"середній інтервал {ia:g} хв")
    ax.axhline(5, ls="--", color="gray", label="вимога 5%")
    ax.set_xlabel("кількість повторних запусків n")
    ax.set_ylabel("відносна напівширина 95% ДІ для M̄, %")
    ax.set_title(f"Точність оцінки (T = {T:g} хв)")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E2_precision.png"), dpi=200)
    plt.close(fig)
    save_csv("E2_precision.csv",
             ["середній інтервал, хв", "n", "M̄", "напівширина ДІ",
              "відн. напівширина, %"], erows)
    SUMMARY["E2"]["precision"] = erows



def exp3(reps, workers, T, warmup):
    print("\nE3. Вплив інтенсивності надходження завдань")
    means = [12.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0, 5.5, 5.0,
             4.5, 4.0, 3.5, 3.0]
    rows, table = [], []
    data = {}
    hists = {}
    for ia in means:
        res = replicate(reps, workers, ia, T, warmup, seed0=11000)
        m1, h1 = ci95([r["mean_M"] for r in res])
        m2, h2 = ci95([r["max_M"] for r in res])
        sd, _ = ci95([r["std_M"] for r in res])
        mn = min(r["min_M"] for r in res)
        u = [ci95([r["util"][i] for r in res])[0] for i in range(3)]
        q = [ci95([r["mean_queue"][i] for r in res])[0] for i in range(3)]
        soj, _ = ci95([r["soj_mean"] for r in res])
        data[ia] = (m1, h1, m2, h2)
        keys = sorted({k for r in res for k in r["hist_M"]})
        hists[ia] = {k: sum(r["hist_M"].get(k, 0) for r in res) / reps
                     for k in keys}
        rows.append([ia, round(1 / ia, 4), round(m1, 2), round(h1, 2),
                     round(sd, 2), mn, round(m2, 1), round(h2, 1),
                     *[round(x, 3) for x in u], *[round(x, 2) for x in q],
                     round(soj, 1)])
        table.append(f"  інтервал {ia:>4g}: M̄={fmt(m1, h1):>14}  "
                     f"max M={m2:7.1f}  ρ=({u[0]:.2f},{u[1]:.2f},{u[2]:.2f})"
                     f"  черги=({q[0]:.1f},{q[1]:.1f},{q[2]:.1f})")
    print("\n".join(table))
    save_csv("E3_intensity.csv",
             ["середній інтервал, хв", "λ, 1/хв", "M̄", "ДІ M̄", "σ(M)",
              "min M", "max M (сер.)", "ДІ max M", "ρ1", "ρ2", "ρ3",
              "черга1", "черга2", "черга3", "час у системі, хв"], rows)


    stable = [ia for ia in means if ia >= 5.5]
    over = [ia for ia in means if ia <= 4.5]
    lam_s = [1 / ia for ia in stable]
    ys = [data[ia][0] for ia in stable]
    reg_s = regression([lam_s], ys)                      # M = b0 + b1·λ
    reg_s2 = regression([lam_s, [1 / (1 - 5.0 / ia) for ia in stable]], ys)
    LITTLE = 1 * 5.0 + 2 * 5.0 + 3 * 4.5                 # 28,5 стор.·хв/завд.
    lam_o = [1 / ia for ia in over]
    reg_o = regression([lam_o], [data[ia][0] for ia in over])
    mu1 = 0.2
    slope_theory = (T + warmup) / 2.0
    SUMMARY["E3"] = {
        "T": T, "warmup": warmup, "reps": reps,
        "stable_ia": stable, "over_ia": over,
        "reg_stable": {"b0": reg_s["b"][0], "b1": reg_s["b"][1],
                       "R": reg_s["R"], "little": LITTLE},
        "reg_stable2": {"b": reg_s2["b"], "R": reg_s2["R"]},
        "reg_over": {"b0": reg_o["b"][0], "b1": reg_o["b"][1],
                     "R": reg_o["R"], "slope_theory": slope_theory,
                     "mu1": mu1,
                     "ratio_b0_b1": -reg_o["b"][0] / reg_o["b"][1]}}
    print(f"  регресія (стац.): M̄ = {reg_s['b'][0]:.3f} + "
          f"{reg_s['b'][1]:.3f}·λ, R={reg_s['R']:.4f}; Літтл: {LITTLE}·λ")
    print(f"  регресія 2 (стац.): b={[round(b, 3) for b in reg_s2['b']]}, "
          f"R={reg_s2['R']:.5f}")
    print(f"  регресія (перевант.): M̄ = {reg_o['b'][0]:.1f} + "
          f"{reg_o['b'][1]:.1f}·λ, R={reg_o['R']:.4f}; "
          f"теор. нахил (T+W)/2={slope_theory:.0f}; "
          f"-b0/b1={-reg_o['b'][0]/reg_o['b'][1]:.3f} (μ1=0,2)")

    lam = [1 / ia for ia in means]

    fig, a = plt.subplots(figsize=(5.6, 3.3))
    a.errorbar(lam, [data[i][0] for i in means],
               yerr=[data[i][1] for i in means], marker="o", capsize=3,
               label="середня M̄")
    a.errorbar(lam, [data[i][2] for i in means],
               yerr=[data[i][3] for i in means], marker="s", capsize=3,
               color="tab:red", label="максимальна M")
    a.axvline(0.2, ls="--", color="gray")
    a.text(0.205, max(data[i][2] for i in means) * 0.5, "ρ=1\nλ=0,2",
           color="gray")
    a.set_xlabel("інтенсивність надходження λ, завдань/хв")
    a.set_ylabel("зайнято сторінок ОЗП")
    a.set_title(f"M̄ та max M залежно від λ (T={T:g} хв)")
    a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E3_intensity.png"), dpi=200)
    plt.close(fig)


    fig, a = plt.subplots(figsize=(5.6, 3.3))
    xs = [1 / i for i in stable]
    a.errorbar(xs, [data[i][0] for i in stable],
               yerr=[data[i][1] for i in stable], fmt="o", capsize=3,
               label="імітація")
    grid = [min(xs) + (max(xs) - min(xs)) * k / 100 for k in range(101)]
    a.plot(grid, [reg_s["b"][0] + reg_s["b"][1] * x for x in grid],
           "r-", label=f"b₀+b₁·λ, R={reg_s['R']:.3f}")
    a.plot(grid, [reg_s2["b"][0] + reg_s2["b"][1] * x
                  + reg_s2["b"][2] / (1 - 5.0 * x) for x in grid],
           "g--", label=f"b₀+b₁·λ+b₂/(1−ρ₁), R={reg_s2['R']:.4f}")
    a.plot(grid, [LITTLE * x for x in grid], "k:",
           label="28,5·λ (закон Літтла, без черг)")
    a.set_xlabel("інтенсивність надходження λ, завдань/хв")
    a.set_ylabel("середня зайнята пам'ять M̄, стор.")
    a.set_title("Стаціонарна область (ρ<1): регресія M̄(λ)")
    a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E3_stable_regression.png"), dpi=200)
    plt.close(fig)


    fig, a = plt.subplots(figsize=(5.6, 3.3))
    for j, ia in enumerate((10.0, 8.0, 6.0)):
        h = hists[ia]
        ks = sorted(h)
        a.bar([k + (j - 1) * 0.27 for k in ks], [h[k] for k in ks],
              width=0.27, label=f"середній інтервал {ia:g} хв")
    a.set_xlim(-0.5, 16)
    a.set_xlabel("зайнято сторінок ОЗП, m")
    a.set_ylabel("частка часу P(M = m)")
    a.set_title("Розподіл зайнятої пам'яті (стаціонарні режими)")
    a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E3_pmf.png"), dpi=200)
    plt.close(fig)


    fig, a = plt.subplots(figsize=(5.6, 3.3))
    lo = [1 / i for i in over]
    a.errorbar(lo, [data[i][0] for i in over], yerr=[data[i][1] for i in over],
               fmt="o", capsize=3, label="імітація")
    g2 = [min(lo) + (max(lo) - min(lo)) * k / 100 for k in range(101)]
    a.plot(g2, [reg_o["b"][0] + reg_o["b"][1] * x for x in g2], "r-",
           label=f"регресія, R={reg_o['R']:.4f}")
    a.set_xlabel("інтенсивність надходження λ, завдань/хв")
    a.set_ylabel("середня зайнята пам'ять M̄, стор.")
    a.set_title("Область перевантаження: M̄ = b₀ + b₁·λ")
    a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E3_overload_regression.png"), dpi=200)
    plt.close(fig)



def exp4(reps, workers, T, warmup):
    print("\nE4. Пропозиції щодо поліпшення та дисперсійний аналіз")
    variants = [("базова (1,1,1)", (1, 1, 1)),
                ("+CPU1 (2,1,1)", (2, 1, 1)),
                ("+CPU2 (1,2,1)", (1, 2, 1)),
                ("+CPU1,CPU2 (2,2,1)", (2, 2, 1)),
                ("+усі (2,2,2)", (2, 2, 2))]
    rows, groups, labels, vals, errs, maxs = [], [], [], [], [], []
    for name, srv in variants:
        res = replicate(reps, workers, 5.0, T, warmup, servers=srv,
                        seed0=13000)
        v = [r["mean_M"] for r in res]
        m, h = ci95(v)
        mx, _ = ci95([r["max_M"] for r in res])
        soj, _ = ci95([r["soj_mean"] for r in res])
        u = [ci95([r["util"][i] for r in res])[0] for i in range(3)]
        rows.append([name, round(m, 2), round(h, 2), round(mx, 1),
                     round(soj, 1), *[round(x, 3) for x in u]])
        groups.append(v); labels.append(name)
        vals.append(m); errs.append(h); maxs.append(mx)
        print(f"  {name:<22} M̄={fmt(m, h):>14}  max M={mx:6.1f}  "
              f"час у системі={soj:7.1f}  ρ=({u[0]:.2f},{u[1]:.2f},"
              f"{u[2]:.2f})")
    save_csv("E4_improvement.csv",
             ["варіант", "M̄", "ДІ", "max M", "час у системі, хв",
              "ρ1", "ρ2", "ρ3"], rows)
    an_all = anova_oneway(groups)
    an_2 = anova_oneway([groups[0], groups[3]])
    an_3 = anova_oneway([groups[3], groups[4]])
    an_b1 = anova_oneway([groups[0], groups[1]])
    SUMMARY["E4"] = {
        "variants": rows, "reps": reps,
        "anova_all": {k: an_all[k] for k in
                      ("F", "d1", "d2", "p", "D_factor", "D_resid")},
        "anova_base_vs_221": {k: an_2[k] for k in ("F", "d1", "d2", "p")},
        "anova_221_vs_222": {k: an_3[k] for k in ("F", "d1", "d2", "p")},
        "anova_base_vs_211": {k: an_b1[k] for k in ("F", "d1", "d2", "p")}}
    print(f"  ANOVA (усі 5): F={an_all['F']:.1f}, p={an_all['p']:.2e}")
    print(f"  ANOVA (базова vs 2,1,1): F={an_b1['F']:.2f}, p={an_b1['p']:.3f}")
    print(f"  ANOVA (базова vs 2,2,1): F={an_2['F']:.1f}, p={an_2['p']:.2e}")
    print(f"  ANOVA (2,2,1 vs 2,2,2): F={an_3['F']:.1f}, p={an_3['p']:.2e}")


    rows4 = []
    for name, srv in (("(2,2,1)", (2, 2, 1)), ("(2,2,2)", (2, 2, 2))):
        res = replicate(reps, workers, 4.0, T, warmup, servers=srv,
                        seed0=15000)
        m, h = ci95([r["mean_M"] for r in res])
        mx, _ = ci95([r["max_M"] for r in res])
        u = [ci95([r["util"][i] for r in res])[0] for i in range(3)]
        rows4.append([name, round(m, 2), round(h, 2), round(mx, 1),
                      *[round(x, 3) for x in u]])
        print(f"  інтервал 4 хв, {name}: M̄={fmt(m, h)}, max M={mx:.1f}, "
              f"ρ=({u[0]:.2f},{u[1]:.2f},{u[2]:.2f})")
    save_csv("E4_interval4.csv",
             ["варіант", "M̄", "ДІ", "max M", "ρ1", "ρ2", "ρ3"], rows4)
    SUMMARY["E4"]["interval4"] = rows4

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    x = range(len(labels))
    ax.bar(x, vals, yerr=errs, capsize=4, color="tab:blue", label="M̄")
    ax.plot(x, maxs, "rs", label="max M")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("зайнято сторінок ОЗП")
    ax.set_title("Вплив додаткових процесорів (інтервал 5±2 хв)")
    ax.grid(alpha=.3, axis="y"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E4_improvement.png"), dpi=200)
    plt.close(fig)



def exp5(workers, T, warmup):
    print("\nE5. Ідентифікація закону розподілу (інтервал 8 хв, ρ<1)")
    ia = 8.0
    res_a = replicate(100, workers, ia, T, warmup, seed0=17000)
    data_a = [r["mean_M"] for r in res_a]
    res_b = replicate(20, workers, ia, T, warmup, seed0=19000,
                      sample_dt=200.0, keep=True)
    stride = 30
    data_b = [v for r in res_b for v in r["soj_samples"][::stride]]
    data_c = [m for r in res_b for (t, m) in r["trajectory"] if t >= warmup]

    rows, plots = [], []
    for title, data, kind in (
            ("M̄ за запуск, стор.", data_a, "cont"),
            ("Час перебування завдання, хв", data_b, "cont"),
            ("Миттєва зайнята пам'ять M(t), стор.", data_c, "disc")):
        m, s = mean_std(data)
        if kind == "cont":
            cand = fit_continuous(data)
            for c in cand:
                nb = 8 if len(data) < 200 else 15
                t = chi2_test_continuous(data, c["cdf"], c["k"], bins=nb)
                c.update(t)
        else:
            cand = fit_discrete(data)
            for c in cand:
                c.update(chi2_test_discrete(data, c["pmf"], c["k"]))
        cand = [c for c in cand if c["df"] > 0]
        best = max(cand, key=lambda c: (c["p"], -c["chi2"]))
        for c in cand:
            rows.append([title, len(data), round(m, 3), round(s, 3),
                         c["name"], c["params"], round(c["chi2"], 2), c["df"],
                         f"{c['p']:.4f}",
                         "так" if c["p"] > 0.05 else "ні",
                         "★" if c is best else ""])
        print(f"  {title}: N={len(data)}, середнє={m:.3f}, σ={s:.3f}; "
              f"найкращий закон: {best['name']} ({best['params']}), "
              f"χ²={best['chi2']:.2f}, df={best['df']}, p={best['p']:.4f}")
        plots.append((title, data, kind, best))
    save_csv("E5_distribution.csv",
             ["Величина", "N", "Середнє", "σ", "Гіпотеза", "Параметри",
              "χ²", "df", "p-значення", "Не відхиляється (p>0,05)", "Найкращий"],
             rows)
    SUMMARY["E5"] = [r for r in rows]
    SUMMARY["E5_pmf"] = {str(k): data_c.count(k) / len(data_c)
                         for k in range(0, max(data_c) + 1)}

    for tag, (title, data, kind, best) in zip("abc", plots):
        fig, a = plt.subplots(figsize=(5.4, 3.1))
        if kind == "cont":
            nb = 12 if len(data) < 300 else 25
            a.hist(data, bins=nb, density=True, alpha=.6, label="гістограма")
            lo, hi = min(data), max(data)
            xs = [lo + (hi - lo) * k / 200 for k in range(1, 200)]
            cdf = best["cdf"]
            dx = (hi - lo) / 200
            a.plot(xs, [(cdf(x + dx / 2) - cdf(x - dx / 2)) / dx for x in xs],
                   "r-", label=f"гіпотеза: {best['name']}")
            a.set_ylabel("щільність, 1/од.")
        else:
            ks = list(range(0, max(data) + 1))
            a.bar(ks, [data.count(k) / len(data) for k in ks], alpha=.6,
                  label="емпіричні частоти")
            a.plot(ks, [best["pmf"](k) for k in ks], "ro-",
                   label=f"гіпотеза: {best['name']}")
            a.set_ylabel("частота")
        a.set_xlabel(title)
        a.set_title(f"N = {len(data)}")
        a.grid(alpha=.3); a.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"E5{tag}_distribution.png"), dpi=200)
        plt.close(fig)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100,
                    help="кількість повторних запусків у серіях E3-E4")
    ap.add_argument("--T", type=float, default=10000.0,
                    help="тривалість одного запуску, хв")
    ap.add_argument("--eps", type=float, default=0.05,
                    help="точність ε у формулі (3.1), стор.")
    ap.add_argument("--workers", type=int, default=1,
                    help="кількість процесів для паралельних запусків")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.T = 12, 3000.0
    warm = exp1(a.reps, a.workers, a.eps)
    t_w = max(w["t_w"] or 0 for w in warm.values())
    WARMUP = max(100.0, math.ceil(2 * t_w / 100.0) * 100.0)
    SUMMARY["WARMUP"] = WARMUP
    print(f"\nОбрано перехідний період (прогрів): {WARMUP:g} хв")
    exp2(a.workers, a.T, WARMUP)
    exp3(a.reps, a.workers, a.T, WARMUP)
    exp4(a.reps, a.workers, a.T, WARMUP)
    exp5(a.workers, a.T, WARMUP)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, ensure_ascii=False, indent=1, default=str)
    print("\nГотово. Результати в папці:", OUT)
