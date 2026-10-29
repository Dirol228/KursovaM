
import csv
import os

from qn_model import (QNSimulation, StationConfig, build_d4, const, exp)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

rows = []   # (тест, показник, очікуване, отримане, відхилення %, статус)


def check(test, name, expected, got, tol_pct):
    dev = abs(got - expected) / abs(expected) * 100 if expected else abs(got)
    ok = dev <= tol_pct
    rows.append((test, name, expected, got, dev, "OK" if ok else "FAIL"))


def test_deterministic():
    stations = [StationConfig("CPU1", const(5.0)),
                StationConfig("CPU2", const(2.5)),
                StationConfig("CPU3", const(1.5))]
    sim = QNSimulation(stations, const(100.0), [1, 2, 3], seed=1)
    r = sim.run(1000.0)
    # одне завдання: 5 + 2*2.5 + 3*1.5 = 14.5 хв; сторінко-хвилин:
    # 1*5 + 2*5 + 3*4.5 = 28.5; завершено 9 завдань (приходи в 100..900)
    check("1 детермінований", "завершено завдань", 9, r["completed"], 0)
    check("1 детермінований", "час перебування, хв", 14.5, r["soj_mean"], 1e-9)
    check("1 детермінований", "середня зайнята пам'ять, стор.",
          9 * 28.5 / 1000, r["mean_M"], 1e-9)
    check("1 детермінований", "max пам'яті, стор.", 3, r["max_M"], 0)
    check("1 детермінований", "завантаження CPU1", 9 * 5 / 1000,
          r["util"][0], 1e-9)
    check("1 детермінований", "завантаження CPU3", 9 * 4.5 / 1000,
          r["util"][2], 1e-9)


def test_mm1():
    lam_inv, mu_inv = 10.0, 6.0          # ρ = 0.6
    rho = mu_inv / lam_inv
    Ls = rho / (1 - rho)                 # 1.5
    Lq = rho ** 2 / (1 - rho)            # 0.9
    W = mu_inv / (1 - rho)               # 15
    L_acc = Lq_acc = W_acc = U_acc = 0.0
    seeds = range(1, 6)
    for s in seeds:
        sim = QNSimulation(
            [StationConfig("S", exp(mu_inv), per_page=False)],
            exp(lam_inv), [1], seed=s, warmup=2000.0)
        r = sim.run(600000.0)
        L_acc += r["mean_n_sys"]
        Lq_acc += r["mean_queue"][0]
        W_acc += r["soj_mean"]
        U_acc += r["util"][0]
    n = len(list(seeds))
    check("2 M/M/1", "ρ (завантаження)", rho, U_acc / n, 2)
    check("2 M/M/1", "L (завдань у системі)", Ls, L_acc / n, 3)
    check("2 M/M/1", "Lq (довжина черги)", Lq, Lq_acc / n, 4)
    check("2 M/M/1", "W (час у системі)", W, W_acc / n, 3)


def test_d4_stable():
    mean_ia = 8.0
    es = [5.0, 2 * 2.5, 3 * 1.5]         # E[S_i]
    lam = 1.0 / mean_ia
    acc = {"L": 0, "W": 0, "thr": 0, "u": [0, 0, 0]}
    N = 5
    for s in range(1, N + 1):
        sim = build_d4(mean_ia=mean_ia, seed=s, warmup=1000.0)
        r = sim.run(400000.0)
        acc["L"] += r["mean_n_sys"]
        acc["W"] += r["soj_mean"]
        acc["thr"] += r["throughput"]
        for i in range(3):
            acc["u"][i] += r["util"][i]
    L, W, thr = acc["L"] / N, acc["W"] / N, acc["thr"] / N
    check("3 D4, ρ<1", "пропускна здатність λ", lam, thr, 1)
    check("3 D4, ρ<1", "закон Літтла: L = λ·W", L, thr * W, 1)
    for i in range(3):
        check("3 D4, ρ<1", f"завантаження CPU{i+1} = λ·E[S]",
              lam * es[i], acc["u"][i] / N, 1)


def test_invariant():
    sim = build_d4(mean_ia=5.0, seed=7)
    sim.run(3000.0)
    expected = sum((len(sim.queue[i]) + sim.busy[i]) * sim.pages_in[i]
                   for i in range(3))
    check("4 інваріант", "M = Σ n_i·pages_i", expected, sim.M, 0)


if __name__ == "__main__":
    test_deterministic()
    test_mm1()
    test_d4_stable()
    test_invariant()

    print(f"{'Тест':<18}{'Показник':<38}{'Очікуване':>12}{'Отримане':>12}"
          f"{'Відх.,%':>9}  Статус")
    for t, n, e, g, d, s in rows:
        print(f"{t:<18}{n:<38}{e:>12.4f}{g:>12.4f}{d:>9.3f}  {s}")
    with open(os.path.join(OUT, "verification.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Тест", "Показник", "Очікуване", "Отримане",
                    "Відхилення, %", "Статус"])
        w.writerows(rows)
    print("\nПідсумок:", "ВСІ ТЕСТИ ПРОЙДЕНО" if all(
        r[-1] == "OK" for r in rows) else "Є ПОМИЛКИ")
