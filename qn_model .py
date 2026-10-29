from __future__ import annotations

import heapq
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


Gen = Callable[[random.Random], float]


def const(d: float) -> Gen:
    return lambda rng: d


def exp(d: float) -> Gen:
    return lambda rng: rng.expovariate(1.0 / d)


def unif(d: float, s: float) -> Gen:
    return lambda rng: rng.uniform(d - s, d + s)



@dataclass
class StationConfig:
    name: str
    time_fn: Gen
    servers: int = 1
    per_page: bool = True


class Job:
    __slots__ = ("id", "t_arrive", "pages", "counted")

    def __init__(self, jid: int, t: float, pages: int, counted: bool):
        self.id = jid
        self.t_arrive = t
        self.pages = pages
        self.counted = counted


# Типи подій
ARRIVAL, DEPART, RESET, SAMPLE = 0, 1, 2, 3


class QNSimulation:

    def __init__(self, stations: List[StationConfig], interarrival: Gen,
                 pages_in: List[int], seed: Optional[int] = None,
                 warmup: float = 0.0, sample_dt: Optional[float] = None,
                 keep_samples: bool = False):
        assert len(stations) == len(pages_in)
        self.st = stations
        self.k = len(stations)
        self.ia = interarrival
        self.pages_in = pages_in
        self.rng = random.Random(seed)
        self.warmup = warmup
        self.sample_dt = sample_dt
        self.keep_samples = keep_samples
        self.soj_samples: List[float] = []   # час перебування кожного завдання
        self.run_mean: List[Tuple[float, float]] = []  # (t, ȳ(t)) за (3.2)

        self.t = 0.0
        self._heap: List[Tuple[float, float, int, int, object]] = []
        self._seq = 0

        # стан системи
        self.queue = [deque() for _ in range(self.k)]
        self.busy = [0] * self.k
        self.M = 0                      # зайнято сторінок ОЗП
        self.n_sys = 0                  # завдань у системі
        self.next_id = 0
        self.trajectory: List[Tuple[float, int]] = []
        self._reset_stats(0.0)


    def _reset_stats(self, t0: float):
        self.t_stat0 = t0
        self.last_t = t0
        self.int_M = 0.0
        self.int_M2 = 0.0
        self.int_nsys = 0.0
        self.int_q = [0.0] * self.k
        self.int_busy = [0.0] * self.k
        self.hist: Dict[int, float] = defaultdict(float)
        self.M_min = self.M
        self.M_max = self.M
        self.max_q = [len(q) for q in self.queue]
        self.completed = 0
        self.arrived = 0
        # середній час перебування (алгоритм Велфорда)
        self.soj_n = 0
        self.soj_mean = 0.0
        self.soj_m2 = 0.0
        self.soj_min = float("inf")
        self.soj_max = 0.0

    def _advance(self, t: float):
        dt = t - self.last_t
        if dt > 0:
            self.int_M += self.M * dt
            self.int_M2 += self.M * self.M * dt
            self.int_nsys += self.n_sys * dt
            self.hist[self.M] += dt
            for i in range(self.k):
                self.int_q[i] += len(self.queue[i]) * dt
                self.int_busy[i] += self.busy[i] * dt
            self.last_t = t
        self.t = t

    def _record_sojourn(self, x: float):
        self.soj_n += 1
        d = x - self.soj_mean
        self.soj_mean += d / self.soj_n
        self.soj_m2 += d * (x - self.soj_mean)
        self.soj_min = min(self.soj_min, x)
        self.soj_max = max(self.soj_max, x)
        if self.keep_samples:
            self.soj_samples.append(x)


    def _push(self, t: float, kind: int, data=None):
        # одночасні події виконуються у випадковому порядку (випадковий
        # ключ другим елементом кортежу), як рекомендує методичка
        self._seq += 1
        heapq.heappush(self._heap,
                       (t, self.rng.random(), self._seq, kind, data))


    def _start_service(self, i: int, job: Job):
        cfg = self.st[i]
        self.busy[i] += 1
        n = job.pages if cfg.per_page else 1
        dur = sum(cfg.time_fn(self.rng) for _ in range(n))
        self._push(self.t + dur, DEPART, (i, job))

    def _enter(self, i: int, job: Job):
        if self.busy[i] < self.st[i].servers:
            self._start_service(i, job)
        else:
            q = self.queue[i]
            q.append(job)
            if len(q) > self.max_q[i]:
                self.max_q[i] = len(q)

    def _on_arrival(self):
        job = Job(self.next_id, self.t, self.pages_in[0],
                  self.t >= self.warmup)
        self.next_id += 1
        self.arrived += 1
        self.n_sys += 1
        self.M += job.pages
        self._update_minmax()
        self._push(self.t + self.ia(self.rng), ARRIVAL)
        self._enter(0, job)

    def _on_depart(self, i: int, job: Job):
        self.busy[i] -= 1
        if self.queue[i]:                       # наступне з черги
            self._start_service(i, self.queue[i].popleft())
        if i + 1 < self.k:                      # маршрут до наступної СМО
            new_pages = self.pages_in[i + 1]
            self.M += new_pages - job.pages     # зростання обсягу пам'яті
            job.pages = new_pages
            self._update_minmax()
            self._enter(i + 1, job)
        else:                                   # вихід, пам'ять звільнена
            self.M -= job.pages
            self.n_sys -= 1
            self.completed += 1
            self._update_minmax()
            if job.counted:
                self._record_sojourn(self.t - job.t_arrive)

    def _update_minmax(self):
        if self.M > self.M_max:
            self.M_max = self.M
        if self.M < self.M_min:
            self.M_min = self.M

-
    def run(self, T: float) -> dict:
        self._push(self.ia(self.rng), ARRIVAL)
        if self.warmup > 0:
            self._push(self.warmup, RESET)
        if self.sample_dt:
            self._push(0.0, SAMPLE)
        while self._heap:
            t, _, _, kind, data = heapq.heappop(self._heap)
            if t > T:
                break
            self._advance(t)
            if kind == ARRIVAL:
                self._on_arrival()
            elif kind == DEPART:
                self._on_depart(*data)
            elif kind == RESET:
                self._reset_stats(t)
            else:  # SAMPLE
                self.trajectory.append((t, self.M))
                if t > self.t_stat0:
                    self.run_mean.append(
                        (t, self.int_M / (t - self.t_stat0)))
                self._push(t + self.sample_dt, SAMPLE)
        self._advance(T)
        return self.results(T)


    def results(self, T: float) -> dict:
        Teff = T - self.t_stat0
        mean_M = self.int_M / Teff
        var_M = max(self.int_M2 / Teff - mean_M ** 2, 0.0)
        hist = {m: v / Teff for m, v in sorted(self.hist.items())}
        soj_std = (self.soj_m2 / (self.soj_n - 1)) ** 0.5 \
            if self.soj_n > 1 else float("nan")
        return {
            "T": T, "Teff": Teff,
            "mean_M": mean_M, "std_M": var_M ** 0.5,
            "min_M": self.M_min, "max_M": self.M_max,
            "hist_M": hist,
            "mean_n_sys": self.int_nsys / Teff,
            "util": [self.int_busy[i] / (self.st[i].servers * Teff)
                     for i in range(self.k)],
            "mean_queue": [self.int_q[i] / Teff for i in range(self.k)],
            "max_queue": list(self.max_q),
            "arrived": self.arrived, "completed": self.completed,
            "throughput": self.completed / Teff,
            "soj_mean": self.soj_mean if self.soj_n else float("nan"),
            "soj_std": soj_std,
            "soj_min": self.soj_min if self.soj_n else float("nan"),
            "soj_max": self.soj_max if self.soj_n else float("nan"),
            "soj_n": self.soj_n,
            "trajectory": self.trajectory,
            "run_mean": self.run_mean,
            "soj_samples": self.soj_samples,
        }



def build_d4(mean_ia: float = 5.0, ia_half: float = 2.0,
             servers: Tuple[int, int, int] = (1, 1, 1),
             seed: Optional[int] = None, warmup: float = 0.0,
             sample_dt: Optional[float] = None,
             keep_samples: bool = False) -> QNSimulation:
    """Модель D4.

    Надходження       : unif(mean_ia, ia_half)       (5±2 хв)
    СМО1 (трансляція) : unif(5, 1)   на сторінку      (1 стор.)
    СМО2 (редагування): unif(2.5,0.5) на сторінку     (2 стор.)
    СМО3 (обробка)    : unif(1.5,0.4) на сторінку     (3 стор.)
    """
    stations = [
        StationConfig("CPU1 (трансляція)", unif(5.0, 1.0), servers[0]),
        StationConfig("CPU2 (редагування)", unif(2.5, 0.5), servers[1]),
        StationConfig("CPU3 (обробка)", unif(1.5, 0.4), servers[2]),
    ]
    return QNSimulation(stations, unif(mean_ia, ia_half), [1, 2, 3],
                        seed=seed, warmup=warmup, sample_dt=sample_dt,
                        keep_samples=keep_samples)
