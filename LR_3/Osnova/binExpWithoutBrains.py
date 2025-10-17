# from __future__ import annotations
import random
import math
from collections import deque
from typing import Deque, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

def poisson_sample(lmbda: float) -> int:
    if lmbda <= 0:
        return 0
    L = math.exp(-lmbda)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1

class Abonent:
    def __init__(self, id_: int, lambda_in: float, minW: int, maxW: int,
                 max_queue_size: int = 10**6):
        self.id = id_
        self.lambda_in = lambda_in
        self.minW = max(1, int(minW))
        self.maxW = max(1, int(maxW))
        self.W = self.minW
        self.queue: Deque[dict] = deque()
        self.max_queue_size = max_queue_size
        self.successes = 0
        self.collisions = 0
        self.dropped = 0

    def generate_arrivals(self, current_time: int):
        k = poisson_sample(self.lambda_in)
        for _ in range(k):
            if len(self.queue) < self.max_queue_size:
                self.queue.append({'arrival_time': current_time, 'send_slot': None})
            else:
                self.dropped += 1

    def has_scheduled_in_slot(self, slot: int) -> bool:
        return bool(self.queue and self.queue[0].get('send_slot') == slot)

    def schedule_head_if_needed(self, current_time: int):
        if self.queue and self.queue[0].get('send_slot') is None:
            k = random.randint(1, self.W)
            self.queue[0]['send_slot'] = current_time + k

    def notify_result(self, result: str, current_time: int) -> Optional[int]:
        if result == 'SUCCESS':
            if not self.queue:
                return None
            head = self.queue.popleft()
            delay = current_time - head['arrival_time']
            self.successes += 1
            # self.W = max((self.W + 1) // 2, self.minW)
            self.W = self.minW
            if self.queue:
                self.queue[0]['send_slot'] = None
                self.schedule_head_if_needed(current_time)
            return delay
        elif result == 'COLLISION':
            self.collisions += 1
            self.W = min(self.W * 2, self.maxW)
            if self.queue:
                self.queue[0]['send_slot'] = None
                self.schedule_head_if_needed(current_time)
        return None

class SystemChannel:
    def __init__(self, M: int, lambda_total: float, minW: int, maxW: int,
                 max_queue_size: int = 10**6):
        self.M = M
        self.lambda_total = lambda_total
        self.lambda_per_abonent = lambda_total / M
        self.minW = minW
        self.maxW = maxW
        self.max_queue_size = max_queue_size
        self.abonents = [Abonent(i, self.lambda_per_abonent, minW, maxW, max_queue_size)
                         for i in range(M)]
        self.total_successes = 0
        self.total_collisions = 0
        self.total_empty = 0
        self.total_delay_sum = 0.0
        self.total_delivered = 0
        self.total_queue_sum = 0.0
        self.time = 0

    def step(self):
        t = self.time
        for a in self.abonents:
            a.generate_arrivals(t)
        for a in self.abonents:
            a.schedule_head_if_needed(t)
        senders = [a for a in self.abonents if a.has_scheduled_in_slot(t)]
        # накопим текущее количество пакетов в системе
        q_sum = sum(len(a.queue) for a in self.abonents)
        self.total_queue_sum += q_sum

        if len(senders) == 0:
            self.total_empty += 1
        elif len(senders) == 1:
            self.total_successes += 1
            sender = senders[0]
            delay = sender.notify_result('SUCCESS', t)
            if delay is not None:
                self.total_delivered += 1
                self.total_delay_sum += delay
        else:
            self.total_collisions += 1
            for s in senders:
                s.notify_result('COLLISION', t)
        self.time += 1

    def run(self, T: int):
        for _ in range(T):
            self.step()

    def get_metrics(self) -> Tuple[float, float, float]:
        total_slots = self.time or 1
        out_intensity = self.total_successes / total_slots
        avg_delay = (self.total_delay_sum / self.total_delivered) if self.total_delivered else 0.0
        avg_queue = self.total_queue_sum / total_slots
        return out_intensity, avg_delay, avg_queue

def experiment_plot(M: int = 50,
                    lambdas: List[float] = None,
                    T: int = 20000,
                    minW: int = 1,
                    maxW: int = 64,
                    max_queue_size: int = 1000,
                    trials: int = 3):
    if lambdas is None:
        lambdas = [i * 0.1 for i in range(0, 201)]

    results = []
    for lam in lambdas:
        outs, delays, abonents = [], [], []
        for _ in range(trials):
            ch = SystemChannel(M, lam, minW, maxW, max_queue_size)
            ch.run(T)
            o, d, q = ch.get_metrics()
            outs.append(o)
            delays.append(d)
            abonents.append(q)
        results.append((lam,
                        sum(outs) / len(outs),
                        sum(delays) / len(delays),
                        sum(abonents) / len(abonents)))

    lambdas = [r[0] for r in results]
    outs = [r[1] for r in results]
    delays = [r[2] for r in results]
    abonents = [r[3] for r in results]

    plt.figure(figsize=(10, 7))
    plt.plot(lambdas, outs)
    plt.title(f"Интервальная экспоненциальная отсрочка: M={M}, T={T}, W[{minW},{maxW}]")
    plt.ylabel("Интенсивность выхода (успехи/слот)")
    plt.grid(True)

    async_avg_D_theor = np.zeros(len(lambdas))
    avg_N_theor = np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        async_avg_D_theor[i] = avg_D(_lambda=lambdas[i], sync_mode=False)
        avg_N_theor[i] = avg_N(lambdas[i])

    plt.figure(figsize=(10, 7))
    plt.title(f"Интервальная экспоненциальная отсрочка: M={M}, T={T}, W[{minW},{maxW}]")
    plt.plot(lambdas, delays, label="$d_{avg}$")
    plt.plot(lambdas, async_avg_D_theor, linestyle=":", label="Теоретическое (асинхр.)")
    plt.ylabel("$d_{avg}$")
    plt.grid(True)
    plt.legend()

    # TODO: Не работает. Подсчитывает больше N_avg, чем есть на самом деле
    plt.figure(figsize=(10, 7))
    plt.title(f"Интервальная экспоненциальная отсрочка: M={M}, T={T}, W[{minW},{maxW}]")
    plt.plot(lambdas, abonents, label="$N_{avg}$")
    plt.plot(lambdas, avg_N_theor, linestyle=":", label=f"Теоретическое")
    plt.xlabel("Интенсивность входного потока $\lambda$")
    plt.ylabel("$N_{avg}$")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def avg_N(_lambda=0.3):
    return (_lambda * (2 - _lambda)) / (2 * (1 - _lambda))

def avg_D(_lambda=0.3, sync_mode=False):
    d = avg_N(_lambda=_lambda) / _lambda
    return d + 0.5 if sync_mode else d


if __name__ == "__main__":
    M = 1
    T = 200_000
    minW = 1
    maxW = 4096
    max_queue_size = 50_000
    trials = 1

    lambdas = np.linspace(0.0001, 0.8, 9)

    experiment_plot(M=M, lambdas=lambdas, T=T, minW=minW, maxW=maxW,
                    max_queue_size=max_queue_size, trials=trials)
