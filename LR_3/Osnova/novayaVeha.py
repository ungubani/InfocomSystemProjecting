import addcopyfighandler
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
                 max_queue_size: int = 10 ** 6):
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

            ## FIX 1: Задержка считается до конца текущего слота (current_time + 1). ##
            delay = (current_time) - head['arrival_time']

            self.successes += 1
            self.W = self.minW
            if self.queue:
                self.queue[0]['send_slot'] = None
                self.schedule_head_if_needed(current_time)
            return delay + 0.5

        elif result == 'COLLISION':
            self.collisions += 1
            self.W = min(self.W * 2, self.maxW)
            if self.queue:
                self.queue[0]['send_slot'] = None
                self.schedule_head_if_needed(current_time)
        return None


class SystemChannel:
    def __init__(self, M: int, lambda_total: float, minW: int, maxW: int,
                 max_queue_size: int = 10 ** 6):
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

        q_sum = sum(len(a.queue) for a in self.abonents)
        self.total_queue_sum += q_sum

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
        lambdas = np.linspace(0.001, 0.99, 50)

    results = []
    print("Проведение симуляции...")
    for i, lam in enumerate(lambdas):
        print(f"  Шаг {i + 1}/{len(lambdas)}: λ = {lam:.3f}")
        outs, delays, abonents = [], [], []
        for _ in range(trials):
            ch = SystemChannel(M, lam, minW, maxW, max_queue_size)
            ch.run(T)
            o, d, q = ch.get_metrics()
            outs.append(o)
            delays.append(d)
            abonents.append(q)
        results.append((lam,
                        np.mean(outs),
                        np.mean(delays),
                        np.mean(abonents)))

    lmbds = [r[0] for r in results]
    outs = [r[1] for r in results]
    delays = [r[2] for r in results]
    abonents = [r[3] for r in results]

    theor_lambdas = [l for l in lmbds if l < 1.0]
    avg_N_theor = [avg_N(l) for l in theor_lambdas]
    sync_avg_D_theor = [avg_D(l, sync_mode=True) for l in theor_lambdas]

    plt.figure(figsize=(8, 5))

    plt.plot(lmbds, outs, 'o-', label="Интенсивность выхода (симуляция)")
    plt.plot(theor_lambdas, theor_lambdas, '--', label="Теоретическая (λ_out = λ)")
    plt.title(f"Интенсивность выходного потока BEB. M={M}, T={T}, W=[{minW},{maxW}]")
    plt.xlabel("$\\lambda_{вх}$")
    plt.ylabel("$\\lambda_{вых}=$(успехи/слот)")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, max(lmbds) * 1.05)
    plt.ylim(0, max(outs) * 1.1)

    # plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(lmbds, delays, 'o-', label="$d_{avg}$ (симуляция)")
    if M == 1:
        plt.plot(theor_lambdas, sync_avg_D_theor, linestyle="--", label="Теоретическая M/D/1 (синхр.)")
    plt.title(f"Средняя задержка BEB. M={M}, T={T}, W=[{minW},{maxW}]")
    plt.xlabel("$\\lambda_{вх}$")
    plt.ylabel("$d_{avg}$")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, max(lmbds) * 1.05)
    stable_delays = [d for l, d in zip(lmbds, delays) if l < 0.95]
    if stable_delays:
        plt.ylim(0, max(stable_delays) * 1.5)

    # plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(lmbds, abonents, 'o-', label="$N_{avg}$ (симуляция)")
    if M == 1:
        plt.plot(theor_lambdas, avg_N_theor, linestyle="--", label="Теоретическая M/D/1")
    plt.title(f"Среднее число пакетов в системе BEB. M={M}, T={T}, W=[{minW},{maxW}]")
    plt.xlabel("$\\lambda_{вх}$")
    plt.ylabel("$N_{avg}$")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, max(lmbds) * 1.05)
    stable_N = [n for l, n in zip(lmbds, abonents) if l < 0.95]
    if stable_N:
        plt.ylim(0, max(stable_N) * 1.5)

    plt.tight_layout()
    # plt.show()


def avg_N(_lambda=0.3):
    if _lambda >= 1.0:
        return float('inf')
    return (_lambda * (2 - _lambda)) / (2 * (1 - _lambda))


def avg_D(_lambda=0.3, sync_mode=False):
    if _lambda >= 1.0:
        return float('inf')
    d = avg_N(_lambda=_lambda) / _lambda
    return d + 0.5 if sync_mode else d


def run_and_log_successes(M: int, lambda_total: float, T_total: int, minW: int, maxW: int,
                          warmup_ratio: float = 0.6):
    ch = SystemChannel(M, lambda_total, minW, maxW)
    success_log = []

    T_warmup = int(T_total * warmup_ratio)
    for _ in range(T_total):
        t = ch.time

        # Генерация пакетов и планирование
        for a in ch.abonents:
            a.generate_arrivals(t)
        for a in ch.abonents:
            a.schedule_head_if_needed(t)

        # Передача
        senders = [a for a in ch.abonents if a.has_scheduled_in_slot(t)]

        if len(senders) == 1:
            sender = senders[0]
            sender.notify_result('SUCCESS', t)
            # сохраняем только после прогрева
            if t >= T_warmup:
                success_log.append((t, sender.id))
        elif len(senders) > 1:
            for s in senders:
                s.notify_result('COLLISION', t)

        ch.time += 1

    return success_log, T_warmup, T_total


def plot_success_scatter(success_log, lambda_total, minW, maxW, T_warmup, T_total):
    if not success_log:
        print(f"Нет успешных передач для $\\lambda_{{вх}}$={lambda_total}")
        return

    times, ids = zip(*success_log)
    plt.figure(figsize=(8, 5))
    plt.scatter(times, ids, s=8, alpha=0.6)
    plt.xlabel("Номер слота T")
    plt.ylabel("ID абонента (успешная передача)")
    plt.title(f"Успешные передачи: $\\lambda_{{вх}}$={lambda_total}, W=[{minW},{maxW}]")
    plt.axvline(T_warmup, color='red', linestyle='--', label='конец прогрева')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    # plt.show()


def run_and_log_all_transmissions(M: int, lambda_total: float, T_total: int,
                                  minW: int, maxW: int, warmup_ratio: float = 0.6):
    ch = SystemChannel(M, lambda_total, minW, maxW)
    transmissions = []

    T_warmup = int(T_total * warmup_ratio)
    for _ in range(T_total):
        t = ch.time

        for a in ch.abonents:
            a.generate_arrivals(t)
        for a in ch.abonents:
            a.schedule_head_if_needed(t)

        senders = [a for a in ch.abonents if a.has_scheduled_in_slot(t)]

        if len(senders) == 0:
            result = 'EMPTY'
        elif len(senders) == 1:
            result = 'SUCCESS'
            senders[0].notify_result('SUCCESS', t)
        else:
            result = 'COLLISION'
            for s in senders:
                s.notify_result('COLLISION', t)

        if t >= T_warmup:
            for s in senders:
                transmissions.append((t, s.id, result))

        ch.time += 1

    return transmissions, T_warmup, T_total


def plot_all_transmissions(transmissions, lambda_total, minW, maxW, T_warmup, T_total):
    if not transmissions:
        print(f"Нет событий для $\\lambda_{{вх}}$={lambda_total}")
        return

    times_success = [t for (t, _, r) in transmissions if r == 'SUCCESS']
    ids_success = [i for (_, i, r) in transmissions if r == 'SUCCESS']
    times_coll = [t for (t, _, r) in transmissions if r == 'COLLISION']
    ids_coll = [i for (_, i, r) in transmissions if r == 'COLLISION']

    plt.figure(figsize=(8, 5))
    plt.scatter(times_success, ids_success, s=10, color='blue', alpha=0.6, label='SUCCESS')
    plt.scatter(times_coll, ids_coll, s=10, color='red', alpha=0.4, label='COLLISION')
    plt.axvline(T_warmup, color='green', linestyle='--', label='конец прогрева')
    plt.xlabel("Номер слота T")
    plt.ylabel("ID абонента (все передачи)")
    plt.title(f"Все передачи (успехи и коллизии): $\\lambda_{{вх}}$={lambda_total}, W=[{minW},{maxW}]")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()



if __name__ == "__main__":
    # Была подтверждена работоспособность на паре

    M = 10
    T = 150_000
    minW = 1
    maxW = 1024
    max_queue_size = 150_000
    trials = 1

    lambdas = np.linspace(0.000001, 0.6, 13)

    experiment_plot(M=M, lambdas=lambdas, T=T, minW=minW, maxW=maxW,
                    max_queue_size=max_queue_size, trials=trials)


    # ------------------ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ--------------------
    T_total = 150_000
    count_slots = 2000
    warmup_ratio = 1 - count_slots / T_total
    for lam in [0.1, 0.6]:
        print(f"\nСимуляция для λ={lam}...")
        success_log, T_warmup, T_total = run_and_log_successes(
            M, lam, T_total=T_total, minW=minW, maxW=maxW, warmup_ratio=warmup_ratio
        )
        plot_success_scatter(success_log, lam, minW, maxW, T_warmup, T_total)

    for lam in [0.1, 0.6]:
        print(f"\nСимуляция (все передачи) для λ={lam}...")
        transmissions, T_warmup, T_total = run_and_log_all_transmissions(
            M, lam, T_total=T_total, minW=minW, maxW=maxW, warmup_ratio=warmup_ratio
        )
        plot_all_transmissions(transmissions, lam, minW, maxW, T_warmup, T_total)


    plt.show()
    input("Press any key for process finish...")
