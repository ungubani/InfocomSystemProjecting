"""
2.5. Моделирование алгоритма с возвратом
"""

import random
import matplotlib.pyplot as plt

def algo_go_back_n(tau, p, K):
    status = []
    i = 0

    while i < K:
        if random.random() < p:
            status.extend([-1] * (tau + 1))

        else:
            status.append(1)
            i += 1

    time = list(range(1, len(status) + 1))
    return time, status


def visualize_go_back_n(tau, p, K):
    time, status = algo_go_back_n(tau, p, K)
    print(f"$\\tau$Eta_theor=", (1 - p) / (1 + p * tau))
    print("Eta_exp=", status.count(1) / len(status))

    plt.figure(figsize=(12, 6))
    plt.step(time, status, where='post', label='Статус приёма', linewidth=2)
    plt.title(f'2.5 Алгоритм с возвратом. Временная диаграмма \n$\\tau$={tau}, p={p}')
    plt.xlabel('Время')
    plt.ylabel('Событие на приёмнике')
    plt.yticks([-1, 1], ['Уничтожено', 'Принято'])
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    tau = 4
    p = 0.3
    K = 3000

    visualize_go_back_n(tau, p, K)
