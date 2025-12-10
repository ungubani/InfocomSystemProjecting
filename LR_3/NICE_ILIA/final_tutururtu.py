import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque


class Message:
    def __init__(self, arrival_time):
        self.arrival_time = arrival_time  # Время поступления сообщения
        self.attempt_count = 0  # Количество попыток передачи


class Abonent:
    def __init__(self, queue_size=-1):
        self.queue_size = queue_size  # -1 означает неограниченную очередь
        self.queue = deque(maxlen=queue_size if queue_size != -1 else None)

    def add_message(self, msg):
        if self.queue_size == -1 or len(self.queue) < self.queue_size:
            self.queue.append(msg)
            return True
        return False


class ALOHASimulation:
    def __init__(self, M, lambda_val, p, case_type, queue_size=-1, simulation_windows=100000):
        self.M = M  # Количество абонентов
        self.lambda_val = lambda_val  # Интенсивность входящего трафика
        self.p = p  # Вероятность передачи
        self.case_type = case_type  # 'a' или 'b'
        self.queue_size = queue_size  # Размер очереди
        self.simulation_windows = simulation_windows  # Количество окон имитации

        # Инициализация абонентов
        self.abonents = [Abonent(queue_size) for _ in range(M)]

        # Статистика
        self.total_delay = 0.0
        self.total_messages_transmitted = 0
        self.total_messages_in_system = 0.0

        # Для сбора данных по всем окнам
        self.delays = []
        self.messages_in_system = []
        self.transmission_attempts = []

    def simulate(self):
        for n in range(self.simulation_windows):
            transmissions = 0  # Счетчик попыток передачи
            transmitting_abonent = None  # Абонент, пытающийся передать

            # Проверка очередей всех абонентов
            for abonent in self.abonents:
                if abonent.queue:
                    msg = abonent.queue[0]  # Первое сообщение в очереди
                    transmit = False

                    # Логика передачи в зависимости от case_type
                    if self.case_type == 'a':
                        # Случай 'a': передача с вероятностью p
                        transmit = random.random() < self.p
                    else:
                        # Случай 'b': первая попытка передается всегда, последующие - с вероятностью p
                        if msg.attempt_count == 0 and len(abonent.queue) == 1:
                            transmit = True
                        else:
                            transmit = random.random() < self.p

                    if transmit:
                        transmissions += 1
                        transmitting_abonent = abonent
                        msg.attempt_count += 1

            # Обработка успешной передачи (только одна попытка в окне)
            if transmissions == 1 and transmitting_abonent:
                msg = transmitting_abonent.queue.popleft()  # Удаляем сообщение из очереди
                delay = (n + 1) - msg.arrival_time  # Вычисляем задержку
                self.total_delay += delay
                self.total_messages_transmitted += 1
                self.delays.append(delay)

            # Генерация новых сообщений
            for abonent in self.abonents:
                for _ in range(np.random.poisson(self.lambda_val / self.M)):
                    arrival_time = n + random.random()  # Случайное время в пределах окна
                    new_msg = Message(arrival_time)
                    abonent.add_message(new_msg)
                # if random.random() < self.lambda_val / self.M:
                #     arrival_time = n + random.random()  # Случайное время в пределах окна
                #     new_msg = Message(arrival_time)
                #     abonent.add_message(new_msg)

            # Сбор статистики сообщений в системе
            total_messages = sum(len(abonent.queue) for abonent in self.abonents)
            self.total_messages_in_system += total_messages
            self.messages_in_system.append(total_messages)
            self.transmission_attempts.append(transmissions)

    def get_results(self):
        avg_delay = self.total_delay / self.total_messages_transmitted if self.total_messages_transmitted > 0 else 0
        avg_messages = self.total_messages_in_system / self.simulation_windows
        output_rate = self.total_messages_transmitted / self.simulation_windows

        return {
            'average_delay': avg_delay,
            'average_messages': avg_messages,
            'output_rate': output_rate,
            'delays': self.delays,
            'messages_in_system': self.messages_in_system,
            'transmission_attempts': self.transmission_attempts
        }


def run_comparison(M, p, queue_size=-1, lambdas=np.linspace(0.01, 0.4, 31), simulation_windows = 100_000):
    # Параметры симуляции
    lambdas = lambdas  # Значения lambda для исследования
    simulation_windows = simulation_windows  # Количество окон имитации

    # Теоретическое значение λ_кр = 1/e
    lambda_critical = M * p * (1 - p) ** (M - 1)

    # Результаты для случаев 'a' и 'b'
    results_a = {'delay': [], 'messages': [], 'output': []}
    results_b = {'delay': [], 'messages': [], 'output': []}

    print("Запуск симуляции для случая A...")
    for i, lambda_val in enumerate(lambdas):
        print(f"Прогресс: {i + 1}/{len(lambdas)} (λ={lambda_val:.3f})")
        sim = ALOHASimulation(M, lambda_val, p, 'a', queue_size, simulation_windows)
        sim.simulate()
        res = sim.get_results()
        results_a['delay'].append(res['average_delay'])
        results_a['messages'].append(res['average_messages'])
        results_a['output'].append(res['output_rate'])

    print("Запуск симуляции для случая B...")
    for i, lambda_val in enumerate(lambdas):
        print(f"Прогресс: {i + 1}/{len(lambdas)} (λ={lambda_val:.3f})")
        sim = ALOHASimulation(M, lambda_val, p, 'b', queue_size, simulation_windows)
        sim.simulate()
        res = sim.get_results()
        results_b['delay'].append(res['average_delay'])
        results_b['messages'].append(res['average_messages'])
        results_b['output'].append(res['output_rate'])

    # Построение отдельных графиков

    # График 1: Средняя задержка
    plt.figure(figsize=(7, 4.5))
    plt.plot(lambdas, results_a['delay'], 'b-', linewidth=2, label=f'Случай A (всегда p={p:.3f})')
    plt.plot(lambdas, results_b['delay'], 'r-', linewidth=2, label=f'Случай B (первая=1, потом p={p:.3f})')
    plt.axvline(x=lambda_critical, color='k', linestyle='--', alpha=0.7, label='$\\lambda_{crit.}=M*p*(1-p)^{(M-1)}$')
    plt.xlabel('Интенсивность входного потока (λ)')
    plt.ylabel('Средняя задержка')
    plt.title(f'Зависимость средней задержки от интенсивности входного потока (M={M})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # График 2: Среднее число сообщений в системе
    plt.figure(figsize=(7, 4.5))
    plt.plot(lambdas, results_a['messages'], 'b-', linewidth=2, label=f'Случай A (всегда p={p:.3f})')
    plt.plot(lambdas, results_b['messages'], 'r-', linewidth=2, label=f'Случай B (первая=1, потом p={p:.3f})')
    plt.axvline(x=lambda_critical, color='k', linestyle='--', alpha=0.7, label='$\\lambda_{crit.}=M*p*(1-p)^{(M-1)}$')
    plt.xlabel('Интенсивность входного потока (λ)')
    plt.ylabel('Среднее число сообщений в системе')
    plt.title(f'Зависимость среднего числа сообщений от интенсивности входного потока (M={M})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # График 3: Выходная интенсивность
    plt.figure(figsize=(7, 4.5))
    plt.plot(lambdas, results_a['output'], 'b-', linewidth=2, label=f'Случай A (всегда p={p:.3f})')
    plt.plot(lambdas, results_b['output'], 'r-', linewidth=2, label=f'Случай B (первая=1, потом p={p:.3f})')
    plt.plot(lambdas, lambdas, 'g--', linewidth=2, label='Входная интенсивность')
    plt.axvline(x=lambda_critical, color='k', linestyle='--', alpha=0.7, label='$\\lambda_{crit.}=M*p*(1-p)^{(M-1)}$')
    plt.xlabel('Интенсивность входного потока (λ)')
    plt.ylabel('Интенсивность выходного потока')
    plt.title(f'Зависимость выходной интенсивности от входной (M={M})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Основные параметры моделирования - можно легко менять
    M = 15  # Количество абонентов
    p = 0.1  # Вероятность передачи

    # run_comparison(M, p)

    # ИЛИ дополнить так:
    lambdas = np.linspace(0.3, 0.4, 11)
    simulation_windows = 200_000
    queue_size = -1

    run_comparison(M, p, queue_size=queue_size, lambdas=lambdas, simulation_windows=simulation_windows)
