"""
Моделирование системы передачи с возвратом и CRC-кодированием
"""

import random
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import List


# ========== Функции для CRC-кодирования и канала ==========

def degree_polynomial(poly: List[int]) -> int:
    """Вычисляет степень многочлена"""
    return len(poly) - 1


def polynomial_mod_2_division(dividend: List[int], divisor: List[int]) -> List[int]:
    """Деление многочленов по модулю 2"""
    dividend = dividend.copy()
    divisor_degree = degree_polynomial(divisor)

    while len(dividend) >= len(divisor):
        if dividend[0] == 1:
            for i in range(len(divisor)):
                dividend[i] ^= divisor[i]
        dividend.pop(0)

    return dividend


def encode(message: List[int], generating_polynomial: List[int]) -> List[int]:
    """Кодирование сообщения с помощью CRC"""
    poly_degree = degree_polynomial(generating_polynomial)
    extended_message = message + [0] * poly_degree
    remainder = polynomial_mod_2_division(extended_message, generating_polynomial)

    # Добавляем остаток к исходному сообщению
    return message + remainder


def decoder(received_word: List[int], generating_polynomial: List[int]) -> (List[int], bool):
    """Декодирование с проверкой CRC"""
    remainder = polynomial_mod_2_division(received_word.copy(), generating_polynomial)
    has_error = any(remainder)  # Если остаток не нулевой - есть ошибка

    # Возвращаем исходное сообщение (без контрольных битов) и флаг ошибки
    message_length = len(received_word) - degree_polynomial(generating_polynomial)
    decoded_message = received_word[:message_length] if message_length > 0 else []

    return decoded_message, has_error


def errors_vector_generator(length: int, probability: float) -> List[int]:
    """Генератор вектора ошибок"""
    return [1 if random.random() < probability else 0 for _ in range(length)]


def adding_errors(code_word: List[int], errors: List[int]) -> List[int]:
    """Добавление ошибок к кодовому слову"""
    if len(code_word) != len(errors):
        raise ValueError("Длины кодового слова и вектора ошибок должны совпадать")

    return [bit ^ error for bit, error in zip(code_word, errors)]


def has_errors(errors: List[int]) -> bool:
    """Проверка наличия ошибок"""
    return sum(errors) != 0


def message_random_generator(length: int, probability: float = 0.5) -> List[int]:
    """Генератор случайных сообщений"""
    return [1 if random.random() < probability else 0 for _ in range(length)]


# ========== Модифицированный алгоритм с возвратом ==========

def algo_go_back_n(tau, p_bit, K, message_length, generating_polynomial):
    """
    Модифицированный алгоритм с возвратом, использующий CRC-проверку

    Parameters:
    tau - задержка
    p_bit - вероятность ошибки на бит
    K - количество успешных передач
    message_length - длина информационного сообщения
    generating_polynomial - порождающий многочлен CRC
    epsilon - точность (определяет количество экспериментов)
    """

    status = []
    successful_transmissions = 0
    total_transmissions = 0

    while successful_transmissions < K:
        # Генерация и кодирование сообщения
        source_message = message_random_generator(message_length)
        code_word = encode(source_message, generating_polynomial)

        # Передача через канал с ошибками
        errors = errors_vector_generator(len(code_word), p_bit)
        channel_word = adding_errors(code_word, errors)

        # Декодирование и проверка CRC
        decoder_message, decoder_decision = decoder(channel_word, generating_polynomial)

        total_transmissions += 1

        if decoder_decision:  # Обнаружена ошибка
            status.extend([-1] * (tau + 1))
        else:  # Передача успешна
            status.append(1)
            successful_transmissions += 1

    time = list(range(1, len(status) + 1))
    return time, status, total_transmissions


def visualize_go_back_n(tau, p_bit, K, message_length, generating_polynomial):
    """Визуализация модифицированного алгоритма с возвратом"""
    time, status, total_transmissions = algo_go_back_n(tau, p_bit, K, message_length, generating_polynomial)

    # Расчет эффективности
    successful_count = status.count(1)
    total_slots = len(status)
    efficiency = successful_count / total_slots if total_slots > 0 else 0
    actual_error_rate = (total_transmissions - K) / total_transmissions if total_transmissions > 0 else 0

    print(f"Теоретическая эффективность (без учета CRC): {(1 - p_bit) / (1 + p_bit * tau):.4f}")
    print(f"Экспериментальная эффективность: {efficiency:.4f}")
    print(f"Фактическая вероятность ошибки пакета: {actual_error_rate:.4f}")
    print(f"Всего передач: {total_transmissions}, успешных: {K}")
    print(f"Всего временных слотов: {total_slots}")

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.step(time, status, where='post', label='Статус приёма', linewidth=2)
    plt.title(f'Алгоритм с возвратом с CRC-проверкой\n'
              f'$\\tau$={tau}, p_bit={p_bit}, Длина сообщения={message_length}')
    plt.xlabel('Время (слоты)')
    plt.ylabel('Событие на приёмнике')
    plt.yticks([-1, 1], ['Ошибка CRC', 'Успех'])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Параметры системы
    tau = 4  # Задержка
    p_bit = 0.3  # Вероятность ошибки на бит (фиксированная)
    K = 10_000  # Количество успешных передач
    message_length = 8  # Длина информационного сообщения
    generating_polynomial = [1, 1, 0, 1]  # Порождающий многочлен CRC-3
    epsilon = 0.05  # Точность

    print(f"Параметры системы:")
    print(f"  Задержка (tau): {tau}")
    print(f"  Вероятность ошибки на бит: {p_bit}")
    print(f"  Количество успешных передач: {K}")
    print(f"  Длина сообщения: {message_length}")
    print(f"  CRC многочлен: {generating_polynomial}")
    print(f"  Точность (epsilon): {epsilon}")
    print()

    visualize_go_back_n(tau, p_bit, K, message_length, generating_polynomial)