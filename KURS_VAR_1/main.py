import coder
import symmetrical_channel

from matplotlib import pyplot as plt
from typing import List
import math
import numpy as np
import random


def has_errors(errors: List[int]) -> True:
    return sum(errors) != 0


def message_random_generator(length: int, probability: float = 0.5) -> List[int]:
    # probability - вероятность "1" на i-й позиции

    message = []
    for i in range(length):
        if random.random() < probability:
            message.append(1)
        else:
            message.append(0)

    return message


if __name__ == "__main__":
    _GENERATING_POLYNOMIAL = [1, 1, 0, 1]  # Порождающий многочлен (младший индекс соответствует младшей степени)
    _Pe_BIT = np.linspace(0, 1, 11)  # Вероятности ошибки на бит (избегаем 0)

    epsilon = 0.005  # 0.01 - 22500 экспериментов; 0.005 - 90000 экспериментов
    numbers_experiments = math.ceil(9 / (4 * epsilon ** 2))
    print(f"Точность = {epsilon}, количество экспериментов = {numbers_experiments}")

    lengths = [3, 4, 6, 11, 29]  # Длина кодируемой последовательности

    plt.figure(figsize=(12, 8))

    for length in lengths:
        print(f"Обрабатывается длина сообщения {length}")
        pe_values = []  # Сброс списка для текущей длины

        for pe in _Pe_BIT:
            decoder_error_counter = 0  # Количество ошибок декодирования

            for _ in range(numbers_experiments):
                # Источник
                source_message = message_random_generator(length)

                # Кодер
                code_word = coder.encode(source_message, _GENERATING_POLYNOMIAL)

                # Канальный уровень
                errors = symmetrical_channel.errors_vector_generator(len(code_word), pe)
                channel_has_errors = has_errors(errors)
                channel_word = symmetrical_channel.adding_errors(code_word, errors)

                # Декодер
                decoder_message, decoder_decision = coder.decoder(channel_word, _GENERATING_POLYNOMIAL)

                if decoder_decision != channel_has_errors:
                    decoder_error_counter += 1

            pe_value = decoder_error_counter / numbers_experiments  # Вероятность ошибки декодирования
            pe_values.append(pe_value)

        plt.plot(_Pe_BIT, pe_values, marker=".", label=f"Длина = {length}")
        # print(pe_values.index(max(pe_values)))

    # plt.yscale("log")
    plt.title(f"Вероятность ошибки декодирования CRC-{coder.degree_polynomial(_GENERATING_POLYNOMIAL)}")
    plt.xlabel("Вероятность ошибки на бит p")
    plt.ylabel("Вероятность ошибки декодирования Pe")

    plt.legend()
    plt.grid()
    plt.show()

