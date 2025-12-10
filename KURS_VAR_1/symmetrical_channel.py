from typing import List
import random


def errors_vector_generator(length: int, probability: float) -> List[int]:
    errors = []

    for i in range(length):
        error = 0
        if random.random() < probability:
            error = 1
            has_error = True

        errors.append(error)

    return errors


def adding_errors(code_word: List[int], errors: List[int]) -> List[int]:
    if len(code_word) != len(errors):
        raise ValueError("Длины кодового слова  вектора ошибок не равны")

    channel_word = []
    for i in range(len(code_word)):
        channel_word.append(code_word[i] ^ errors[i])

    return channel_word
