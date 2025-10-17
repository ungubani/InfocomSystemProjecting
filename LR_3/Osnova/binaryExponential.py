import numpy as np
import statistics
import queue

class Abonent:
    def __init__(self, _lambda, buffer_capacity=None):
        self.translate_flag = False
        self.last_arrival_time = 0
        self._lambda = _lambda

        if not buffer_capacity:
            self.buffer_capacity = 5 * 10**6 + 1  # Какое-то большое число
        else:
            self.buffer_capacity = buffer_capacity + 1

        self.buffer = queue.Queue(maxsize=self.buffer_capacity)

    def get_next(self, start_slot):
        interval = np.random.exponential(1 / self._lambda)
        while self.last_arrival_time + interval < start_slot:
            if len(self.buffer) < self.buffer_capacity:
                self.last_arrival_time += interval
                self.buffer.put(self.last_arrival_time)

        if not self.translate_flag:
            self.translate_flag = True
            return self.buffer.get()

