from collections import deque
from typing import Deque, List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


class ALOHApAbonent:
    def __init__(self, id_abonent, p, lam, mode_always_first=False ,buff_size=10**6):
        self.count_translated = 0
        self.total_delays = 0
        self.id_abonent = id_abonent
        self.p = p
        self.id_arrival = 0
        self.lam = lam
        self.mode_always_first = mode_always_first
        self.buff_size = buff_size
        self.buffer: Dict = deque()
        self.discarded = 0  # Число отброшенных заявок
        self.delays = 0  # Общее время задержки заявок
        self.processed = 0  # Число завершенных заявок
        self.arrivals_info: Dict = dict()
        self.id_last_translated = None

    def start_slot_process(self, slot_now):
        arrival_in_prev_slot = np.random.poisson(lam=self.lam)

        while arrival_in_prev_slot > 0:
            if len(self.buffer) < self.buff_size:
                self.buffer.append({"id_arrival" : self.id_arrival,
                                    "slot_in" : slot_now,
                                    "slot_out" : None})
            else:
                self.discarded += 1
                print(f"DISCARDED "
                      f"id_abonent: {self.id_abonent}; " 
                      f"id_arrival: {self.id_arrival}")

            arrival_in_prev_slot -= 1
            self.id_arrival += 1

        # Решение о том, будет ли передача в данном окне
        if not self.buffer:
            return False
        
        if ((self.mode_always_first and len(self.buffer) == 1 and self.id_last_translated != self.buffer[0]["id_arrival"])
                or (self.buffer and np.random.random() < self.p)):
            
            self.id_last_translated = self.buffer[0]["id_arrival"]
            return True

        return False

    def end_slot_process(self, channel_status, slot_now):
        if channel_status == "SUCCESSFUL" and self.buffer:
            self.count_translated += 1

            arrival_info = self.buffer.popleft()
            arrival_info["slot_out"] = slot_now

            self.arrivals_info.update({arrival_info["id_arrival"] :
                                       {"slot_in" : arrival_info["slot_in"],
                                        "slot_out" : arrival_info["slot_out"]}
                                       })

            self.total_delays += slot_now - arrival_info["slot_in"] + 1.5

    def get_count_arrival(self):
        return len(self.buffer)

    def get_avg_delay(self):
        if self.count_translated == 0:
            return np.inf
        return self.total_delays / self.count_translated

    def get_id(self):
        return self.id_abonent

    def print_abonent_info(self):
        delays = 0
        last_out = None

        i = 0
        for key, value in self.arrivals_info.items():
            if value["slot_out"]:
                delays += value["slot_out"] - value["slot_in"]
                last_out = value["slot_in"]

            if i > 0.9 * len(self.arrivals_info):
                print(f"id_abonent: {self.id_abonent}; id_arrival: {key}; {value}")

            i += 1

#         print(f"""
# ID_ABONENT: {self.id_abonent}
# LAST_OUT_SLOT: {last_out}
# AVG_DELAY: {delays / last_out}
# """)


class ALOHApChannel:
    def __init__(self, M, lam, p=None, count_slots=10**5, buff_size=10**6):
        self.M = M
        self.p = 1 / M if not p else p

        self.lam = lam
        self.lam_abonent = self.lam / M

        self.count_slots = count_slots
        self.buff_size = buff_size

        self.current_slot = 0
        self.count_translated = 0
        self.total_arrival_in_system = 0
        self.slots_info = [[] for _ in range(self.count_slots)]

        self.abonents = [ALOHApAbonent(id_abonent,
                                       self.p,
                                       self.lam_abonent,
                                       buff_size=buff_size)
                         for id_abonent in range(M)]


    def modelling(self):
        for slot in range(self.count_slots):
            count_transletes = 0

            id_abonents = []
            for abonent in self.abonents:
                this_abonent_translated = abonent.start_slot_process(slot)

                if this_abonent_translated:
                    id_abonents.append(abonent.get_id())
                    count_transletes += 1

            self.slots_info[slot] = id_abonents
            if count_transletes == 0:
                channel_status = "EMPTY"
            elif count_transletes == 1:
                channel_status = "SUCCESSFUL"
                self.count_translated += 1
            else:
                channel_status = "COLLISION"

            for abonent in self.abonents:
                abonent.end_slot_process(channel_status, slot)
                self.total_arrival_in_system += abonent.get_count_arrival()

    def get_lambda_out(self):
        return self.count_translated / self.count_slots

    def get_avg_delay(self):
        total_delay = 0

        for abonent in self.abonents:
            total_delay += abonent.get_avg_delay()

        return total_delay / self.M

    def get_avg_arrival_in_system(self):
        return self.total_arrival_in_system / self.count_slots

    def print_slots_info(self):
        print("___PRINTING_SLOTS_INFO___")
        for i, slot_info in enumerate(self.slots_info[-100::]):
            print(f"SLOT: {i}; info: {slot_info}")

    def print_abonents_info(self):
        for abonent in self.abonents:
            abonent.print_abonent_info()


def run_experiments():
    M = 10
    p = 1 / M
    buff_size = 100_000
    slots = 200_000
    lambdas = np.linspace(0.0001, 4.999, 11)

    lambda_out_list = []
    avg_N_list = []
    avg_delay_list = []

    for lam in lambdas:
        model = ALOHApChannel(M=M, lam=lam, p=p, count_slots=slots, buff_size=buff_size)
        model.modelling()

        # model.print_slots_info()
        # model.print_abonents_info()

        lambda_out_list.append(model.get_lambda_out())
        avg_N_list.append(model.get_avg_arrival_in_system())
        avg_delay_list.append(model.get_avg_delay())

    plt.figure()
    plt.plot(lambdas, lambda_out_list, 'o-', label='λ_out(λ)')
    plt.xlabel('Входная интенсивность λ')
    plt.ylabel('Выходная интенсивность λ_out')
    plt.title(f'{M} Абонентов. Интенсивность выходного потока')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(lambdas, avg_N_list, 's-', label='N(λ)')
    plt.xlabel('Входная интенсивность λ')
    plt.ylabel('Среднее число заявок в системе')
    plt.title(f'{M} Абонентов. Среднее число заявок N(λ)')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(lambdas, avg_delay_list, 'd-', label='d(λ)')
    plt.xlabel('Входная интенсивность λ')
    plt.ylabel('Среднее время задержки')
    plt.title(f'{M} Абонентов. Средняя задержка d(λ)')
    plt.grid(True)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    run_experiments()
