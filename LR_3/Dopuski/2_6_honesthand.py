import numba as nb
import numpy as np
import math
import matplotlib.pyplot as plt


class SystemMD1:
    def __init__(self, _lambda=0.3, _number_expiriences=1_000_000, _sync_mode=False):
        self.__mean_D = 0
        self.__mean_N = 0
        self.__N_stat = dict()  # Ключ - какое-то целое N, значение - время, в течение которого оно сохранялось

        self.__sync_mode = _sync_mode
        self.__lambda = _lambda
        self.__number_expiriences = _number_expiriences

        self.__global_time_requests_in = np.zeros(self.__number_expiriences)
        self.__global_time_requests_out = np.zeros(self.__number_expiriences)
        self.__req_wait_time = np.zeros(self.__number_expiriences)

        self.__modelling()

    def __modelling(self):
        if self.__lambda == 0:
            return

        global_time_in = 0
        for i_req in range(self.__number_expiriences):
            time_to_next_req = np.random.exponential(1 / self.__lambda)

            global_time_in += time_to_next_req
            self.__global_time_requests_in[i_req] = global_time_in

        global_time_out = 0  # Последняя вышедшая заявка
        for i_req in range(self.__number_expiriences):
            upper_bound_for_empty = self.__global_time_requests_in[i_req] + 1

            if self.__sync_mode:
                upper_bound_for_empty = math.ceil(upper_bound_for_empty)

            global_time_out = max(global_time_out + 1, upper_bound_for_empty)  # Для синхронной второй параметр будет ceil(...) + 1
            # if self.__sync_mode and global_time_out != int(global_time_out): print(global_time_out)
            self.__global_time_requests_out[i_req] = global_time_out
            self.__req_wait_time[i_req] = self.__global_time_requests_out[i_req] - self.__global_time_requests_in[i_req]

        if self.__sync_mode: self.__sync_calculate_mean_N()
        else: self.__async_calculate_mean_N()

        self.__mean_D = self.__req_wait_time.mean()

    def __sync_calculate_mean_N(self):
        # Двойной указатель
        i_in = 0
        j_out = 0

        N_last = 0
        global_time_last = 0

        while j_out < self.__number_expiriences:
            N_in_this_slot = 0

            while (i_in < self.__number_expiriences and math.ceil(self.__global_time_requests_in[i_in]) <
                   self.__global_time_requests_out[j_out]):
                N_in_this_slot += 1

                global_time_last = math.ceil(self.__global_time_requests_in[i_in])
                i_in += 1

            N_last += N_in_this_slot

            if N_last not in self.__N_stat:
                self.__N_stat[N_last] = 0

            time_N_last = self.__global_time_requests_out[j_out] - global_time_last
            self.__N_stat[N_last] += time_N_last

            global_time_last = self.__global_time_requests_out[j_out]
            N_last -= 1
            j_out += 1

        for N, N_time in self.__N_stat.items():
            N_prob = N_time / self.__global_time_requests_out[self.__number_expiriences - 1]
            self.__mean_N += N * N_prob

    def __async_calculate_mean_N(self):
        # Двойной указатель
        i_in = 0
        j_out = 0

        N_last = 0
        global_time_last = 0

        while j_out < self.__number_expiriences:
            while (i_in < self.__number_expiriences and self.__global_time_requests_in[i_in] <=
                   self.__global_time_requests_out[j_out]):
                if N_last not in self.__N_stat:
                    self.__N_stat[N_last] = 0

                self.__N_stat[N_last] += self.__global_time_requests_in[i_in] - global_time_last

                global_time_last = self.__global_time_requests_in[i_in]
                N_last += 1
                i_in += 1

            if N_last not in self.__N_stat:
                self.__N_stat[N_last] = 0

            time_N_last = self.__global_time_requests_out[j_out] - global_time_last
            self.__N_stat[N_last] += time_N_last

            global_time_last = self.__global_time_requests_out[j_out]
            N_last -= 1
            j_out += 1

        for N, N_time in self.__N_stat.items():
            N_prob = N_time / self.__global_time_requests_out[self.__number_expiriences - 1]
            self.__mean_N += N * N_prob

    def get_mean_D(self):
        return self.__mean_D

    def get_mean_N(self):
        return self.__mean_N


def avg_N(_lambda=0.3):
    return (_lambda * (2 - _lambda)) / (2 * (1 - _lambda))

def avg_D(_lambda=0.3, sync_mode=False):
    d = avg_N(_lambda=_lambda) / _lambda
    return d + 0.5 if sync_mode else d


if __name__ == "__main__":
    number_experiences = 10_000_000

    count_points = 21
    # lambdas = [0.99]
    lambdas = np.linspace(0.0001, 0.9, count_points)

    async_avg_N_modelling = np.zeros(count_points)
    async_avg_D_modelling = np.zeros(count_points)

    sync_avg_N_modelling = np.zeros(count_points)
    sync_avg_D_modelling = np.zeros(count_points)

    avg_N_theor = np.zeros(count_points)
    async_avg_D_theor = np.zeros(count_points)
    sync_avg_D_theor = np.zeros(count_points)

    for i in nb.prange(count_points):
        print(f"Моделирование #{i + 1}/{count_points}")
        _lambda = lambdas[i]
        async_system = SystemMD1(_lambda=_lambda, _number_expiriences=number_experiences, _sync_mode=False)
        async_avg_N_modelling[i] = async_system.get_mean_N()
        async_avg_D_modelling[i] = async_system.get_mean_D()

        sync_system = SystemMD1(_lambda=_lambda, _number_expiriences=number_experiences, _sync_mode=True)
        sync_avg_N_modelling[i] = sync_system.get_mean_N()
        sync_avg_D_modelling[i] = sync_system.get_mean_D()

        avg_N_theor[i] = avg_N(_lambda)
        async_avg_D_theor[i] = avg_D(_lambda=_lambda, sync_mode=False)
        sync_avg_D_theor[i] = avg_D(_lambda=_lambda, sync_mode=True)

    plt.figure()
    plt.title(f"Среднее число абонентов в системе для СМО M|D|1")
    plt.xlabel(f"$\lambda$")
    plt.ylabel(f"N($\lambda$)")

    plt.plot(lambdas, async_avg_N_modelling, label=f"Моделирование, асинхр.")
    plt.plot(lambdas, sync_avg_N_modelling, label=f"Моделирование, синхр.")
    plt.plot(lambdas, avg_N_theor, linestyle=":", label=f"Теоретическое")

    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.title("Среднее время нахождения заявки в системе для СМО M|D|1")
    plt.xlabel("$\lambda$")
    plt.ylabel("D($\lambda$)")
    plt.plot(lambdas, async_avg_D_modelling, label="Моделирование, асинхр.")
    plt.plot(lambdas, sync_avg_D_modelling, label="Моделирование, синхр.")
    plt.plot(lambdas, async_avg_D_theor, linestyle=":", label="Теоретическое (асинхр.)")
    plt.plot(lambdas, sync_avg_D_theor, linestyle="--", label="Теоретическое (синхр.)")
    plt.legend()
    plt.grid(True)
    plt.show()

print(0)
