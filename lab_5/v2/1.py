import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, weibull_min, gamma
import math

# Основные параметры программы
N = 2000  # Объем выборки
K = 64  # Размер вспомогательной таблицы для генератора
M = 16  # Количество интервалов гистограммы


# Генератор случайных чисел
class MarsagliaMacLarenGenerator:
    def __init__(self, k_size=64):
        self.K = k_size
        self.Z1 = np.random.random(self.K)

    # Генерация одного случайного числа по алгоритму
    def rnd(self):
        g1 = np.random.random()
        g2 = np.random.random()
        m = int(g2 * self.K)
        res = self.Z1[m]
        self.Z1[m] = g1
        return res

    # Генерация выборки заданного размера
    def generate_sample(self, size):
        return np.array([self.rnd() for _ in range(size)])


# Функция генерации выборки распределения Вейбулла
def generate_weibull_mml(N, a, b, generator):
    uniform_sample = generator.generate_sample(N)
    return b * (-np.log(1 - uniform_sample)) ** (1 / a)


# Функция генерации выборки гамма-распределения
def generate_gamma_mml(N, k, theta, generator):
    uniform_sample = generator.generate_sample(N)
    return gamma.ppf(uniform_sample, a=k, scale=theta)


# Построение гистограмм и функций распределения для обеих выборок
def plot_all(sample_weibull, sample_gamma, M):
    counts_w, bin_edges_w = np.histogram(sample_weibull, bins=M)
    cdf_w = np.cumsum(counts_w) / len(sample_weibull)

    counts_g, bin_edges_g = np.histogram(sample_gamma, bins=M)
    cdf_g = np.cumsum(counts_g) / len(sample_gamma)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Гистограммы и CDF", fontsize=16)

    axes[0, 0].bar(bin_edges_w[:-1], counts_w, width=np.diff(bin_edges_w), align='edge', alpha=0.6)
    axes[0, 0].set_title("Вейбулл")
    axes[0, 0].grid()

    axes[0, 1].step(bin_edges_w[1:], cdf_w, where='post', color="red")
    axes[0, 1].set_title("CDF Вейбулл")
    axes[0, 1].grid()

    axes[1, 0].bar(bin_edges_g[:-1], counts_g, width=np.diff(bin_edges_g), align='edge', alpha=0.6)
    axes[1, 0].set_title("Гамма")
    axes[1, 0].grid()

    axes[1, 1].step(bin_edges_g[1:], cdf_g, where='post', color="red")
    axes[1, 1].set_title("CDF Гамма")
    axes[1, 1].grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return (counts_w, bin_edges_w), (counts_g, bin_edges_g)


# Вычисление статистических характеристик выборки
def compute_stats(sample, dist_name, dist_params):
    mean = np.mean(sample)
    var = np.var(sample, ddof=1)
    moment2 = np.mean(sample ** 2)
    moment3 = np.mean(sample ** 3)

    if dist_name == "weibull":
        a, b = dist_params
        theor_mean = b * math.gamma(1 + 1 / a)
        theor_var = b ** 2 * (math.gamma(1 + 2 / a) - (math.gamma(1 + 1 / a)) ** 2)
    elif dist_name == "gamma":
        k, theta = dist_params
        theor_mean = k * theta
        theor_var = k * (theta ** 2)
    else:
        theor_mean = theor_var = np.nan

    print(f"Мат. ожидание (выборочное): {mean:.6f}")
    print(f"Мат. ожидание (теоретическое): {theor_mean:.6f}")
    print(f"Дисперсия (выборочная): {var:.6f}")
    print(f"Дисперсия (теоретическая): {theor_var:.6f}")
    print(f"Второй момент: {moment2:.6f}")
    print(f"Третий момент: {moment3:.6f}")

    return mean, var, moment2, moment3


# Проверка согласия распределения с помощью критерия хи-квадрат
def chi2_test(counts, bin_edges, dist_name, dist_params):
    n = np.sum(counts)

    if dist_name == "weibull":
        a, b = dist_params
        cdf_values = weibull_min.cdf(bin_edges, c=a, scale=b)
    elif dist_name == "gamma":
        k, theta = dist_params
        cdf_values = gamma.cdf(bin_edges, a=k, scale=theta)

    expected_probs = []
    for i in range(len(bin_edges) - 1):
        p = cdf_values[i + 1] - cdf_values[i]
        expected_probs.append(p)

    expected_probs = np.array(expected_probs)
    expected_probs = expected_probs / np.sum(expected_probs)
    expected_counts = expected_probs * n

    if not np.isclose(np.sum(counts), np.sum(expected_counts), rtol=1e-8):
        expected_counts = expected_counts * np.sum(counts) / np.sum(expected_counts)

    if np.any(expected_counts == 0):
        valid_indices = expected_counts > 0
        counts = counts[valid_indices]
        expected_counts = expected_counts[valid_indices]

    chi2, p_value = chisquare(counts, f_exp=expected_counts)
    print(f"Критерий хи-квадрат: {chi2:.3f}, p-value: {p_value:.5f}")
    if p_value > 0.05:
        print("Гипотеза согласия не отвергается")
    else:
        print("Гипотеза согласия отвергается")
    return chi2, p_value


# Вывод таблицы частот распределения
def print_frequency_table(freq, M, sample_size):
    print(f"\nРаспределение чисел по интервалам [{(1.0 / M):.4f}]:")
    print('-' * 65)
    print('Интервал | Частота | Норм. частота | F(x) <= граница')
    print('-' * 65)

    cum_freq = 0
    for i in range(M):
        cum_freq += freq[i]
        cdf_val = cum_freq / sample_size
        print(f'{i + 1:3d}      | {freq[i]:5d}   | {freq[i] / sample_size:.6f}    | {cdf_val:.6f}')
    print('-' * 65)


# Основная часть программы
if __name__ == "__main__":
    # Параметры распределений
    a = 1.5
    b = 1.0
    k = 2.0
    theta = 2.0

    # Инициализация генератора
    generator = MarsagliaMacLarenGenerator(K)

    # Генерация выборок
    sample_weibull = generate_weibull_mml(N, a, b, generator)
    sample_gamma = generate_gamma_mml(N, k, theta, generator)

    # Построение графиков
    (counts_w, bin_edges_w), (counts_g, bin_edges_g) = plot_all(sample_weibull, sample_gamma, M)

    # Анализ распределения Вейбулла
    print("\nВЕЙБУЛЛ:")
    print_frequency_table(counts_w, M, N)
    mean_w, var_w, moment2_w, moment3_w = compute_stats(sample_weibull, "weibull", (a, b))
    chi2_test(counts_w, bin_edges_w, "weibull", (a, b))

    # Анализ гамма-распределения
    print("\nГАММА:")
    print_frequency_table(counts_g, M, N)
    mean_g, var_g, moment2_g, moment3_g = compute_stats(sample_gamma, "gamma", (k, theta))
    chi2_test(counts_g, bin_edges_g, "gamma", (k, theta))
