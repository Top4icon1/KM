import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, gamma, chisquare
import math

# основные параметры
N = 2000          # объем выборки
K = 15            # количество интервалов
a = 1.5           # shape параметр Вейбулла
b = 1.0           # scale параметр Вейбулла
k = 2.0           # параметр k для гамма
theta = 2.0       # параметр theta для гамма

# генерация выборки Вейбулла
def generate_weibull(N, a, b):
    return weibull_min.rvs(c=a, scale=b, size=N)

# генерация выборки гамма-распределения
def generate_gamma(N, k, theta):
    return gamma.rvs(a=k, scale=theta, size=N)

# рисуем гистограммы и CDF для обеих выборок на одной панели (2x2)
def plot_all(sample_weibull, sample_gamma, K):
    counts_w, bin_edges_w = np.histogram(sample_weibull, bins=K)
    cdf_w = np.cumsum(counts_w) / len(sample_weibull)

    counts_g, bin_edges_g = np.histogram(sample_gamma, bins=K)
    cdf_g = np.cumsum(counts_g) / len(sample_gamma)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Гистограммы и CDF", fontsize=16)

    # гистограмма Вейбулла
    axes[0, 0].bar(bin_edges_w[:-1], counts_w, width=np.diff(bin_edges_w), align='edge', alpha=0.6)
    axes[0, 0].set_title("Гистограмма (Вейбулл)")
    axes[0, 0].grid()

    # эмпирическая функция распределения (CDF) для Вейбулла
    axes[0, 1].step(bin_edges_w[1:], cdf_w, where='post', color="red")
    axes[0, 1].set_title("CDF (Вейбулл)")
    axes[0, 1].grid()

    # гистограмма гамма-распределения
    axes[1, 0].bar(bin_edges_g[:-1], counts_g, width=np.diff(bin_edges_g), align='edge', alpha=0.6)
    axes[1, 0].set_title("Гистограмма (Гамма)")
    axes[1, 0].grid()

    # CDF для гамма
    axes[1, 1].step(bin_edges_g[1:], cdf_g, where='post', color="red")
    axes[1, 1].set_title("CDF (Гамма)")
    axes[1, 1].grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return (counts_w, bin_edges_w), (counts_g, bin_edges_g)

# расчет матожидания и дисперсии + сравнение с теоретическими
def compute_stats(sample, dist_name, dist_params):
    mean = np.mean(sample)
    var = np.var(sample, ddof=1)

    if dist_name == "weibull":
        a, b = dist_params
        theor_mean = b * math.gamma(1 + 1 / a)
        theor_var = b**2 * (math.gamma(1 + 2 / a) - (math.gamma(1 + 1 / a))**2)
    elif dist_name == "gamma":
        k, theta = dist_params
        theor_mean = k * theta
        theor_var = k * (theta**2)
    else:
        theor_mean = theor_var = np.nan

    print(f"Мат. ожидание (выборочное): {mean:.6f}")
    print(f"Мат. ожидание (теоретическое): {theor_mean:.6f}")
    print(f"Дисперсия (выборочная): {var:.6f}")
    print(f"Дисперсия (теоретическая): {theor_var:.6f}")
    return mean, var

# критерий Пирсона (χ²) для проверки согласия распределения
def chi2_test(counts, bin_edges, dist_name, dist_params):
    n = np.sum(counts)

    # считаем теоретические вероятности для каждого интервала
    if dist_name == "weibull":
        a, b = dist_params
        cdf = weibull_min.cdf(bin_edges, c=a, scale=b)
    elif dist_name == "gamma":
        k, theta = dist_params
        cdf = gamma.cdf(bin_edges, a=k, scale=theta)

    expected_counts = []
    for i in range(len(bin_edges) - 1):
        p = cdf[i+1] - cdf[i]
        expected_counts.append(n * p)

    expected_counts = np.array(expected_counts)
    expected_counts *= n / np.sum(expected_counts)  # нормировка на случай погрешности сумм

    chi2, p_value = chisquare(counts, f_exp=expected_counts)
    print(f"Критерий хи квадрат: {chi2:.3f}, p-value: {p_value:.5f}")
    if p_value > 0.05:
        print("Гипотеза согласия не отвергается")
    else:
        print("Гипотеза согласия отвергается")
    return chi2, p_value

if __name__ == "__main__":
    # генерируем выборки
    sample_weibull = generate_weibull(N, a, b)
    sample_gamma = generate_gamma(N, k, theta)

    # строим все графики на одной панели
    (counts_w, bin_edges_w), (counts_g, bin_edges_g) = plot_all(sample_weibull, sample_gamma, K)

    # анализ Вейбулла
    print("\n Вейбулл")
    compute_stats(sample_weibull, "weibull", (a, b))
    chi2_test(counts_w, bin_edges_w, "weibull", (a, b))

    # анализ Гамма
    print("\n Гамма")
    compute_stats(sample_gamma, "gamma", (k, theta))
    chi2_test(counts_g, bin_edges_g, "gamma", (k, theta))
