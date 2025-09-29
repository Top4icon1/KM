import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, exp
from scipy.special import erf
from scipy.stats import lognorm

# Параметры модели
M = 20        # количество шагов в одном блуждании
N = 20000     # количество симуляций (траекторий)

# Симуляция случайного блуждания
# Каждый шаг равновероятно равен +1 или -1.
# Сумма шагов S = 2*Binomial(M, 0.5) - M (быстрое векторное решение).
S = 2 * np.random.binomial(M, 0.5, size=N) - M
distances = np.abs(S)  # расстояние от начала координат

# Основные статистики
mean_dist = distances.mean()
std_dist = distances.std(ddof=1)
print(f"Среднее расстояние = {mean_dist:.4f}, стандартное отклонение = {std_dist:.4f}")

# Теоретическая модель: половинно-нормальное распределение
sigma = sqrt(M)  # стандартное отклонение суммы шагов

def half_normal_pdf(x, sigma):
    return np.sqrt(2)/(sigma*np.sqrt(pi)) * np.exp(-x**2/(2*sigma**2))

def half_normal_cdf(x, sigma):
    return erf(x / (np.sqrt(2)*sigma))

# Экспоненциальная аппроксимация (λ берём из эмпирического среднего)
lambda_exp = 1.0 / mean_dist
def exp_pdf(x, lam): return lam * np.exp(-lam * x) * (x >= 0)
def exp_cdf(x, lam): return 1 - np.exp(-lam * x)

# Логнормальная аппроксимация (по выборочному логарифму положительных значений)
positive = distances[distances > 0]
mu_log = np.log(positive).mean()
sigma_log = np.log(positive).std(ddof=1)

# Подготовка данных для отрисовки
counts, bins = np.histogram(distances, bins=range(int(distances.max())+2), density=True)
bin_centers = 0.5*(bins[:-1] + bins[1:])
x = np.linspace(0, distances.max(), 400)

# Построение на одной панели: гистограмма + ECDF
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Левая панель — гистограмма с аппроксимациями
axes[0].hist(distances, bins=range(int(distances.max())+2), density=True, alpha=0.6, label="Эмпирическая гистограмма")
axes[0].plot(x, half_normal_pdf(x, sigma), 'r', label="Половинно-нормальное")
axes[0].plot(x, exp_pdf(x, lambda_exp), 'g', label="Экспоненциальное")
axes[0].plot(x, lognorm.pdf(x, s=sigma_log, scale=np.exp(mu_log)), 'b', label="Логнормальное")
axes[0].set_title(f"Распределение расстояния |S| (M={M} шагов)")
axes[0].set_xlabel("|S|")
axes[0].set_ylabel("Плотность вероятности")
axes[0].grid()
axes[0].legend()

# Правая панель — эмпирическая функция распределения
sorted_d = np.sort(distances)
ecdf = np.arange(1, N+1) / N
axes[1].step(sorted_d, ecdf, where="post", label="Эмпирическая CDF")
axes[1].plot(x, half_normal_cdf(x, sigma), 'r', label="CDF половинно-нормальное")
axes[1].plot(x, exp_cdf(x, lambda_exp), 'g', label="CDF экспоненциальное")
axes[1].plot(x, lognorm.cdf(x, s=sigma_log, scale=np.exp(mu_log)), 'b', label="CDF логнормальное")
axes[1].set_title("Эмпирическая функция распределения (ECDF)")
axes[1].set_xlabel("|S|")
axes[1].set_ylabel("F(x)")
axes[1].grid()
axes[1].legend()

plt.tight_layout()
plt.show()
