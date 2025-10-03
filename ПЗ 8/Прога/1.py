import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, lognorm

# Параметры задачи
M = 20        # количество шагов пешехода
N0 = 5000     # пробное число симуляций

#  расстояние от начала (модуль)
S = 2*np.random.binomial(M, 0.5, size=N0) - M
distances = np.abs(S)

# среднее и стандартное отклонение
mean_dist = distances.mean()
std_dist = distances.std(ddof=1)
print(f"[Пробный эксперимент] Среднее = {mean_dist:.4f}, std = {std_dist:.4f}")

# Планирование эксперимента

# оценка среднего
epsilon_mean = 0.1     # допустимая ошибка для среднего (задали сами)
gamma = 0.95           # достоверность
alpha = 1 - gamma
z = norm.ppf(1 - alpha/2)  # квантиль нормального распределения

# формула объёма выборки для оценки среднего
n_mean = int((z * std_dist / epsilon_mean)**2) + 1
print(f"Необходимое n для оценки среднего: {n_mean}")

# оценка дисперсии
# доверительный интервал для дисперсии через распределение хи-квадрат
epsilon_var = 0.2 * std_dist**2   # допустимая ошибка для дисперсии (20% от неё)
n_var = None
for n in range(30, 20000):
    chi2_low = chi2.ppf(alpha/2, df=n-1)
    chi2_high = chi2.ppf(1-alpha/2, df=n-1)
    low = (n-1)*std_dist**2 / chi2_high
    high = (n-1)*std_dist**2 / chi2_low
    if (high - low) <= 2*epsilon_var:
        n_var = n
        break
print(f"Необходимое n для оценки дисперсии: {n_var}")

# Основной эксперимент (берём максимум из двух n)
N = max(n_mean, n_var)
S = 2*np.random.binomial(M, 0.5, size=N) - M
distances = np.abs(S)

print(f"[Основной эксперимент] Среднее = {distances.mean():.4f}, std = {distances.std(ddof=1):.4f}")

# Построение гистограммы (для наглядности)
plt.hist(distances, bins=range(int(distances.max())+2), density=True, alpha=0.6)
plt.title(f"Распределение расстояния |S| при M={M}")
plt.xlabel("|S|")
plt.ylabel("Плотность вероятности")
plt.grid()
plt.show()
