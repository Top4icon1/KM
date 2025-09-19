import random
import math
import matplotlib.pyplot as plt

# Константы
N = 1000  # объем выборки
K = 25  # количество интервалов


# Генератор Макларена-Марсальи
class MyGenerator:
    def __init__(self):
        self.K = 64
        self.Z1 = [0.0] * self.K
        self.init_z1()

    def init_z1(self):
        # Начальные значения
        for i in range(self.K):
            self.Z1[i] = random.random()

    def generate(self):
        # Одно случайное число
        u1 = random.random()
        u2 = random.random()
        idx = int(u2 * self.K)
        result = self.Z1[idx]
        self.Z1[idx] = u1
        return result


# Функция для генерации выборки по заданному распределению
def generate_sample(size):
    # Выборка с помощью метода обратных функций
    generator = MyGenerator()
    sample = []
    e = math.e

    for i in range(size):
        u = generator.generate()
        boundary = (e - 1) / e

        if u < boundary:
            x = u * e / (2 * (e - 1))
        else:
            x = -0.5 * math.log(1 - u)

        sample.append(x)

    return sample


# Теоретические функции
def theoretical_cdf(x):
    e = math.e
    if x < 0.5:
        return (2 * (e - 1) / e) * x
    else:
        return 1 - math.exp(-2 * x)


def theoretical_pdf(x):
    e = math.e
    if x < 0.5:
        return 2 * (e - 1) / e
    else:
        return 2 * math.exp(-2 * x)


# Теоретические моменты
def theoretical_mean():
    e = math.e
    part1 = (2 * (e - 1) / e) * (0.5 ** 2 / 2)
    part2 = 0.5 * math.exp(-1) + 0.25 * math.exp(-1)
    return part1 + part2


def theoretical_variance():
    mean = theoretical_mean()
    e = math.e

    part1 = (2 * (e - 1) / e) * (0.5 ** 3 / 3)
    part2 = 0.25 * math.exp(-1) + 0.5 * math.exp(-1) + 0.25 * math.exp(-1)
    second_moment = part1 + part2

    return second_moment - mean ** 2


# Основная программа
def main():
    print(f"Объем выборки: {N}")
    print(f"Число интервалов: {K}")
    print()

    # Выборка
    sample = generate_sample(N)

    # Основные статистики
    mean = sum(sample) / len(sample)
    variance = sum((x - mean) ** 2 for x in sample) / (len(sample) - 1)

    # Теоретические значения
    theor_mean = theoretical_mean()
    theor_var = theoretical_variance()

    print("Основные статистики:")
    print(f"Выборочное среднее: {mean:.6f}")
    print(f"Теоретическое среднее: {theor_mean:.6f}")
    print(f"Отклонение: {abs(mean - theor_mean):.6f}")
    print()
    print(f"Выборочная дисперсия: {variance:.6f}")
    print(f"Теоретическая дисперсия: {theor_var:.6f}")
    print(f"Отклонение: {abs(variance - theor_var):.6f}")
    print()

    # Гистограмма
    min_val = 0
    max_val = max(sample)
    interval_width = (max_val - min_val) / K

    # Частоты
    frequencies = [0] * K
    for x in sample:
        idx = int((x - min_val) / interval_width)
        if idx >= K:
            idx = K - 1
        frequencies[idx] += 1

    # Нормированные частоты
    norm_freq = [f / N for f in frequencies]

    # Вывод
    print("Таблица частот:")
    print("Интервал\tЧастота\tНорм. частота")
    print("-" * 40)
    for i in range(K):
        left = min_val + i * interval_width
        right = min_val + (i + 1) * interval_width
        print(f"{left:.3f}-{right:.3f}\t{frequencies[i]}\t{norm_freq[i]:.4f}")

    # Графики
    plt.figure(figsize=(12, 8))

    # Гистограмма с теоретической плотностью
    plt.subplot(2, 2, 1)
    intervals = [min_val + i * interval_width for i in range(K + 1)]
    plt.hist(sample, bins=intervals, alpha=0.7, color='skyblue', edgecolor='black', density=True)

    # Теоретическая плотность
    x_pdf = [i * max_val / 200 for i in range(201)]
    y_pdf = [theoretical_pdf(x) for x in x_pdf]
    plt.plot(x_pdf, y_pdf, 'r-', linewidth=2, label='Теоретическая PDF')

    plt.title('Гистограмма с теоретической плотностью')
    plt.xlabel('Значение')
    plt.ylabel('Плотность вероятности')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Эмпирическая функция распределения
    plt.subplot(2, 2, 2)
    sorted_sample = sorted(sample)
    y_empirical = [(i + 1) / len(sorted_sample) for i in range(len(sorted_sample))]
    plt.step(sorted_sample, y_empirical, where='post', label='Эмпирическая')

    # Теоретическая функция распределения
    x_theoretical = [i * max_val / 100 for i in range(101)]
    y_theoretical = [theoretical_cdf(x) for x in x_theoretical]
    plt.plot(x_theoretical, y_theoretical, 'r-', label='Теоретическая')

    plt.title('Функция распределения')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Колмогоров
    D_max = 0

    # Проверка всех точек выборки
    for i, x in enumerate(sorted_sample):
        F_emp = (i + 1) / len(sorted_sample)
        F_theor = theoretical_cdf(x)
        D_current = abs(F_emp - F_theor)
        if D_current > D_max:
            D_max = D_current

    # Проверка точек между выборками
    for i in range(len(sorted_sample)):
        if i > 0:
            x = sorted_sample[i - 1]
            F_emp = i / len(sorted_sample)
            F_theor = theoretical_cdf(x)
            D_current = abs(F_emp - F_theor)
            if D_current > D_max:
                D_max = D_current

    critical_value = 1.36 / math.sqrt(N)
    print(f"Статистика D = {D_max:.6f}")
    print(f"Критическое значение = {critical_value:.6f}")

    if D_max < critical_value:
        print("Гипотеза о соответствии распределения принимается")
    else:
        print("Гипотеза о соответствии распределения отвергается")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()