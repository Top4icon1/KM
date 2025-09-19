import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Константы
N = 1000  # объем выборки
K = 25  # количество интервалов
MU = 1.0  # математическое ожидание N(1, 0.7)
SIGMA = 0.7  # стандартное отклонение


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

    def generate_normal_clt(self, n=12):
        # Метод на основе ЦПТ (сумма 12 равномерных)
        sum_u = 0.0
        for i in range(n):
            sum_u += self.generate()
        z = (sum_u - n / 2) / math.sqrt(n / 12)
        return MU + SIGMA * z

    def generate_normal_box_muller(self):
        # Метод Бокса-Мюллера
        u1 = self.generate()
        u2 = self.generate()
        while u1 == 0:
            u1 = self.generate()

        # Преобразование Бокса-Мюллера
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)

        return MU + SIGMA * z0


# Теоретические функции для нормального распределения
def theoretical_normal_pdf(x, mu, sigma):
    # Плотность нормального распределения
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def theoretical_normal_cdf(x, mu, sigma):
    # Функция распределения (через функцию ошибок)
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


# Основная программа
def main():
    print(f"Объем выборки: {N}")
    print(f"Количество интервалов: {K}")
    print(f"Параметры распределения: N({MU}, {SIGMA:.1f})")
    print()

    # Создаем генератор
    generator = MyGenerator()

    # Генерируем выборки двумя методами
    # Генерация выборки методом ЦПТ
    sample_clt = []
    for i in range(N):
        sample_clt.append(generator.generate_normal_clt())

    # Генерация выборки методом Бокса-Мюллера
    sample_box = []
    for i in range(N):
        sample_box.append(generator.generate_normal_box_muller())

    # Анализируем обе выборки
    samples = [sample_clt, sample_box]
    methods = ["Центральная предельная теорема", "Бокса-Мюллера"]

    for i, sample in enumerate(samples):
        print(f"Анализ для метода: {methods[i]}")
        # Основные статистики
        mean = sum(sample) / len(sample)
        variance = sum((x - mean) ** 2 for x in sample) / (len(sample) - 1)

        print("Основные статистики:")
        print(f"Выборочное среднее: {mean:.6f}")
        print(f"Теоретическое среднее: {MU:.6f}")
        print(f"Отклонение: {abs(mean - MU):.6f}")
        print()
        print(f"Выборочная дисперсия: {variance:.6f}")
        print(f"Теоретическая дисперсия: {SIGMA ** 2:.6f}")
        print(f"Отклонение: {abs(variance - SIGMA ** 2):.6f}")
        print()

        # Гистограмма
        min_val = min(sample)
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

        # Вывод таблицы частот
        print("Таблица частот:")
        print("Интервал\t\tЧастота\tНорм. частота")
        print("-" * 50)
        for j in range(K):
            left = min_val + j * interval_width
            right = min_val + (j + 1) * interval_width
            print(f"{left:.3f}-{right:.3f}\t{frequencies[j]:4d}\t{norm_freq[j]:.4f}")

        # Графики
        plt.figure(figsize=(15, 5))

        # Гистограмма с теоретической плотностью
        plt.subplot(1, 2, 1)
        intervals = [min_val + j * interval_width for j in range(K + 1)]
        plt.hist(sample, bins=intervals, alpha=0.7, color='lightblue',
                 edgecolor='black', density=True, label='Эмпирическая')

        # Теоретическая плотность
        x_plot = np.linspace(min_val, max_val, 1000)
        y_pdf = [theoretical_normal_pdf(x, MU, SIGMA) for x in x_plot]
        plt.plot(x_plot, y_pdf, 'r-', linewidth=2, label='Теоретическая')

        plt.title(f'Гистограмма ({methods[i]})')
        plt.xlabel('Значение')
        plt.ylabel('Плотность вероятности')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Функция распределения
        plt.subplot(1, 2, 2)
        sorted_sample = sorted(sample)
        y_empirical = [(j + 1) / len(sorted_sample) for j in range(len(sorted_sample))]
        plt.step(sorted_sample, y_empirical, where='post', label='Эмпирическая')

        # Теоретическая функция распределения
        y_theoretical = [theoretical_normal_cdf(x, MU, SIGMA) for x in x_plot]
        plt.plot(x_plot, y_theoretical, 'r-', label='Теоретическая')

        plt.title(f'Функция распределения ({methods[i]})')
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Критерий Колмогорова
        print("\nПроверка критерием Колмогорова:")
        D_max = 0

        # Проверяем все точки выборки
        for j, x in enumerate(sorted_sample):
            F_emp = (j + 1) / len(sorted_sample)
            F_theor = theoretical_normal_cdf(x, MU, SIGMA)
            D_current = abs(F_emp - F_theor)
            if D_current > D_max:
                D_max = D_current

        critical_value = 1.36 / math.sqrt(N)
        print(f"Статистика D = {D_max:.6f}")
        print(f"Критическое значение = {critical_value:.6f}")

        if D_max < critical_value:
            print("Гипотеза о соответствии распределения принимается")
            print()
            print()
        else:
            print("Гипотеза о соответствии распределения отвергается")
            print()
            print()


if __name__ == "__main__":
    main()