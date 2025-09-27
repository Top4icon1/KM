import random
import math

# Входные данные
a1, a2, a3, a4 = 10, 20, 30, 40       # количество стрелков в группах
p1, p2, p3, p4 = 0.9, 0.75, 0.5, 0.2  # вероятности попадания для групп
NSIM = 100_000                        # число испытаний (>= 1000)

# Генератор Макларена–Марсальи
K = 64
Z1 = [0.0] * K

def init_Z1(seed=None):
    if seed is not None:
        random.seed(seed)
    for i in range(K):
        Z1[i] = random.random()

def rnd():
    g1 = random.random()
    g2 = random.random()
    m = int(g2 * K)
    res = Z1[m]
    Z1[m] = g1
    return res

# Аналитическое решение
def analytic_probability(a, p):
    N = sum(a)
    q = [1 - pi for pi in p]
    S = sum(ai * qi for ai, qi in zip(a, q))
    T = sum(ai * qi**2 for ai, qi in zip(a, q))
    if N * (N - 1) == 0:
        return 0.0
    P_both_miss = (S**2 - T) / (N * (N - 1))
    return 1 - P_both_miss

# Монте-Карло
def monte_carlo(a, p, nsim):
    init_Z1()
    Ntotal = sum(a)

    # создаем список групп стрелков
    groups = []
    for gi, ai in enumerate(a):
        groups.extend([gi] * ai)

    successes = 0
    for _ in range(nsim):
        # случайный выбор 2 разных индексов
        u1 = rnd()
        u2 = rnd()
        i = int(u1 * Ntotal)
        if i >= Ntotal: i = Ntotal - 1
        j_ = int(u2 * (Ntotal - 1))
        if j_ >= i:
            j = j_ + 1
        else:
            j = j_

        g1 = groups[i]
        g2 = groups[j]

        hit1 = rnd() < p[g1]
        hit2 = rnd() < p[g2]

        if hit1 or hit2:
            successes += 1

    phat = successes / nsim

    # доверительный интервал по теореме Лапласа
    z = 1.96  # для β=0.95
    se = math.sqrt(phat * (1 - phat) / nsim)
    ci_low = phat - z * se
    ci_high = phat + z * se

    return phat, (ci_low, ci_high), successes

# Основная программа
if __name__ == "__main__":
    a = [a1, a2, a3, a4]
    p = [p1, p2, p3, p4]

    print(f"Число стрелков: a = {a}")
    print(f"Вероятности: p = {p}")
    print(f"Число испытаний: {NSIM}\n")

    p_theor = analytic_probability(a, p)
    p_mc, (ci_low, ci_high), hits = monte_carlo(a, p, NSIM)

    print(f"Аналитическая вероятность:        {p_theor:.6f}")
    print(f"Оценка методом Монте-Карло:       {p_mc:.6f}")
    print(f"Доверительный интервал (β=0.95): [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Попадает ли аналитическая вероятность в интервал? -> {'Да' if ci_low <= p_theor <= ci_high else 'Нет'}")
    print(f"Благоприятных исходов: {hits} из {NSIM}")