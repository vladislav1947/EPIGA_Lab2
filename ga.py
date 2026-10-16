"""Генетический алгоритм для задачи назначения аудиторий (ЛР № 2, вариант 9).

Представление особи — вектор назначений длины n: a_i in {0, ..., m-1} —
номер аудитории занятия i. Максимизируется лексикографическая
приспособленность: в первую очередь число сэкономленных аудиторий,
во вторую — экономия свободных мест.

Все операторы (инициализация, турнирная/ранговая селекция, одноточечный,
двухточечный и равномерный кроссовер, случайная мутация,
специализированная мутация-перемещение, ремонт, штрафование) реализованы
самостоятельно.
"""

import random

import problem


# --------------------------------------------------------------------------
# Селекция (максимизация приспособленности)
# --------------------------------------------------------------------------

def tournament_selection(fit, rng, k=3):
    n = len(fit)
    best = rng.randrange(n)
    for _ in range(k - 1):
        cand = rng.randrange(n)
        if fit[cand] > fit[best]:
            best = cand
    return best


def rank_selection(fit, rng, pressure=2.0):
    """Ранговая селекция: вероятность выбора линейно растёт с рангом."""
    n = len(fit)
    order = sorted(range(n), key=lambda i: fit[i])  # order[-1] — лучшая
    cum = []
    total = 0.0
    for j in range(n):
        # лучшая особь (позиция n-1 в порядке возрастания) получает вес s
        w = (2.0 - pressure) + 2.0 * (pressure - 1.0) * j / (n - 1)
        total += w
        cum.append(total)
    r = rng.random() * total
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] < r:
            lo = mid + 1
        else:
            hi = mid
    return order[lo]


SELECTIONS = {
    "tournament": tournament_selection,
    "rank": rank_selection,
}


# --------------------------------------------------------------------------
# Кроссовер для векторов назначений
# --------------------------------------------------------------------------

def one_point_crossover(p1, p2, rng):
    n = len(p1)
    point = rng.randrange(n)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2


def two_point_crossover(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.sample(range(n + 1), 2))
    c1 = p1[:a] + p2[a:b] + p1[b:]
    c2 = p2[:a] + p1[a:b] + p2[b:]
    return c1, c2


def uniform_crossover(p1, p2, rng):
    c1 = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(len(p1))]
    c2 = [p2[i] if rng.random() < 0.5 else p1[i] for i in range(len(p1))]
    return c1, c2


CROSSOVERS = {
    "one_point": one_point_crossover,
    "two_point": two_point_crossover,
    "uniform": uniform_crossover,
}


# --------------------------------------------------------------------------
# Мутация
# --------------------------------------------------------------------------

def flip_mutation(inst, chrom, rng, pm):
    """Случайная мутация: каждый ген с вероятностью pm заменяется на
    случайную аудиторию.

    Может нарушать оба ограничения; допустимость восстанавливается
    ремонтом (при constraint="repair") или учитывается штрафом.
    """
    return [rng.randrange(inst.m) if rng.random() < pm else chrom[i]
            for i in range(len(chrom))]


def move_mutation(inst, chrom, rng, pm):
    """Специализированная мутация-перемещение для вектора назначений.

    С вероятностью pm (на особь): выбирается случайное занятие и
    переносится в случайную допустимую для него аудиторию (подходит
    по вместимости и свободна в его слот). Оператор устроен так, что
    допустимое решение остаётся допустимым.
    """
    chrom = list(chrom)
    if rng.random() >= pm:
        return chrom
    i = rng.randrange(len(chrom))
    feasible = inst.feasible_rooms(chrom, i)
    feasible = [r for r in feasible if r != chrom[i]]
    if feasible:
        chrom[i] = rng.choice(feasible)
    return chrom


MUTATIONS = {
    "flip": flip_mutation,
    "move": move_mutation,
}


# --------------------------------------------------------------------------
# Инициализация популяции
# --------------------------------------------------------------------------

def init_population(inst, rng, pop_size, mode):
    """Инициализация популяции.

    mode="repair":   случайные векторы назначений + ремонт;
    mode="feasible": специализированная генерация допустимых решений;
    mode="mixed":    половина ремонт, половина допустимая генерация.
    """
    pop = []
    for i in range(pop_size):
        if mode == "repair":
            chrom = problem.random_assignment(inst, rng)
        elif mode == "feasible":
            chrom = problem.random_feasible_solution(inst, rng)
        elif mode == "mixed":
            if i % 2 == 0:
                chrom = problem.random_assignment(inst, rng)
            else:
                chrom = problem.random_feasible_solution(inst, rng)
        else:
            raise ValueError(f"Неизвестный режим инициализации: {mode!r}")
        pop.append(chrom)
    return pop


# --------------------------------------------------------------------------
# Основной цикл генетического алгоритма
# --------------------------------------------------------------------------

def run_ga(inst,
           pop_size=50, max_fe=20000, pc=0.9, pm=0.1,
           selection="tournament", crossover="two_point", mutation="flip",
           init="mixed", constraint="repair",
           alpha=1.0, beta=1.0, elitism=1, seed=0,
           tournament_k=3, rank_pressure=2.0):
    """Запустить ГА на экземпляре задачи назначения аудиторий.

    Способ обработки ограничений:
      constraint="repair"  — к каждому потомку применяется ремонт,
                              приспособленность равна fit;
      constraint="penalty" — недопустимые решения штрафуются (alpha, beta),
                              ремонт не применяется.

    Возвращает словарь:
        best_x      — лучшее ДОПУСТИМОЕ найденное решение;
        best_value  — его приспособленность;
        best_rooms  — число используемых аудиторий в нём;
        best_waste  — число свободных мест в нём;
        history     — список пар (число вычислений функции, лучшая fit);
        fe          — число вычислений функции;
        feasible_frac — доля допустимых решений в финальной популяции.
    """
    rng = random.Random(seed)
    select = SELECTIONS[selection]
    cross = CROSSOVERS[crossover]

    pop = init_population(inst, rng, pop_size, init)
    if constraint == "repair":
        pop = [problem.repair(inst, c, rng) for c in pop]

    fit = []
    for chrom in pop:
        if constraint == "repair":
            fit.append(inst.fitness(chrom))
        else:
            fit.append(problem.penalty_fitness(inst, chrom, alpha, beta))
    fe = pop_size

    # Лучшее допустимое решение
    feasible_idx = [i for i in range(pop_size) if inst.is_feasible(pop[i])]
    if feasible_idx:
        b = max(feasible_idx, key=lambda i: fit[i])
        best_x = list(pop[b])
        best_value = fit[b]
    else:
        best_x = None
        best_value = float("-inf")
    history = [(fe, best_value if best_x is not None else float("nan"))]

    generation = 0
    while fe < max_fe:
        generation += 1
        # --- Элитизм ---
        order = sorted(range(pop_size), key=lambda i: fit[i], reverse=True)
        new_pop = [list(pop[order[i]]) for i in range(elitism)]
        new_fit = [fit[order[i]] for i in range(elitism)]

        while len(new_pop) < pop_size and fe < max_fe:
            i1 = (select(fit, rng, k=tournament_k) if selection == "tournament"
                  else select(fit, rng, pressure=rank_pressure))
            i2 = (select(fit, rng, k=tournament_k) if selection == "tournament"
                  else select(fit, rng, pressure=rank_pressure))

            if rng.random() < pc:
                c1, c2 = cross(pop[i1], pop[i2], rng)
            else:
                c1, c2 = list(pop[i1]), list(pop[i2])

            for c in (c1, c2):
                if len(new_pop) >= pop_size or fe >= max_fe:
                    break
                if mutation == "flip":
                    c = flip_mutation(inst, c, rng, pm)
                else:
                    c = move_mutation(inst, c, rng, pm)
                if constraint == "repair":
                    c = problem.repair(inst, c, rng)
                    f = inst.fitness(c)
                else:
                    f = problem.penalty_fitness(inst, c, alpha, beta)
                new_pop.append(c)
                new_fit.append(f)
                fe += 1
                # Обновление лучшего допустимого решения
                if constraint == "repair" or inst.is_feasible(c):
                    if f > best_value:
                        best_value = f
                        best_x = list(c)

        pop, fit = new_pop, new_fit
        pop_size = len(pop)

        if best_x is not None:
            history.append((fe, best_value))
        else:
            history.append((fe, float("nan")))

    if best_x is None:
        # вырожденный случай: берём ремонт лучшего по штрафу
        order = sorted(range(pop_size), key=lambda i: fit[i], reverse=True)
        best_x = problem.repair(inst, pop[order[0]], rng)
        best_value = inst.fitness(best_x)

    feasible_frac = (sum(1 for c in pop if inst.is_feasible(c)) / pop_size)

    return {
        "best_x": best_x,
        "best_value": best_value,
        "best_rooms": inst.rooms_used(best_x),
        "best_waste": inst.waste(best_x),
        "history": history,
        "fe": fe,
        "generations": generation,
        "feasible_frac": feasible_frac,
    }
