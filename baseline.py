"""Базовые методы отсчёта для ЛР № 2 (вариант 9):

  * greedy_heuristic — детерминированная конструктивная эвристика
    (жадный best-fit по убыванию числа студентов);
  * run_random_feasible_search — случайный допустимый поиск при том же
    бюджете вычислений функции, что и у ГА.
"""

import problem


def greedy_heuristic(inst):
    """Конструктивная эвристика: один запуск, детерминированный результат."""
    chrom = problem.greedy_heuristic(inst)
    return {
        "best_x": chrom,
        "best_value": inst.fitness(chrom),
        "best_rooms": inst.rooms_used(chrom),
        "best_waste": inst.waste(chrom),
        "history": [(1, inst.fitness(chrom))],
        "fe": 1,
        "generations": 0,
        "feasible_frac": 1.0,
    }


def run_random_feasible_search(inst, max_fe, seed=0):
    """Случайный допустимый поиск: max_fe независимых попыток.

    Каждая попытка — рандомизированная генерация допустимого решения
    (см. problem.random_feasible_solution). Отслеживается лучшее
    найденное значение приспособленности.
    """
    import random
    rng = random.Random(seed)
    best = problem.random_feasible_solution(inst, rng)
    best_value = inst.fitness(best)
    history = [(1, best_value)]
    for i in range(1, max_fe):
        chrom = problem.random_feasible_solution(inst, rng)
        value = inst.fitness(chrom)
        if value > best_value:
            best_value = value
            best = list(chrom)
        history.append((i + 1, best_value))
    return {
        "best_x": best,
        "best_value": best_value,
        "best_rooms": inst.rooms_used(best),
        "best_waste": inst.waste(best),
        "history": history,
        "fe": max_fe,
        "generations": 0,
        "feasible_frac": 1.0,
    }
