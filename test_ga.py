"""Самопроверка операторов и обработки ограничений (ЛР № 2, вариант 9).

Запуск: python3 test_ga.py
Не требует сторонних библиотек. Проверяется:
  * ремонт восстанавливает допустимость;
  * специализированная генерация даёт только допустимые решения;
  * жадная эвристика даёт допустимое решение;
  * мутация-перемещение сохраняет допустимость допустимых решений;
  * кроссоверы сохраняют длину и допустимый диапазон генов;
  * штрафная функция корректно штрафует недопустимые решения;
  * элитизм обеспечивает монотонное неубывание рекорда.
"""

import random

import problem
from ga import (flip_mutation, move_mutation, one_point_crossover, run_ga,
                two_point_crossover, uniform_crossover)
from baseline import greedy_heuristic, run_random_feasible_search

failures = []


def check(cond, msg):
    status = "OK " if cond else "FAIL"
    if not cond:
        failures.append(msg)
    print(f"[{status}] {msg}")


def main():
    inst = problem.generate_instance(n=25, m=8, n_slots=5, seed=42)
    rng = random.Random(0)

    # --- Ремонт ---
    all_zero = [0] * inst.n
    check(not inst.is_feasible(all_zero),
          "все занятия в одной аудитории — недопустимое решение")
    repaired = problem.repair(inst, all_zero, rng)
    check(inst.is_feasible(repaired),
          "ремонт делает решение «все в аудитории 0» допустимым")

    # --- Специализированная генерация ---
    ok = all(inst.is_feasible(problem.random_feasible_solution(inst, rng))
             for _ in range(200))
    check(ok, "случайная специализированная генерация всегда допустима")

    # --- Жадная эвристика ---
    check(inst.is_feasible(greedy_heuristic(inst)["best_x"]),
          "жадная эвристика даёт допустимое решение")

    # --- Мутация-перемещение ---
    chrom = problem.random_feasible_solution(inst, rng)
    ok = all(inst.is_feasible(move_mutation(inst, chrom, rng, pm=1.0))
             for _ in range(200))
    check(ok, "мутация-перемещение сохраняет допустимость")

    # --- Случайная мутация + ремонт ---
    chrom = problem.random_feasible_solution(inst, rng)
    ok = all(inst.is_feasible(
        problem.repair(inst, flip_mutation(inst, chrom, rng, pm=0.1), rng))
        for _ in range(100))
    check(ok, "случайная мутация + ремонт всегда дают допустимое решение")

    # --- Кроссоверы ---
    p1 = problem.random_assignment(inst, rng)
    p2 = problem.random_assignment(inst, rng)
    for name, cross in [("one_point", one_point_crossover),
                        ("two_point", two_point_crossover),
                        ("uniform", uniform_crossover)]:
        c1, c2 = cross(p1, p2, rng)
        ok = (len(c1) == inst.n and len(c2) == inst.n
              and all(0 <= a < inst.m for a in c1 + c2))
        check(ok, f"кроссовер {name} сохраняет длину и диапазон генов")

    # --- Штрафная функция ---
    fit_all = problem.penalty_fitness(inst, all_zero, alpha=500, beta=300)
    fit_repaired = problem.penalty_fitness(inst, repaired, alpha=500,
                                           beta=300)
    check(fit_repaired > fit_all,
          "штрафы делают допустимое решение лучше недопустимого")

    # --- Жадная эвристика: детерминизм ---
    g1 = greedy_heuristic(inst)["best_value"]
    g2 = greedy_heuristic(inst)["best_value"]
    check(g1 == g2, "жадная эвристика детерминирована")
    rs_val = run_random_feasible_search(inst, 200, seed=0)["best_value"]
    print(f"       (жадная эвристика: {g1}, случайный поиск (200): {rs_val})")

    # --- ГА: монотонность рекорда и допустимость ---
    res = run_ga(inst, pop_size=50, max_fe=20000, seed=0)
    hist = res["history"]
    check(all(hist[i][1] <= hist[i + 1][1] + 1e-9
              for i in range(len(hist) - 1)),
          "рекорд ГА монотонно не убывает (элитизм)")
    check(inst.is_feasible(res["best_x"]), "лучшее решение ГА допустимо")
    check(res["fe"] <= 20000 + 50, "число вычислений функции в пределах бюджета")
    greedy_val = greedy_heuristic(inst)["best_value"]
    check(res["best_value"] >= greedy_val,
          f"ГА не хуже жадной эвристики (ГА: {res['best_value']}, "
          f"жадная: {greedy_val})")

    print()
    if failures:
        print("Ошибки:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
