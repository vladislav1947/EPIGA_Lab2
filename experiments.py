"""Обязательные эксперименты лабораторной работы № 2.

Эксперимент 1: серия из >= 20 независимых запусков базовой конфигурации.
Эксперимент 2: примеры допустимых и недопустимых решений.
Эксперимент 3: сравнение способов обработки ограничений
               (два коэффициента штрафа vs ремонт vs специализированная
                генерация).
Эксперимент 4: сравнение операторов (одноточечный/двухточечный/равномерный
               кроссовер; битовая мутация vs мутация-обмен).
Эксперимент 5: представление лучшего решения в табличном виде.
Эксперимент 6: сравнение ГА с конструктивной эвристикой и случайным
               допустимым поиском при том же бюджете вычислений функции.
"""

import csv
import os

import problem
from baseline import greedy_heuristic, run_random_feasible_search
from ga import run_ga
import runner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Параметры синтетического экземпляра задачи (фиксированный seed) ---
INSTANCE_PARAMS = dict(n=25, m=8, n_slots=5, seed=42, s_lo=10, s_hi=40)
RUNS = 21          # >= 20 независимых запусков
MAX_FE = 20000     # бюджет вычислений функции

# --- Параметры базовой конфигурации ГА ---
BASE = dict(
    pop_size=50,
    max_fe=MAX_FE,
    pc=0.9,
    pm=0.1,                 # вероятность мутации на бит
    selection="tournament",
    crossover="two_point",
    mutation="flip",
    init="mixed",           # половина ремонт, половина спец. генерация
    constraint="repair",
    elitism=1,
)

# --- Эксперимент 3: способы обработки ограничений ---
CONSTRAINT_CONFIGS = [
    ("repair", "constraint=repair (base)", {}),
    ("penalty_weak", "penalty weak (alpha=50, beta=30)",
     dict(constraint="penalty", alpha=50.0, beta=30.0)),
    ("penalty_strong", "penalty strong (alpha=500, beta=300)",
     dict(constraint="penalty", alpha=500.0, beta=300.0)),
    ("feasible_init", "init=feasible (спец. генерация)",
     dict(init="feasible")),
]

# --- Эксперимент 4: операторы вариации ---
# flip: случайная мутация, pm на ген (~2-3 изменённых гена на особь);
# move: специализированная мутация-перемещение, pm на особь (ровно одно
#       перемещение занятия в допустимую аудиторию), поэтому pm=1.0.
OPERATOR_CONFIGS = [
    ("xover_two_point", "crossover=two_point (base)", {}),
    ("xover_one_point", "crossover=one_point", dict(crossover="one_point")),
    ("xover_uniform", "crossover=uniform", dict(crossover="uniform")),
    ("mut_move", "mutation=move", dict(mutation="move", pm=1.0)),
]


def write_examples_csv(inst, path):
    """Эксперимент 2: примеры допустимых и недопустимых решений."""
    import random
    rng = random.Random(0)
    # недопустимое: все занятия в аудитории 0 (массовые конфликты
    # по времени и нарушения вместимости)
    all_zero = [0] * inst.n
    # недопустимое: занятие, не помещающееся в назначенную аудиторию
    # недопустимое: занятие, не помещающееся в назначенную аудиторию
    # (если такого нет — два занятия одного слота в одной аудитории)
    small_room = min(range(inst.m), key=lambda j: inst.caps[j])
    big_cands = [i for i in range(inst.n)
                 if inst.sizes[i] > inst.caps[small_room]]
    over = greedy_feasible(inst, rng)
    if big_cands:
        big = max(big_cands, key=lambda i: inst.sizes[i])
        over[big] = small_room
        over_name = "недопустимое (переполнение аудитории)"
    else:
        same = [i for i in range(inst.n)
                if inst.slots[i] == inst.slots[0]][:2]
        over[same[0]] = 0
        over[same[1]] = 0
        over_name = "недопустимое (конфликт по времени)"
    # допустимые: жадная эвристика и случайное допустимое решение
    greedy = problem.greedy_heuristic(inst)
    feasible = problem.random_feasible_solution(inst, rng)
    examples = [
        ("допустимое (жадная эвристика)", greedy),
        ("допустимое (случайная генерация)", feasible),
        ("недопустимое (все в аудитории 0)", all_zero),
        (over_name, over),
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["пример", "тип", "решение", "аудиторий",
                    "нарушений_вместимости", "конфликтующих_пар",
                    "допустимо"])
        for name, chrom in examples:
            w.writerow([
                name,
                "допустимое" if inst.is_feasible(chrom) else "недопустимое",
                ";".join(str(a) for a in chrom),
                inst.rooms_used(chrom),
                len(inst.cap_violations(chrom)),
                len(inst.clash_pairs(chrom)),
                "да" if inst.is_feasible(chrom) else "нет",
            ])


def greedy_feasible(inst, rng):
    """Вспомогательное: допустимое решение для построения примеров."""
    return problem.random_feasible_solution(inst, rng)


def write_best_solution_csv(inst, res, path):
    """Эксперимент 5: лучшее решение в форме, понятной пользователю.

    Расписание: строки — аудитории, столбцы — временные слоты, в ячейках —
    занятия с числом студентов.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assign = res["best_x"]
    grid = {(r, t): [] for r in range(inst.m) for t in range(inst.n_slots)}
    for i, a in enumerate(assign):
        grid[(a, inst.slots[i])].append(i)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["аудитория", "вместимость"] +
                   [f"слот {t}" for t in range(inst.n_slots)])
        for r in range(inst.m):
            row = [f"аудитория {r}", inst.caps[r]]
            for t in range(inst.n_slots):
                cell = grid[(r, t)]
                row.append("; ".join(
                    f"занятие {i} ({inst.sizes[i]} студ.)" for i in cell)
                    if cell else "—")
            w.writerow(row)
        w.writerow([])
        w.writerow(["занятий", inst.n, "", "", "", "", ""])
        w.writerow(["аудиторий использовано", inst.rooms_used(assign),
                    "", "", "", "", ""])
        w.writerow(["свободных мест", inst.waste(assign),
                    "", "", "", "", ""])
        w.writerow(["допустимо",
                    "да" if inst.is_feasible(assign) else "нет",
                    "", "", "", "", ""])


def write_params_csv(inst, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["параметр", "значение"])
        w.writerow(["задача", "назначение аудиторий занятиям (вариант 9)"])
        w.writerow(["тип данных", "синтетические, seed=42"])
        w.writerow(["число занятий n", inst.n])
        w.writerow(["число аудиторий m", inst.m])
        w.writerow(["число временных слотов", inst.n_slots])
        w.writerow(["число студентов", "целые из [10, 40] (равномерно)"])
        w.writerow(["вместимости аудиторий", ", ".join(str(c) for c in inst.caps)])
        w.writerow(["представление особи",
                    "вектор назначений длины n: a_i — аудитория занятия i"])
        w.writerow(["целевая функция",
                    "минимум числа аудиторий, затем минимум свободных мест; "
                    "fit = (m - rooms) * K - waste"])
        w.writerow(["число запусков в серии", RUNS])
        w.writerow(["бюджет вычислений функции", MAX_FE])
        w.writerow(["размер популяции", BASE["pop_size"]])
        w.writerow(["вероятность кроссинговера", BASE["pc"]])
        w.writerow(["вероятность мутации (на ген / на особь)", BASE["pm"]])
        w.writerow(["селекция", "турнирная, k=3"])
        w.writerow(["кроссовер", "двухточечный (также one_point, uniform)"])
        w.writerow(["мутация",
                    "случайная (flip) и спец. перемещение (move)"])
        w.writerow(["обработка ограничений",
                    "ремонт / штрафы (alpha, beta) / спец. генерация"])
        w.writerow(["элитизм", "1 особь"])
        w.writerow(["критерий остановки", f"лимит вычислений функции ({MAX_FE})"])
        for tag, name, overrides in CONSTRAINT_CONFIGS[1:]:
            w.writerow(["эксперимент 3: конфигурация", name])
        for tag, name, overrides in OPERATOR_CONFIGS[1:]:
            w.writerow(["эксперимент 4: конфигурация", name])
        w.writerow(["базовые методы",
                    "жадная эвристика best-fit; случайный "
                    "допустимый поиск"])


def print_stats(name, stats):
    print(f"{name}:")
    print(f"  best={stats['best']}  mean={stats['mean']:.2f}  "
          f"median={stats['median']:.2f}\n  stdev={stats['stdev']:.2f}  "
          f"worst={stats['worst']}  rooms={stats['mean_rooms']:.2f}  "
          f"feasible_frac={stats['mean_feasible_frac']:.3f}"
          f"  time={stats['mean_time_s']:.3f}s")


def main():
    # --- Экземпляр задачи ---
    inst = problem.generate_instance(**INSTANCE_PARAMS)
    problem.save_instance(inst, DATA_DIR)
    summary = inst.summary()
    print(f"Экземпляр: n={summary['n']}, m={summary['m']}, "
          f"слотов={summary['n_slots']}, "
          f"студентов={summary['total_students']}, "
          f"seed={summary['seed']}")
    print(f"Запусков в серии: {RUNS}, бюджет: {MAX_FE}\n")

    compare_csv = os.path.join(RESULTS_DIR, "config_compare.csv")
    # Удаляем старый сводный CSV, иначе append_summary_csv допишет дубликаты.
    if os.path.exists(compare_csv):
        os.remove(compare_csv)

    # --- Эксперимент 2: примеры решений ---
    write_examples_csv(inst, os.path.join(RESULTS_DIR, "examples.csv"))
    print("Примеры допустимых/недопустимых решений: results/examples.csv\n")

    # --- Эксперимент 1: базовая серия ---
    base_results, base_stats = runner.run_series(
        run_ga, runs=RUNS, inst=inst, **BASE)
    runner.write_runs_csv(base_results,
                          os.path.join(RESULTS_DIR, "runs_base.csv"))
    runner.write_history_csv(base_results,
                             os.path.join(RESULTS_DIR, "history_base.csv"))
    runner.append_summary_csv(base_stats, "GA base", compare_csv)
    print_stats("ГА (базовая конфигурация)", base_stats)

    # --- Эксперимент 3: способы обработки ограничений ---
    constraint_series = {"GA base": base_results}
    for tag, name, overrides in CONSTRAINT_CONFIGS[1:]:
        cfg = dict(BASE)
        cfg.update(overrides)
        results, stats = runner.run_series(run_ga, runs=RUNS, inst=inst, **cfg)
        runner.append_summary_csv(stats, "GA " + name, compare_csv)
        runner.write_history_csv(
            results, os.path.join(RESULTS_DIR, f"config_hist_{tag}.csv"))
        constraint_series["GA " + name] = results
        print_stats("ГА (" + name + ")", stats)

    # --- Эксперимент 4: операторы ---
    operator_series = {"GA base": base_results}
    for tag, name, overrides in OPERATOR_CONFIGS[1:]:
        cfg = dict(BASE)
        cfg.update(overrides)
        results, stats = runner.run_series(run_ga, runs=RUNS, inst=inst, **cfg)
        runner.append_summary_csv(stats, "GA " + name, compare_csv)
        runner.write_history_csv(
            results, os.path.join(RESULTS_DIR, f"config_hist_{tag}.csv"))
        operator_series["GA " + name] = results
        print_stats("ГА (" + name + ")", stats)

    # --- Эксперимент 6: базовые методы ---
    greedy = greedy_heuristic(inst)
    greedy_stats = {
        "runs": 1, "best": greedy["best_value"], "mean": greedy["best_value"],
        "median": greedy["best_value"], "stdev": 0.0,
        "worst": greedy["best_value"], "mean_rooms": greedy["best_rooms"],
        "mean_time_s": 0.0, "mean_fe": 1,
        "mean_feasible_frac": 1.0, "finals": [greedy["best_value"]],
    }
    rs_results, rs_stats = runner.run_series(
        run_random_feasible_search, runs=RUNS, inst=inst, max_fe=MAX_FE)
    runner.write_runs_csv(rs_results,
                          os.path.join(RESULTS_DIR, "runs_random_feasible.csv"))
    runner.write_history_csv(rs_results,
                             os.path.join(RESULTS_DIR,
                                          "history_random_feasible.csv"))
    runner.append_summary_csv(greedy_stats, "greedy heuristic", compare_csv)
    runner.append_summary_csv(rs_stats, "random feasible search", compare_csv)
    print_stats("Жадная эвристика", greedy_stats)
    print_stats("Случайный допустимый поиск", rs_stats)

    # --- Эксперимент 5: лучшее решение ---
    # Ищем максимум по всем сериям (база + ограничения + операторы),
    # без двойного учёта базовой серии.
    all_series = (list(base_results)
                  + [r for name, series in constraint_series.items()
                     for r in series if name != "GA base"]
                  + [r for name, series in operator_series.items()
                     for r in series if name != "GA base"])
    best_run = max(all_series, key=lambda r: r["best_value"])
    write_best_solution_csv(inst, best_run,
                            os.path.join(RESULTS_DIR, "best_solution.csv"))
    with open(os.path.join(RESULTS_DIR, "best_solution.txt"), "w") as f:
        f.write(f"best_value = {best_run['best_value']}\n")
        f.write(f"best_rooms = {best_run['best_rooms']} "
                f"(аудиторий всего: {inst.m})\n")
        f.write(f"best_waste = {best_run['best_waste']}\n")
        f.write(f"assign = {best_run['best_x']}\n")
        f.write(f"seed = {best_run['seed']}\n")
        f.write(f"feasible = {inst.is_feasible(best_run['best_x'])}\n")

    # --- Параметры и графики ---
    write_params_csv(inst, os.path.join(RESULTS_DIR, "params.csv"))
    import plots
    plots.plot_convergence(base_results,
                           os.path.join(RESULTS_DIR, "convergence.png"))
    plots.plot_curves_comparison(
        constraint_series,
        os.path.join(RESULTS_DIR, "constraint_compare.png"),
        title="Сравнение способов обработки ограничений")
    plots.plot_curves_comparison(
        operator_series,
        os.path.join(RESULTS_DIR, "operators_compare.png"),
        title="Сравнение операторов вариации")
    plots.plot_boxplot(
        {name: [r["best_value"] for r in res]
         for name, res in constraint_series.items()},
        os.path.join(RESULTS_DIR, "constraint_boxplot.png"))
    plots.plot_baselines(
        {"ГА": base_stats, "жадная эвристика": greedy_stats,
         "случайный допустимый поиск": rs_stats},
        os.path.join(RESULTS_DIR, "ga_vs_baselines.png"))
    plots.plot_curves_comparison(
        {"ГА": base_results,
         "случайный допустимый поиск": rs_results},
        os.path.join(RESULTS_DIR, "ga_vs_random.png"),
        title="ГА vs случайный допустимый поиск")
    plots.plot_flowchart(os.path.join(RESULTS_DIR, "flowchart.png"))
    print(f"\nРезультаты сохранены в {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
