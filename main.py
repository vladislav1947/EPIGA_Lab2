"""Клиент командной строки лабораторной работы № 2 (вариант 9).

Примеры запуска:

  # Полный набор обязательных экспериментов
  python3 experiments.py

  # Один запуск ГА с выводом лучшего решения
  python3 main.py run

  # Запуск с явными параметрами
  python3 main.py run --seed 42 --constraint penalty --alpha 500 --beta 300

  # Сравнение способов обработки ограничений
  python3 main.py compare-constraints

  # Базовые методы: жадная эвристика и случайный допустимый поиск
  python3 main.py greedy
  python3 main.py random-search --max-fe 10000
"""

import argparse
import os

import problem
from baseline import greedy_heuristic, run_random_feasible_search
from ga import run_ga
import runner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

INSTANCE_PARAMS = dict(n=25, m=8, n_slots=5, seed=42, s_lo=10, s_hi=40)


def build_parser():
    p = argparse.ArgumentParser(
        description="ГА для задачи назначения аудиторий (ЛР № 2, вариант 9)")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--n", type=int, default=25,
                        help="число занятий в экземпляре")
    common.add_argument("--inst-seed", type=int, default=42,
                        help="seed генератора экземпляра задачи")
    common.add_argument("--pop-size", type=int, default=50)
    common.add_argument("--max-fe", type=int, default=20000)
    common.add_argument("--pc", type=float, default=0.9)
    common.add_argument("--pm", type=float, default=0.1)
    common.add_argument("--selection", default="tournament",
                        choices=["tournament", "rank"])
    common.add_argument("--crossover", default="two_point",
                        choices=["one_point", "two_point", "uniform"])
    common.add_argument("--mutation", default="flip",
                        choices=["flip", "move"])
    common.add_argument("--init", default="mixed",
                        choices=["repair", "feasible", "mixed"])
    common.add_argument("--constraint", default="repair",
                        choices=["repair", "penalty"])
    common.add_argument("--alpha", type=float, default=50.0,
                        help="штраф за каждое нарушение вместимости")
    common.add_argument("--beta", type=float, default=30.0,
                        help="штраф за каждую конфликтующую пару")
    common.add_argument("--elitism", type=int, default=1)
    common.add_argument("--runs", type=int, default=21)

    p_run = sub.add_parser("run", parents=[common], help="одиночный запуск ГА")
    sub.add_parser("experiments", parents=[common],
                   help="все обязательные эксперименты")
    p_cmp = sub.add_parser("compare-constraints", parents=[common],
                           help="сравнение способов обработки ограничений")
    p_greedy = sub.add_parser("greedy", parents=[common],
                              help="жадная конструктивная эвристика")
    p_rs = sub.add_parser("random-search", parents=[common],
                          help="случайный допустимый поиск")
    return p


def ga_kwargs(args):
    return dict(
        pop_size=args.pop_size, max_fe=args.max_fe, pc=args.pc, pm=args.pm,
        selection=args.selection, crossover=args.crossover,
        mutation=args.mutation, init=args.init, constraint=args.constraint,
        alpha=args.alpha, beta=args.beta, elitism=args.elitism)


def get_instance(args):
    params = dict(INSTANCE_PARAMS)
    params["n"] = args.n
    params["seed"] = args.inst_seed
    return problem.generate_instance(**params)


def print_solution(inst, res):
    print(f"best_value = {res['best_value']} "
          f"(аудиторий: {res['best_rooms']}, свободных мест: "
          f"{res['best_waste']})")
    print(f"назначения = {res['best_x']}")
    print(f"допустимо: {inst.is_feasible(res['best_x'])}")


def cmd_run(args):
    inst = get_instance(args)
    res = run_ga(inst, seed=args.seed, **ga_kwargs(args))
    print_solution(inst, res)


def cmd_greedy(args):
    inst = get_instance(args)
    res = greedy_heuristic(inst)
    print("Жадная эвристика (best-fit по убыванию числа студентов):")
    print_solution(inst, res)


def cmd_random_search(args):
    inst = get_instance(args)
    res = run_random_feasible_search(inst, args.max_fe, seed=args.seed)
    print("Случайный допустимый поиск:")
    print_solution(inst, res)


def cmd_compare_constraints(args):
    inst = get_instance(args)
    cfgs = {
        "repair": dict(constraint="repair"),
        "penalty weak (a=50, b=30)": dict(constraint="penalty",
                                          alpha=50.0, beta=30.0),
        "penalty strong (a=500, b=300)": dict(constraint="penalty",
                                               alpha=500.0, beta=300.0),
        "init=feasible": dict(init="feasible"),
    }
    for name, overrides in cfgs.items():
        kw = ga_kwargs(args)
        kw.update(overrides)
        _, stats = runner.run_series(run_ga, runs=args.runs, inst=inst, **kw)
        print(f"{name}: mean={stats['mean']:.2f} best={stats['best']} "
              f"median={stats['median']:.2f} stdev={stats['stdev']:.2f} "
              f"feasible_frac={stats['mean_feasible_frac']:.3f}")


def cmd_experiments(args):
    import experiments
    experiments.INSTANCE_PARAMS["n"] = args.n
    experiments.INSTANCE_PARAMS["seed"] = args.inst_seed
    experiments.RUNS = args.runs
    experiments.MAX_FE = args.max_fe
    experiments.BASE.update(ga_kwargs(args))
    experiments.main()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        cmd_run(args)
    elif args.command == "experiments":
        cmd_experiments(args)
    elif args.command == "greedy":
        cmd_greedy(args)
    elif args.command == "random-search":
        cmd_random_search(args)
    elif args.command == "compare-constraints":
        cmd_compare_constraints(args)


if __name__ == "__main__":
    main()
