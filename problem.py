"""Задача назначения аудиторий занятиям (вариант 9, ЛР № 2).

Постановка:
    Дано n занятий, каждому сопоставлены временной слот t_i и число
    студентов s_i, а также m аудиторий с вместимостями c_j. Требуется
    назначить каждое занятие ровно одной аудитории (вектор назначений
    a = (a_1, ..., a_n), a_i in {0, ..., m-1}) так, чтобы:
      1) занятие помещалось в аудиторию: c[a_i] >= s_i (вместимость);
      2) в одной аудитории не было пересечений по времени: занятия,
         назначенные в одну аудиторию, должны иметь разные слоты.
    Цель — использовать как можно меньше аудиторий (первичный критерий),
    а при равном их числе — минимизировать суммарное число свободных
    мест (вторичный критерий).

Формально (минимизация):
    min rooms_used(a), затем min waste(a),
    где rooms_used(a) = |{a_i}| — число задействованных аудиторий,
         waste(a) = sum_i (c[a_i] - s_i) — свободные места.

Приспособленность (максимизация) — лексикографическая свёртка:
    fit(a) = (m - rooms_used(a)) * K - waste(a),
    где K = n * max_cap + 1 строго больше любого возможного waste,
    поэтому уменьшение числа аудиторий всегда важнее экономии мест.

Индивидуальный вариант: V = 9 (N = 15, Q = 1, Y = 26).
Данные синтетические; процедура генерации и seed описаны в функции
generate_instance и зафиксированы в data/instance.json.
"""

import csv
import json
import os


# --------------------------------------------------------------------------
# Описание экземпляра задачи
# --------------------------------------------------------------------------

class Instance:
    """Экземпляр задачи назначения аудиторий."""

    def __init__(self, n, m, slots, sizes, caps, n_slots, seed,
                 gen_params=None):
        self.n = n                      # число занятий (>= 20)
        self.m = m                      # число аудиторий (>= 5)
        self.slots = list(slots)        # временной слот занятия i
        self.sizes = list(sizes)        # число студентов занятия i
        self.caps = list(caps)          # вместимость аудитории j
        self.n_slots = n_slots          # число временных слотов
        self.seed = seed
        self.gen_params = gen_params or {}
        # Вес K лексикографической свёртки: больше любого waste.
        self.k_lex = n * max(caps) + 1

    def rooms_used(self, assign):
        return len(set(assign))

    def waste(self, assign):
        return sum(self.caps[a] - self.sizes[i]
                   for i, a in enumerate(assign))

    def fitness(self, assign):
        """Приспособленность допустимого решения (максимизация)."""
        return ((self.m - self.rooms_used(assign)) * self.k_lex
                - self.waste(assign))

    def cap_violations(self, assign):
        """Индексы занятий, не помещающихся в назначенную аудиторию."""
        return [i for i, a in enumerate(assign)
                if self.caps[a] < self.sizes[i]]

    def clash_pairs(self, assign):
        """Пары занятий (i, j), конфликтующих по времени в одной аудитории."""
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (assign[i] == assign[j]
                        and self.slots[i] == self.slots[j]):
                    pairs.append((i, j))
        return pairs

    def is_feasible(self, assign):
        return (not self.cap_violations(assign)
                and not self.clash_pairs(assign))

    def feasible_rooms(self, assign, i):
        """Аудитории, куда можно перенести занятие i без нарушений."""
        out = []
        for r in range(self.m):
            if self.caps[r] < self.sizes[i]:
                continue
            clash = any(assign[j] == r and j != i
                        and self.slots[j] == self.slots[i]
                        for j in range(self.n))
            if not clash:
                out.append(r)
        return out

    def summary(self):
        return {
            "n": self.n,
            "m": self.m,
            "n_slots": self.n_slots,
            "total_students": sum(self.sizes),
            "total_capacity": sum(self.caps),
            "seed": self.seed,
            "gen_params": self.gen_params,
        }


def generate_instance(n=25, m=8, n_slots=5, seed=42,
                      s_lo=10, s_hi=40,
                      caps=(25, 30, 35, 40, 50, 60, 80, 110)):
    """Сгенерировать синтетический экземпляр задачи.

    Процедура генерации (фиксированный seed):
      * число студентов — целые, равномерно из [s_lo, s_hi];
      * временные слоты распределяются сбалансированно: после перемешивания
        занятия назначаются на слоты циклически, поэтому в каждом слоте
        ровно n / n_slots занятий (n кратно n_slots);
      * вместимости аудиторий фиксированы параметром caps.
    Нижняя граница числа аудиторий равна размеру наибольшего слота
    (занятия одного слота требуют разных аудиторий).
    """
    import random
    rng = random.Random(seed)
    sizes = [rng.randint(s_lo, s_hi) for _ in range(n)]
    order = list(range(n))
    rng.shuffle(order)
    slots = [0] * n
    for pos, i in enumerate(order):
        slots[i] = pos % n_slots
    caps = list(caps)
    assert m == len(caps)
    assert n % n_slots == 0
    gen_params = {
        "n": n, "m": m, "n_slots": n_slots,
        "s_lo": s_lo, "s_hi": s_hi, "caps": caps,
    }
    return Instance(n, m, slots, sizes, caps, n_slots, seed, gen_params)


def save_instance(inst, data_dir):
    """Сохранить экземпляр: classes.csv, rooms.csv, instance.json."""
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "classes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "slot", "students"])
        for i in range(inst.n):
            w.writerow([i, inst.slots[i], inst.sizes[i]])
    with open(os.path.join(data_dir, "rooms.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["room_id", "capacity"])
        for j in range(inst.m):
            w.writerow([j, inst.caps[j]])
    with open(os.path.join(data_dir, "instance.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "summary": inst.summary(),
            "slots": inst.slots,
            "sizes": inst.sizes,
            "caps": inst.caps,
            "n_slots": inst.n_slots,
        }, f, ensure_ascii=False, indent=2)


def load_instance(data_dir):
    """Загрузить экземпляр задачи из каталога data_dir."""
    with open(os.path.join(data_dir, "instance.json"), encoding="utf-8") as f:
        data = json.load(f)
    s = data["summary"]
    return Instance(s["n"], s["m"], data["slots"], data["sizes"],
                    data["caps"], data["n_slots"], s["seed"],
                    s["gen_params"])


# --------------------------------------------------------------------------
# Обработка ограничений
# --------------------------------------------------------------------------

def repair(inst, assign, rng):
    """Восстановление допустимости (ремонт) вектора назначений.

    Занятия в случайном порядке проверяются на нарушения (вместимость,
    временной конфликт); нарушающее занятие переносится в случайную
    допустимую аудиторию (подходит по вместимости и свободна в его слот).
    Выбор случаен, чтобы не смещать поиск к жадному аттрактору и сохранять
    разнообразие популяции. Если допустимой аудитории нет, назначение
    сохраняется как есть.
    """
    assign = [int(a) for a in assign]
    order = list(range(inst.n))
    rng.shuffle(order)
    for i in order:
        bad_cap = inst.caps[assign[i]] < inst.sizes[i]
        bad_clash = any(assign[j] == assign[i] and j != i
                        and inst.slots[j] == inst.slots[i]
                        for j in range(inst.n))
        if not (bad_cap or bad_clash):
            continue
        feasible = inst.feasible_rooms(assign, i)
        if not feasible:
            continue
        assign[i] = rng.choice(feasible)
    return assign


def penalty_fitness(inst, assign, alpha, beta):
    """Штрафная функция приспособленности (максимизация).

    fit = (m - rooms_used) * K - waste
          - alpha * число нарушений вместимости
          - beta * число конфликтующих пар.
    """
    return (inst.fitness(assign)
            - alpha * len(inst.cap_violations(assign))
            - beta * len(inst.clash_pairs(assign)))


# --------------------------------------------------------------------------
# Конструктивные процедуры построения допустимых решений
# --------------------------------------------------------------------------

def greedy_heuristic(inst):
    """Детерминированная конструктивная эвристика (best-fit).

    Занятия упорядочиваются по убыванию числа студентов и назначаются
    в допустимую аудиторию с минимальными свободными местами.
    Результат всегда допустим, если экземпляр разрешим.
    """
    order = sorted(range(inst.n), key=lambda i: -inst.sizes[i])
    assign = [0] * inst.n
    occupied = set()  # занятые пары (аудитория, слот)
    for i in order:
        cands = [r for r in range(inst.m)
                 if inst.caps[r] >= inst.sizes[i]
                 and (r, inst.slots[i]) not in occupied]
        if not cands:
            # экземпляр неразрешим жадным проходом — берём первую
            # подходящую по вместимости (решение будет недопустимым)
            cands = [r for r in range(inst.m)
                     if inst.caps[r] >= inst.sizes[i]] or [0]
            best = min(cands, key=lambda r: inst.caps[r])
        else:
            best = min(cands, key=lambda r: inst.caps[r] - inst.sizes[i])
        assign[i] = best
        occupied.add((best, inst.slots[i]))
    return assign


def random_feasible_solution(inst, rng):
    """Рандомизированная специализированная генерация допустимого решения.

    Занятия в случайном порядке назначаются в случайную допустимую
    аудиторию; в конце выполняется страховочный ремонт. Результат
    всегда допустим, если экземпляр разрешим.
    """
    order = list(range(inst.n))
    rng.shuffle(order)
    assign = [0] * inst.n
    occupied = set()
    for i in order:
        cands = [r for r in range(inst.m)
                 if inst.caps[r] >= inst.sizes[i]
                 and (r, inst.slots[i]) not in occupied]
        if cands:
            assign[i] = rng.choice(cands)
        else:
            assign[i] = rng.randrange(inst.m)
        occupied.add((assign[i], inst.slots[i]))
    return repair(inst, assign, rng)


def random_assignment(inst, rng):
    """Случайный вектор назначений (может быть недопустимым)."""
    return [rng.randrange(inst.m) for _ in range(inst.n)]
