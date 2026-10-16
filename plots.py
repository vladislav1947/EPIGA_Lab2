"""Построение графиков для ЛР № 2 (требуется matplotlib)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from runner import curve_stats  # noqa: E402


def plot_convergence(results, path, title="Сходимость ГА",
                     n_points=200):
    grid, mins, means, maxs = curve_stats(results, n_points)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, means, color="#1f77b4", lw=2, label="среднее")
    ax.fill_between(grid, mins, maxs, color="#1f77b4", alpha=0.25,
                    label="разброс min–max")
    ax.plot(grid, maxs, color="#2ca02c", lw=1, ls="--", label="максимум")
    ax.plot(grid, mins, color="#d62728", lw=1, ls="--", label="минимум")
    ax.set_xlabel("Число вычислений функции")
    ax.set_ylabel("Лучшая приспособленность")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves_comparison(series_by_name, path,
                           title="Сравнение конфигураций", n_points=200):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, results in series_by_name.items():
        grid, mins, means, maxs = curve_stats(results, n_points)
        ax.plot(grid, means, lw=2, label=name)
    ax.set_xlabel("Число вычислений функции")
    ax.set_ylabel("Лучшая приспособленность")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_boxplot(finals_by_name, path,
                 title="Распределение итоговой приспособленности"):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(finals_by_name)
    ax.boxplot([finals_by_name[n] for n in names], tick_labels=names)
    ax.set_ylabel("Итоговая приспособленность")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_flowchart(path):
    """Блок-схема генетического алгоритма назначения аудиторий."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(6.5, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis("off")

    def box(y, text, color="#dbe9f6", w=6.4, x=1.8, h=1.1):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            fc=color, ec="#333333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9)

    def arrow(y_from, y_to, label=None, x=5.0):
        ax.annotate("", xy=(x, y_to + 0.05), xytext=(x, y_from - 0.05),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4))
        if label:
            ax.text(x + 0.2, (y_from + y_to) / 2, label, fontsize=8)

    box(22.5, "Начало: экземпляр задачи (занятия,\nслоты, аудитории), seed", "#e8f6e3")
    box(20.9, "Инициализация популяции:\nремонт случайных назначений + спец. генерация")
    box(19.5, "Оценка приспособленности\nс учётом способа обработки ограничений")
    box(18.1, "Критерий остановки:\nлимит вычислений функции?", "#fff2cc")
    ax.annotate("", xy=(8.6, 18.15), xytext=(8.2, 18.65),
                arrowprops=dict(arrowstyle="-|>", lw=1.4))
    ax.text(8.3, 18.75, "да", fontsize=8)
    box(17.3, "Лучшее допустимое\nрешение", "#f6d6d6", w=1.6, x=8.2, h=1.1)
    ax.annotate("", xy=(9.0, 22.5), xytext=(9.0, 18.45),
                arrowprops=dict(arrowstyle="-|>", lw=1.4))
    ax.text(9.2, 21.0, "конец", fontsize=8, rotation=90)
    arrow(18.1, 16.9, "нет")
    box(15.7, "Селекция родителей\n(турнирная k=3)")
    box(14.1, "Кроссинговер:\nодноточечный / двухточечный / равномерный")
    box(12.5, "Мутация: случайная (flip) или\nперемещение (move)")
    box(10.9, "Обработка ограничений:\nремонт или штраф (alpha, beta)")
    box(9.3, "Оценка приспособленности потомков")
    box(7.9, "Новое поколение с элитизмом")
    ax.annotate("", xy=(1.8, 18.65), xytext=(1.8, 7.9),
                arrowprops=dict(arrowstyle="-|>", lw=1.4))
    ax.text(0.05, 13.0, "следующее поколение", fontsize=8, rotation=90)

    # Детали ремонта
    box(4.9, "Ремонт (вариант обработки ограничений):\nперенос нарушающих занятий\nв случайную допустимую аудиторию",
        "#fde9d9", w=8.4, x=0.8, h=1.9)
    ax.annotate("", xy=(5.0, 8.35), xytext=(5.0, 8.35),
                arrowprops=dict(arrowstyle="-|>", lw=1.4))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_baselines(values_by_name, path, title="ГА vs базовые методы"):
    """Столбчатая диаграмма: средняя приспособленность по методам."""
    names = list(values_by_name)
    means = [values_by_name[n]["mean"] for n in names]
    stds = [values_by_name[n].get("stdev", 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(names, means, yerr=stds, capsize=5,
           color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(names)])
    ax.set_ylabel("Приспособленность")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
