"""
stationary_optimizer.py
-----------------------
Оптимизация размещения стационарных камер.

Задача формулируется как задача целочисленного линейного программирования (ЦЛП):

    Переменные:
        x[j] ∈ {0,1}  — размещается ли камера в позиции j
        y[i] ∈ {0,1}  — покрыта ли точка наблюдения i

    Целевая функция:
        Maximize  Σᵢ yᵢ

    Ограничения:
        yᵢ ≤ Σⱼ aᵢⱼ · xⱼ,  ∀i      (yᵢ=1 только если хоть одна камера покрывает i)
        Σⱼ xⱼ ≤ K                    (бюджетное ограничение)
        xⱼ, yᵢ ∈ {0,1}

Если установлена библиотека PuLP — используется точный CBC-решатель.
Иначе применяется встроенный жадный алгоритм (greedy set cover),
который гарантирует ln(N)-аппроксимацию оптимума.
"""

import numpy as np

try:
    import pulp
    _PULP_AVAILABLE = True
except ImportError:
    _PULP_AVAILABLE = False


# ---------------------------------------------------------------------------
class StationaryOptimizer:
    """
    Решает задачу оптимального размещения стационарных видеокамер.

    Parameters
    ----------
    coverage_matrix : np.ndarray (N, M)  — матрица покрытия
    costs           : list[float]        — стоимость каждой позиции
    budget_k        : int                — максимальное число камер
    """

    def __init__(self,
                 coverage_matrix: np.ndarray,
                 costs: list[float],
                 budget_k: int = 12):
        self.A = coverage_matrix          # (N, M)
        self.costs = np.array(costs, dtype=float)
        self.K = budget_k
        self.N, self.M = self.A.shape

        self.selected: np.ndarray = np.array([], dtype=int)  # индексы выбранных позиций
        self.coverage_pct: float = 0.0
        self.total_cost: float = 0.0

    # ------------------------------------------------------------------
    def solve(self) -> np.ndarray:
        """
        Запускает решение задачи.
        Возвращает массив индексов выбранных позиций камер.
        """
        if _PULP_AVAILABLE:
            return self._solve_ilp()
        else:
            return self._solve_greedy()

    # ------------------------------------------------------------------
    def _solve_ilp(self) -> np.ndarray:
        """Точное решение через PuLP / CBC."""
        prob = pulp.LpProblem("camera_placement", pulp.LpMaximize)

        x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(self.M)]
        y = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(self.N)]

        # Целевая функция
        prob += pulp.lpSum(y)

        # Бюджетное ограничение
        prob += pulp.lpSum(x) <= self.K

        # Ограничения покрытия: y[i] <= sum(a[i,j]*x[j])
        for i in range(self.N):
            covering = [x[j] for j in range(self.M) if self.A[i, j]]
            if covering:
                prob += y[i] <= pulp.lpSum(covering)
            else:
                prob += y[i] == 0

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = np.array([j for j in range(self.M)
                              if pulp.value(x[j]) and pulp.value(x[j]) > 0.5],
                             dtype=int)
        self._finalize(selected)
        return self.selected

    # ------------------------------------------------------------------
    def _solve_greedy(self) -> np.ndarray:
        """
        Жадный алгоритм покрытия множества (greedy set cover).
        На каждом шаге выбирает позицию, покрывающую наибольшее число
        ещё не покрытых точек.
        Гарантирует ln(N)-аппроксимацию оптимального решения.
        """
        uncovered = set(range(self.N))
        selected = []
        available = set(range(self.M))

        for _ in range(self.K):
            if not uncovered or not available:
                break

            # Для каждой доступной позиции — число новых покрытых точек
            best_j, best_gain = -1, -1
            for j in available:
                gain = len(uncovered & set(np.where(self.A[:, j])[0]))
                if gain > best_gain:
                    best_gain, best_j = gain, j

            if best_j == -1 or best_gain == 0:
                break

            selected.append(best_j)
            available.remove(best_j)
            newly_covered = set(np.where(self.A[:, best_j])[0])
            uncovered -= newly_covered

        self._finalize(np.array(selected, dtype=int))
        return self.selected

    # ------------------------------------------------------------------
    def _finalize(self, selected: np.ndarray):
        """Сохраняет результат и рассчитывает итоговые показатели."""
        self.selected = selected
        if self.N > 0:
            covered = self.A[:, selected].any(axis=1).sum() if len(selected) else 0
            self.coverage_pct = covered / self.N * 100.0
        self.total_cost = self.costs[selected].sum() if len(selected) else 0.0

    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        """Возвращает словарь с результатами оптимизации."""
        return {
            'selected_indices': self.selected.tolist(),
            'num_cameras':      len(self.selected),
            'coverage_pct':     round(self.coverage_pct, 2),
            'total_cost':       self.total_cost,
            'solver':           'ILP (PuLP/CBC)' if _PULP_AVAILABLE else 'Greedy Set Cover',
        }
