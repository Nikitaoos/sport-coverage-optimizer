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

При равенстве покрытия предпочтение отдаётся более дешёвой конфигурации.
Для этого в целевую функцию добавлен штраф за стоимость с весом eps,
подобранным так, что суммарный штраф строго меньше единицы:

    Maximize  Σᵢ yᵢ − eps · Σⱼ (cⱼ / c_max) · xⱼ,   eps = 0.5 / K

Такой штраф не может «перевесить» ни одной покрытой точки и работает
исключительно как правило разрешения ничьих.

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
        self.N, self.M = self.A.shape
        self.K = max(0, min(int(budget_k), self.M))

        self.selected: np.ndarray = np.array([], dtype=int)  # индексы выбранных позиций
        self.coverage_pct: float = 0.0
        self.covered_points: int = 0
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

        # Целевая функция: покрытие с штрафом-разрешителем ничьих по стоимости
        c_max = float(self.costs.max()) if self.M and self.costs.max() > 0 else 1.0
        eps = 0.5 / max(self.K, 1)
        prob += pulp.lpSum(y) - eps * pulp.lpSum(
            (self.costs[j] / c_max) * x[j] for j in range(self.M))

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
        uncovered = np.ones(self.N, dtype=bool)
        available = np.ones(self.M, dtype=bool)
        selected = []

        for _ in range(self.K):
            if not uncovered.any() or not available.any():
                break

            # Прирост покрытия для каждой позиции (векторизованно)
            gains = (self.A[uncovered, :] > 0).sum(axis=0).astype(float)
            gains[~available] = -1.0

            best_gain = gains.max()
            if best_gain <= 0:
                break

            # Разрешение ничьих по стоимости: из позиций с максимальным
            # приростом выбирается самая дешёвая
            ties = np.where(gains == best_gain)[0]
            best_j = int(ties[np.argmin(self.costs[ties])])

            selected.append(best_j)
            available[best_j] = False
            uncovered &= ~(self.A[:, best_j] > 0)

        self._finalize(np.array(selected, dtype=int))
        return self.selected

    # ------------------------------------------------------------------
    def _finalize(self, selected: np.ndarray):
        """Сохраняет результат и рассчитывает итоговые показатели."""
        self.selected = selected
        covered = 0
        if self.N > 0 and len(selected):
            covered = int(self.A[:, selected].any(axis=1).sum())
        self.covered_points = covered
        if self.N > 0:
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
            'covered_points':   int(self.covered_points),
            'solver':           'ILP (PuLP/CBC)' if _PULP_AVAILABLE else 'Greedy Set Cover',
        }
