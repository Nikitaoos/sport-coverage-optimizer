"""
economic_balancer.py
--------------------
Технико-экономический анализ и балансировка конфигураций.

Для каждой конфигурации вычисляется:
- TCO (Total Cost of Ownership) = стоимость камер + стоимость БПЛА
- Итоговый процент покрытия (камеры + БПЛА)

Ранжирование конфигураций выполняется методом взвешенной суммы
нормированных критериев:

    Score = w_cov * Coverage_norm - w_cost * Cost_norm

где:
    Coverage_norm = coverage_pct / max(coverage_pct across configs)
    Cost_norm     = tco / max(tco across configs)
    w_cov + w_cost = 1  (по умолчанию w_cov=0.7, w_cost=0.3)
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
@dataclass
class Configuration:
    """Одна конфигурация системы видеонаблюдения."""
    label:              str   = ''
    num_cameras:        int   = 0
    camera_cost:        float = 0.0
    num_drones:         int   = 0
    drone_cost_per:     float = 0.0
    coverage_cameras:   float = 0.0   # % покрытия только камерами
    coverage_total:     float = 0.0   # % покрытия с учётом БПЛА

    # Вычисляемые поля
    tco:   float = field(default=0.0, init=False)
    score: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.tco = self.camera_cost + self.num_drones * self.drone_cost_per


# ---------------------------------------------------------------------------
class EconomicBalancer:
    """
    Рассчитывает TCO и ранжирует конфигурации.

    Parameters
    ----------
    w_coverage : float  — вес критерия покрытия  (0..1)
    w_cost     : float  — вес критерия стоимости (0..1), w_coverage + w_cost = 1
    """

    def __init__(self, w_coverage: float = 0.7, w_cost: float = 0.3):
        assert abs(w_coverage + w_cost - 1.0) < 1e-6, \
            "Сумма весов должна равняться 1.0"
        self.w_cov  = w_coverage
        self.w_cost = w_cost
        self.configs: list[Configuration] = []

    # ------------------------------------------------------------------
    def add(self, config: Configuration):
        """Добавляет конфигурацию для сравнения."""
        self.configs.append(config)

    # ------------------------------------------------------------------
    def rank(self) -> list[Configuration]:
        """
        Нормирует критерии, вычисляет Score для каждой конфигурации
        и возвращает список, отсортированный по убыванию Score.
        """
        if not self.configs:
            return []

        max_cov  = max(c.coverage_total for c in self.configs) or 1.0
        max_cost = max(c.tco for c in self.configs) or 1.0

        for c in self.configs:
            cov_norm  = c.coverage_total / max_cov
            cost_norm = c.tco / max_cost
            c.score = self.w_cov * cov_norm - self.w_cost * cost_norm

        self.configs.sort(key=lambda c: c.score, reverse=True)
        return self.configs

    # ------------------------------------------------------------------
    def best(self) -> Configuration | None:
        """Возвращает наилучшую конфигурацию."""
        ranked = self.rank()
        return ranked[0] if ranked else None

    # ------------------------------------------------------------------
    def summary(self) -> list[dict]:
        """Возвращает список словарей с показателями всех конфигураций."""
        ranked = self.rank()
        return [
            {
                'label':            c.label,
                'num_cameras':      c.num_cameras,
                'num_drones':       c.num_drones,
                'coverage_cameras': round(c.coverage_cameras, 1),
                'coverage_total':   round(c.coverage_total, 1),
                'tco':              int(c.tco),
                'score':            round(c.score, 4),
            }
            for c in ranked
        ]
