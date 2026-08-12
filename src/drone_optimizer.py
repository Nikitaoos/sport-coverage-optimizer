"""
drone_optimizer.py
------------------
Оптимизация состава и маршрутов БПЛА для доопределения покрытия зон,
недоступных стационарным камерам.

Постановка. Заданы: множество T' непокрытых стационарными камерами точек,
бюджет флота K_d (максимальное число аппаратов), максимальная длина маршрута
одного аппарата R_max и радиус полосы наблюдения r_obs. Требуется выбрать
маршруты, максимизирующие число наблюдаемых точек из T'.

Существенно, что бюджет K_d и ограничение R_max являются связывающими:
аппарат физически не успевает обойти все точки своего кластера, поэтому
часть точек может остаться ненаблюдаемой. Точка считается наблюдаемой,
если расстояние от неё до ломаной маршрута не превышает r_obs, — то есть
покрытие достигается как непосредственным пролётом над точкой, так и
захватом её в полосу наблюдения при пролёте мимо.

Алгоритм:
1. Если T' пусто либо K_d = 0 — маршруты не строятся.
2. T' кластеризуется методом K-means на k = min(K_d, |T'|) кластеров.
3. Для каждого кластера маршрут наращивается жадно: старт в точке, ближайшей
   к центроиду; на каждом шаге выбирается ближайшая ещё не наблюдаемая точка,
   для которой звено (с обходом запретных зон) укладывается в остаток
   лимита R_max. Точки, уже попавшие в полосу наблюдения, пропускаются.
4. Порядок обхода улучшается локальным поиском (2-opt и Or-opt); ход
   принимается только если наблюдаемость не уменьшается. Высвобожденный
   ресурс дальности расходуется на дополнительные точки.
5. Наращивание прекращается, когда ни одна точка не помещается в остаток
   лимита. Непосещённые и не попавшие в полосу точки остаются непокрытыми.

Если установлен scikit-learn — используется его KMeans.
Иначе применяется встроенная реализация (Lloyd's algorithm).
"""

import math
import numpy as np

from .spatial_model import (
    rect_intersects_forbidden,
    points_covered_by_route,
    route_is_clear,
    route_length,
)

try:
    from sklearn.cluster import KMeans as _SKLearnKMeans
    _SK_AVAILABLE = True
except ImportError:
    _SK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Встроенная реализация K-means (Lloyd's algorithm)
# ---------------------------------------------------------------------------

def _simple_kmeans(points: np.ndarray, k: int,
                   max_iter: int = 100, seed: int = 42) -> np.ndarray:
    """
    Простая реализация K-means.
    Возвращает массив меток clusters shape (N,).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=k, replace=False)
    centroids = points[idx].copy()

    labels = np.full(len(points), -1, dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = points[mask].mean(axis=0)

    return labels


# ---------------------------------------------------------------------------
# Планирование звена маршрута с обходом запретных зон
# ---------------------------------------------------------------------------

def _zone_corners(fz: dict, margin: float) -> list:
    """Углы запретной зоны с отступом margin."""
    fx, fy, fw, fh = fz['x'], fz['y'], fz['width'], fz['height']
    return [
        (fx - margin,      fy - margin),
        (fx + fw + margin, fy - margin),
        (fx - margin,      fy + fh + margin),
        (fx + fw + margin, fy + fh + margin),
    ]


def plan_leg(a: tuple, b: tuple, forbidden_zones: list,
             margin: float = 3.0) -> list | None:
    """
    Планирует звено маршрута из точки a в точку b с обходом запретных зон.

    Каждый вариант обхода проверяется целиком: все его плечи должны быть
    свободны от ВСЕХ запретных зон. Из допустимых вариантов выбирается
    кратчайший.

    Returns
    -------
    waypoints : list[tuple] — промежуточные точки и конечная точка b
                (без начальной точки a), либо None, если обход не найден.
    """
    if not rect_intersects_forbidden(a[0], a[1], b[0], b[1], forbidden_zones):
        return [b]

    def leg_clear(p, q) -> bool:
        return not rect_intersects_forbidden(p[0], p[1], q[0], q[1],
                                             forbidden_zones)

    corners = []
    for fz in forbidden_zones:
        corners.extend(_zone_corners(fz, margin))

    # Вариант 1: обход через один угол
    best, best_len = None, float('inf')
    for c in corners:
        if leg_clear(a, c) and leg_clear(c, b):
            length = (math.hypot(c[0] - a[0], c[1] - a[1]) +
                      math.hypot(b[0] - c[0], b[1] - c[1]))
            if length < best_len:
                best_len, best = length, [c, b]
    if best is not None:
        return best

    # Вариант 2: обход через два угла
    for c1 in corners:
        if not leg_clear(a, c1):
            continue
        for c2 in corners:
            if c1 == c2 or not leg_clear(c1, c2) or not leg_clear(c2, b):
                continue
            length = (math.hypot(c1[0] - a[0], c1[1] - a[1]) +
                      math.hypot(c2[0] - c1[0], c2[1] - c1[1]) +
                      math.hypot(b[0] - c2[0], b[1] - c2[1]))
            if length < best_len:
                best_len, best = length, [c1, c2, b]
    return best


def leg_cost(a: tuple, legs: list) -> float:
    """Длина звена маршрута, заданного списком промежуточных точек."""
    total, prev = 0.0, a
    for p in legs:
        total += math.hypot(p[0] - prev[0], p[1] - prev[1])
        prev = p
    return total


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class DroneOptimizer:
    """
    Оптимизирует маршруты флота БПЛА при заданном бюджете аппаратов.

    Parameters
    ----------
    uncovered_pts   : np.ndarray (K, 2) — координаты непокрытых точек
    max_drones      : int               — бюджет флота (число аппаратов)
    drone_range_m   : float             — максимальная длина маршрута (м)
    obs_radius_m    : float             — радиус полосы наблюдения (м)
    forbidden_zones : list[dict]        — запретные зоны
    detour_margin   : float             — отступ от края запретной зоны (м)
    """

    def __init__(self,
                 uncovered_pts: np.ndarray,
                 max_drones: int = 4,
                 drone_range_m: float = 50.0,
                 obs_radius_m: float = 12.0,
                 forbidden_zones: list | None = None,
                 detour_margin: float = 3.0):
        self.pts = np.asarray(uncovered_pts, dtype=float).reshape(-1, 2)
        self.max_drones = max(0, int(max_drones))
        self.drone_range = drone_range_m
        self.obs_radius = obs_radius_m
        self.forbidden_zones = forbidden_zones or []
        self.detour_margin = detour_margin

        # Результаты
        self.routes: list = []
        self.num_drones: int = 0
        self.covered_mask: np.ndarray = np.zeros(len(self.pts), dtype=bool)
        self.unreachable_legs: int = 0

    # ------------------------------------------------------------------
    def optimize(self) -> list:
        """
        Строит маршруты флота.

        Returns
        -------
        routes : list[list[tuple]] — маршруты, каждый список waypoints.
        """
        n = len(self.pts)
        if n == 0 or self.max_drones == 0:
            self.routes, self.num_drones = [], 0
            self.covered_mask = np.zeros(n, dtype=bool)
            return self.routes

        k = min(self.max_drones, n)
        labels = self._cluster(k)

        routes = []
        observed = np.zeros(n, dtype=bool)

        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            route = self._route_for_cluster(idx, observed)
            if route and len(route) > 0:
                routes.append(route)
                observed |= points_covered_by_route(self.pts, route,
                                                    self.obs_radius)

        self.routes = routes
        self.num_drones = len(routes)
        self.covered_mask = observed
        return self.routes

    # ------------------------------------------------------------------
    def _cluster(self, k: int) -> np.ndarray:
        """Кластеризует точки на k групп."""
        if k <= 1:
            return np.zeros(len(self.pts), dtype=int)
        if k >= len(self.pts):
            return np.arange(len(self.pts), dtype=int)
        if _SK_AVAILABLE:
            km = _SKLearnKMeans(n_clusters=k, random_state=42, n_init=10)
            return km.fit_predict(self.pts)
        return _simple_kmeans(self.pts, k)

    # ------------------------------------------------------------------
    def _build_route(self, targets: list, cluster_pts: np.ndarray) -> tuple:
        """
        Строит ломаную маршрута по заданной последовательности целевых точек,
        вставляя обходы запретных зон.

        Returns
        -------
        (waypoints, length) либо (None, inf), если обход не найден.
        """
        if not targets:
            return None, float('inf')

        first = (float(cluster_pts[targets[0]][0]),
                 float(cluster_pts[targets[0]][1]))
        route, length = [first], 0.0

        for t in targets[1:]:
            target = (float(cluster_pts[t][0]), float(cluster_pts[t][1]))
            legs = plan_leg(route[-1], target,
                            self.forbidden_zones, self.detour_margin)
            if legs is None:
                return None, float('inf')
            length += leg_cost(route[-1], legs)
            route.extend(legs)

        return route, length

    # ------------------------------------------------------------------
    def _greedy_extend(self, targets: list, length: float,
                       cluster_pts: np.ndarray,
                       observed: np.ndarray) -> tuple:
        """
        Наращивает последовательность целевых точек по правилу «максимальный
        прирост наблюдаемости на единицу длины маршрута».

        В отличие от чистого ближайшего соседа, критерий учитывает не только
        стоимость перелёта, но и то, сколько новых точек попадёт в полосу
        наблюдения при движении к кандидату: дальняя точка, по пути к которой
        захватывается несколько соседних, предпочтительнее близкой одиночной.
        """
        route, _ = self._build_route(targets, cluster_pts)
        obs = observed.copy()
        if route:
            obs |= points_covered_by_route(cluster_pts, route, self.obs_radius)

        while not obs.all():
            current = route[-1]
            best = None   # (плотность прироста, cost, cand, legs, gain_mask)

            for cand in np.where(~obs)[0]:
                target = (float(cluster_pts[cand][0]),
                          float(cluster_pts[cand][1]))
                legs = plan_leg(current, target,
                                self.forbidden_zones, self.detour_margin)
                if legs is None:
                    self.unreachable_legs += 1
                    obs[cand] = True
                    best = 'skip'
                    break

                cost = leg_cost(current, legs)
                if length + cost > self.drone_range + 1e-9:
                    continue

                # Новые точки, захватываемые добавляемым участком
                added = np.zeros(len(cluster_pts), dtype=bool)
                prev = current
                for wp in legs:
                    added |= points_covered_by_route(cluster_pts, [prev, wp],
                                                     self.obs_radius)
                    prev = wp
                gain = int((added & ~obs).sum())
                if gain == 0:
                    continue

                density = gain / max(cost, 1e-6)
                if best is None or best == 'skip' or density > best[0]:
                    best = (density, cost, cand, legs, added)

            if best == 'skip':
                continue
            if best is None:
                break

            _, cost, cand, legs, added = best
            route.extend(legs)
            length += cost
            targets.append(int(cand))
            obs |= added

        return targets, length, route, obs

    # ------------------------------------------------------------------
    def _prize(self, route: list, cluster_pts: np.ndarray,
               base: np.ndarray) -> int:
        """Число точек кластера, наблюдаемых с маршрута (сверх уже покрытых)."""
        if route is None:
            return -1
        mask = points_covered_by_route(cluster_pts, route, self.obs_radius)
        return int((mask & ~base).sum())

    # ------------------------------------------------------------------
    def _local_search(self, targets: list, length: float,
                      cluster_pts: np.ndarray, base: np.ndarray) -> tuple:
        """
        Локальный поиск по окрестностям 2-opt (разворот участка) и Or-opt
        (перенос одной точки) с лексикографическим критерием приёма:
        сначала наблюдаемость, затем длина.

        Ход принимается, если он увеличивает число наблюдаемых точек либо
        сохраняет его и сокращает маршрут. Чисто длиновой критерий здесь
        неприменим: решаемая подзадача — максимизация числа обслуженных
        точек при ограничении на ресурс (задача об ориентировании), а не
        минимизация длины замкнутого тура, поэтому сокращение маршрута
        может выводить точки из полосы наблюдения.
        """
        if len(targets) < 3:
            return targets, length

        best = list(targets)
        best_route, best_len = self._build_route(best, cluster_pts)
        best_prize = self._prize(best_route, cluster_pts, base)

        def better(prize, ln):
            if prize > best_prize:
                return True
            return prize == best_prize and ln < best_len - 1e-6

        improved, rounds = True, 0
        while improved and rounds < 8:
            improved = False
            rounds += 1

            # 2-opt: разворот участка маршрута
            for i in range(1, len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    route, ln = self._build_route(cand, cluster_pts)
                    if route is None or ln > self.drone_range + 1e-9:
                        continue
                    prize = self._prize(route, cluster_pts, base)
                    if better(prize, ln):
                        best, best_len, best_prize = cand, ln, prize
                        improved = True

            # Замена: посещаемая точка меняется на непосещаемую.
            # Характерный ход для задачи об ориентировании: удалённая
            # точка, ведущая к тупику, уступает место точке, с которой
            # в полосу наблюдения попадает больше соседей.
            visited = set(best)
            for i in range(1, len(best)):
                for cand_node in range(len(cluster_pts)):
                    if cand_node in visited:
                        continue
                    cand = best[:i] + [cand_node] + best[i + 1:]
                    route, ln = self._build_route(cand, cluster_pts)
                    if route is None or ln > self.drone_range + 1e-9:
                        continue
                    prize = self._prize(route, cluster_pts, base)
                    if better(prize, ln):
                        best, best_len, best_prize = cand, ln, prize
                        visited = set(best)
                        improved = True
                        break

            # Or-opt: перенос одной точки в другую позицию
            for i in range(1, len(best)):
                rest = best[:i] + best[i + 1:]
                node = best[i]
                for j in range(1, len(rest) + 1):
                    cand = rest[:j] + [node] + rest[j:]
                    if cand == best:
                        continue
                    route, ln = self._build_route(cand, cluster_pts)
                    if route is None or ln > self.drone_range + 1e-9:
                        continue
                    prize = self._prize(route, cluster_pts, base)
                    if better(prize, ln):
                        best, best_len, best_prize = cand, ln, prize
                        improved = True
                        break

        return best, best_len

    # ------------------------------------------------------------------
    def _route_for_cluster(self, idx: np.ndarray,
                           already_observed: np.ndarray) -> list:
        """
        Строит маршрут одного аппарата в пределах лимита длины.

        Схема: жадное наращивание по плотности прироста наблюдаемости →
        локальный поиск 2-opt/Or-opt с лексикографическим критерием
        (наблюдаемость, затем длина) → повторное наращивание за счёт
        высвободившейся дальности. Точки, уже попавшие в полосу наблюдения,
        не облётываются повторно.
        """
        cluster_pts = self.pts[idx]
        centroid = cluster_pts.mean(axis=0)
        start_local = int(np.hypot(cluster_pts[:, 0] - centroid[0],
                                   cluster_pts[:, 1] - centroid[1]).argmin())

        observed = already_observed[idx].copy()
        targets = [start_local]
        length = 0.0

        base = observed.copy()
        for _ in range(4):
            targets, length, route, _ = self._greedy_extend(
                targets, length, cluster_pts, base)
            new_targets, new_len = self._local_search(
                targets, length, cluster_pts, base)
            gained = (new_targets != targets) or (new_len < length - 1e-6)
            targets, length = new_targets, new_len
            if not gained:
                break

        route, length = self._build_route(targets, cluster_pts)
        return route if route else [(float(cluster_pts[start_local][0]),
                                     float(cluster_pts[start_local][1]))]

    # ------------------------------------------------------------------
    def covered_count(self) -> int:
        """Число непокрытых камерами точек, наблюдаемых с маршрутов БПЛА."""
        return int(self.covered_mask.sum())

    # ------------------------------------------------------------------
    def max_route_length(self) -> float:
        """Длина самого длинного маршрута (м)."""
        return max((route_length(r) for r in self.routes), default=0.0)

    # ------------------------------------------------------------------
    def routes_valid(self) -> bool:
        """True, если все маршруты в пределах лимита и вне запретных зон."""
        return all(route_length(r) <= self.drone_range + 1e-6
                   and route_is_clear(r, self.forbidden_zones)
                   for r in self.routes)

    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        """Возвращает словарь с результатами оптимизации БПЛА."""
        return {
            'num_drones':        self.num_drones,
            'routes':            [
                {'waypoints': [{'x': p[0], 'y': p[1]} for p in route],
                 'length_m': round(route_length(route), 1)}
                for route in self.routes
            ],
            'uncovered_pts_cnt': len(self.pts),
            'observed_pts_cnt':  self.covered_count(),
            'max_route_len_m':   round(self.max_route_length(), 1),
            'obs_radius_m':      self.obs_radius,
            'routes_valid':      self.routes_valid(),
            'solver':            'K-means + Nearest Neighbor (budgeted)',
        }
