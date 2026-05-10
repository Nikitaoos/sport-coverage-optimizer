"""
drone_optimizer.py
------------------
Оптимизация состава и маршрутов БПЛА для покрытия зон,
недоступных стационарным камерам.

Алгоритм:
1. Получает непокрытые точки T' ⊆ T.
2. Если T' пусто — БПЛА не нужны.
3. Кластеризует T' методом K-means.
   Число кластеров K определяется из условия:
       K = ceil(len(T') / max_points_per_drone)
   где max_points_per_drone — оценочное число точек, которые БПЛА
   может охватить за один пролёт (зависит от дальности и шага сетки).
4. Для каждого кластера строит маршрут облёта алгоритмом ближайшего соседа.
5. Проверяет маршрут на пересечение с запретными зонами; при необходимости
   перестраивает с обходом запретной зоны.
6. Если длина маршрута превышает дальность БПЛА — разбивает на сегменты
   (+1 БПЛА на каждый лишний сегмент).

Если установлен scikit-learn — используется его KMeans.
Иначе применяется встроенная реализация (простой lloyd's algorithm).
"""

import math
import numpy as np

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

    labels = np.zeros(len(points), dtype=int)
    for _ in range(max_iter):
        # Назначение
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Пересчёт центроидов
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = points[mask].mean(axis=0)

    return labels


# ---------------------------------------------------------------------------
# Алгоритм ближайшего соседа (TSP-эвристика)
# ---------------------------------------------------------------------------

def _nearest_neighbor_route(points: np.ndarray) -> list[int]:
    """
    Строит маршрут обхода точек методом ближайшего соседа.
    Возвращает список индексов в порядке обхода.
    """
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]

    visited = [False] * n
    route = [0]
    visited[0] = True

    for _ in range(n - 1):
        last = route[-1]
        best_dist, best_j = float('inf'), -1
        for j in range(n):
            if not visited[j]:
                d = math.hypot(points[j][0] - points[last][0],
                               points[j][1] - points[last][1])
                if d < best_dist:
                    best_dist, best_j = d, j
        if best_j == -1:
            break
        route.append(best_j)
        visited[best_j] = True

    return route


def _route_length(points: np.ndarray, order: list[int]) -> float:
    """Суммарная длина маршрута (м)."""
    total = 0.0
    for k in range(len(order) - 1):
        a, b = points[order[k]], points[order[k + 1]]
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


# ---------------------------------------------------------------------------
# Обход запретной зоны (упрощённый waypoint-метод)
# ---------------------------------------------------------------------------

def _reroute_around_forbidden(wp_from: tuple, wp_to: tuple,
                               forbidden_zones: list[dict]) -> list[tuple]:
    """
    Если отрезок wp_from -> wp_to пересекает запретную зону,
    возвращает промежуточные точки для обхода.
    Используется простой обход через угол AABB запретной зоны.
    """
    from .spatial_model import rect_intersects_forbidden

    if not rect_intersects_forbidden(wp_from[0], wp_from[1],
                                      wp_to[0], wp_to[1],
                                      forbidden_zones):
        return [wp_to]

    # Выбираем ближайший угол запретной зоны как промежуточную точку
    best_corner, best_d = None, float('inf')
    for fz in forbidden_zones:
        fx, fy, fw, fh = fz['x'], fz['y'], fz['width'], fz['height']
        margin = 3.0  # отступ от края (м)
        corners = [
            (fx - margin, fy - margin),
            (fx + fw + margin, fy - margin),
            (fx - margin, fy + fh + margin),
            (fx + fw + margin, fy + fh + margin),
        ]
        for cx, cy in corners:
            d = math.hypot(cx - wp_from[0], cy - wp_from[1])
            if d < best_d:
                best_d = d
                best_corner = (cx, cy)

    if best_corner is None:
        return [wp_to]
    return [best_corner, wp_to]


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class DroneOptimizer:
    """
    Оптимизирует состав и маршруты БПЛА для покрытия непокрытых точек.

    Parameters
    ----------
    uncovered_pts   : np.ndarray (K, 2) — координаты непокрытых точек
    drone_range_m   : float             — максимальная дальность маршрута БПЛА (м)
    grid_step       : float             — шаг сетки (м), для оценки числа кластеров
    forbidden_zones : list[dict]        — запретные зоны
    """

    def __init__(self,
                 uncovered_pts: np.ndarray,
                 drone_range_m: float = 50.0,
                 grid_step: float = 5.0,
                 forbidden_zones: list[dict] = None):
        self.pts = uncovered_pts
        self.drone_range = drone_range_m
        self.grid_step = grid_step
        self.forbidden_zones = forbidden_zones or []

        # Результаты
        self.routes: list[list[tuple]] = []  # список маршрутов (каждый — список waypoints)
        self.num_drones: int = 0

    # ------------------------------------------------------------------
    def optimize(self) -> list[list[tuple]]:
        """
        Запускает оптимизацию.
        Возвращает список маршрутов БПЛА.
        Каждый маршрут — список координатных точек [(x1,y1), (x2,y2), ...].
        """
        if len(self.pts) == 0:
            self.routes = []
            self.num_drones = 0
            return self.routes

        # Оценка числа кластеров
        area_per_drone = math.pi * (self.drone_range / 2) ** 2
        pts_per_drone = max(1, int(area_per_drone / (self.grid_step ** 2)))
        k = max(1, math.ceil(len(self.pts) / pts_per_drone))
        k = min(k, len(self.pts))  # не больше числа точек

        # Кластеризация
        labels = self._cluster(k)

        # Построение маршрутов по кластерам
        all_routes = []
        for c in range(k):
            cluster_pts = self.pts[labels == c]
            if len(cluster_pts) == 0:
                continue
            cluster_routes = self._build_cluster_routes(cluster_pts)
            all_routes.extend(cluster_routes)

        self.routes = all_routes
        self.num_drones = len(all_routes)
        return self.routes

    # ------------------------------------------------------------------
    def _cluster(self, k: int) -> np.ndarray:
        """Кластеризует точки на k групп."""
        if k == 1:
            return np.zeros(len(self.pts), dtype=int)
        if _SK_AVAILABLE:
            km = _SKLearnKMeans(n_clusters=k, random_state=42, n_init='auto')
            return km.fit_predict(self.pts)
        else:
            return _simple_kmeans(self.pts, k)

    # ------------------------------------------------------------------
    def _build_cluster_routes(self, cluster_pts: np.ndarray) -> list[list[tuple]]:
        """
        Строит маршруты для одного кластера.
        Если маршрут слишком длинный — разбивает на сегменты.
        """
        order = _nearest_neighbor_route(cluster_pts)

        # Строим waypoints с обходом запретных зон
        waypoints = []
        for idx in order:
            pt = (float(cluster_pts[idx][0]), float(cluster_pts[idx][1]))
            if waypoints:
                detour = _reroute_around_forbidden(waypoints[-1], pt,
                                                   self.forbidden_zones)
                waypoints.extend(detour)
            else:
                waypoints.append(pt)

        # Разбивка на сегменты по дальности
        routes = []
        current_route = [waypoints[0]]
        current_len = 0.0

        for i in range(1, len(waypoints)):
            step = math.hypot(waypoints[i][0] - waypoints[i-1][0],
                              waypoints[i][1] - waypoints[i-1][1])
            if current_len + step > self.drone_range and len(current_route) > 1:
                routes.append(current_route)
                current_route = [waypoints[i-1], waypoints[i]]
                current_len = step
            else:
                current_route.append(waypoints[i])
                current_len += step

        if current_route:
            routes.append(current_route)

        return routes

    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        """Возвращает словарь с результатами оптимизации БПЛА."""
        return {
            'num_drones':        self.num_drones,
            'routes':            [
                {'waypoints': [{'x': p[0], 'y': p[1]} for p in route]}
                for route in self.routes
            ],
            'uncovered_pts_cnt': len(self.pts),
            'solver':            'K-means + Nearest Neighbor',
        }
