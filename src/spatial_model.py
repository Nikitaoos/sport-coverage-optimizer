"""
spatial_model.py
----------------
Геометрическая модель спортивного объекта.

Отвечает за:
- загрузку XML-описания объекта (зоны наблюдения, кандидатные позиции камер,
  запретные зоны, параметры БПЛА);
- построение равномерной сетки точек наблюдения T по всем зонам;
- вычисление бинарной матрицы покрытия a[i][j]:
    a[i][j] = 1, если камера в позиции j покрывает точку i, иначе 0.

Зона покрытия стационарной камеры моделируется как сектор окружности:
    - центр в точке установки (cx, cy);
    - радиус = range_m;
    - угловой раствор = angle_h (горизонтальный угол обзора);
    - ориентация по азимуту (0° = север = +Y, по часовой стрелке).

Зона наблюдения БПЛА моделируется как полоса (буфер) вдоль маршрута:
точка считается наблюдаемой, если расстояние от неё до ломаной маршрута
не превышает obs_radius_m. Это соответствует съёмке камерой с постоянной
высоты при непрерывном пролёте.
"""

import math
import xml.etree.ElementTree as ET
import numpy as np


# ---------------------------------------------------------------------------
# Вспомогательные геометрические функции (без Shapely)
# ---------------------------------------------------------------------------

def point_in_sector(px: float, py: float,
                    cx: float, cy: float,
                    azimuth_deg: float, half_angle_deg: float,
                    range_m: float) -> bool:
    """
    Проверяет, попадает ли точка (px, py) в сектор обзора камеры.

    Parameters
    ----------
    px, py          : координаты проверяемой точки
    cx, cy          : координаты камеры
    azimuth_deg     : ориентация оси объектива (азимут, °)
    half_angle_deg  : половина горизонтального угла обзора (°)
    range_m         : дальность обзора (м)
    """
    dx, dy = px - cx, py - cy
    dist = math.hypot(dx, dy)
    if dist > range_m:
        return False
    if dist < 1e-9:
        # Точка совпадает с позицией камеры — считается наблюдаемой.
        return True

    # atan2(dx, dy) даёт азимут: север = 0, восток = 90
    point_azimuth = math.degrees(math.atan2(dx, dy)) % 360.0

    diff = abs(point_azimuth - azimuth_deg) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff

    return diff <= half_angle_deg


def points_in_sector(points: np.ndarray,
                     cx: float, cy: float,
                     azimuth_deg: float, half_angle_deg: float,
                     range_m: float) -> np.ndarray:
    """
    Векторизованная версия point_in_sector для массива точек.

    Parameters
    ----------
    points : np.ndarray shape (N, 2)

    Returns
    -------
    mask : np.ndarray shape (N,), dtype bool
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    dist = np.hypot(dx, dy)

    in_range = dist <= range_m

    azim = np.degrees(np.arctan2(dx, dy)) % 360.0
    diff = np.abs(azim - azimuth_deg) % 360.0
    diff = np.minimum(diff, 360.0 - diff)

    in_angle = diff <= half_angle_deg
    # Точка ровно в позиции камеры: азимут не определён, считаем наблюдаемой
    in_angle |= dist < 1e-9

    return in_range & in_angle


def point_segment_distance(points: np.ndarray,
                           a: tuple, b: tuple) -> np.ndarray:
    """
    Расстояние от каждой точки массива до отрезка a-b.

    Parameters
    ----------
    points : np.ndarray shape (N, 2)
    a, b   : концы отрезка (x, y)

    Returns
    -------
    dist : np.ndarray shape (N,)
    """
    if len(points) == 0:
        return np.zeros(0)

    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby

    apx = points[:, 0] - ax
    apy = points[:, 1] - ay

    if ab2 < 1e-12:
        return np.hypot(apx, apy)

    t = np.clip((apx * abx + apy * aby) / ab2, 0.0, 1.0)
    projx = ax + t * abx
    projy = ay + t * aby
    return np.hypot(points[:, 0] - projx, points[:, 1] - projy)


def points_covered_by_route(points: np.ndarray,
                            route: list,
                            obs_radius_m: float) -> np.ndarray:
    """
    Маска точек, наблюдаемых с маршрута БПЛА.

    Точка считается наблюдаемой, если расстояние от неё до ломаной маршрута
    не превышает obs_radius_m.

    Parameters
    ----------
    points       : np.ndarray shape (N, 2)
    route        : list[tuple] — waypoints маршрута
    obs_radius_m : радиус зоны наблюдения БПЛА (м)

    Returns
    -------
    mask : np.ndarray shape (N,), dtype bool
    """
    mask = np.zeros(len(points), dtype=bool)
    if len(points) == 0 or not route:
        return mask

    if len(route) == 1:
        return np.hypot(points[:, 0] - route[0][0],
                        points[:, 1] - route[0][1]) <= obs_radius_m

    for a, b in zip(route[:-1], route[1:]):
        mask |= point_segment_distance(points, a, b) <= obs_radius_m
    return mask


def segment_intersects_rect(x1, y1, x2, y2,
                            rx_min, ry_min, rx_max, ry_max) -> bool:
    """Cohen-Sutherland clip test — проверка пересечения отрезка с AABB."""
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def code(x, y):
        c = INSIDE
        if x < rx_min:
            c |= LEFT
        elif x > rx_max:
            c |= RIGHT
        if y < ry_min:
            c |= BOTTOM
        elif y > ry_max:
            c |= TOP
        return c

    c1, c2 = code(x1, y1), code(x2, y2)
    while True:
        if not (c1 | c2):
            return True   # оба конца внутри прямоугольника
        if c1 & c2:
            return False  # оба снаружи одной полуплоскости
        c_out = c1 if c1 else c2
        if c_out & TOP:
            x = x1 + (x2 - x1) * (ry_max - y1) / (y2 - y1 + 1e-12)
            y = ry_max
        elif c_out & BOTTOM:
            x = x1 + (x2 - x1) * (ry_min - y1) / (y2 - y1 + 1e-12)
            y = ry_min
        elif c_out & RIGHT:
            y = y1 + (y2 - y1) * (rx_max - x1) / (x2 - x1 + 1e-12)
            x = rx_max
        else:
            y = y1 + (y2 - y1) * (rx_min - x1) / (x2 - x1 + 1e-12)
            x = rx_min
        if c_out == c1:
            x1, y1, c1 = x, y, code(x, y)
        else:
            x2, y2, c2 = x, y, code(x, y)


def rect_intersects_forbidden(x1, y1, x2, y2,
                              forbidden_zones: list) -> bool:
    """
    Проверяет, пересекает ли отрезок (x1,y1)-(x2,y2) хотя бы одну
    запретную прямоугольную зону. Используется для проверки маршрутов БПЛА.
    """
    for fz in forbidden_zones:
        fx, fy, fw, fh = fz['x'], fz['y'], fz['width'], fz['height']
        if segment_intersects_rect(x1, y1, x2, y2, fx, fy, fx + fw, fy + fh):
            return True
    return False


def route_is_clear(route: list, forbidden_zones: list) -> bool:
    """True, если ни одно звено маршрута не пересекает запретные зоны."""
    for a, b in zip(route[:-1], route[1:]):
        if rect_intersects_forbidden(a[0], a[1], b[0], b[1], forbidden_zones):
            return False
    return True


def route_length(route: list) -> float:
    """Суммарная длина ломаной маршрута (м)."""
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(route[:-1], route[1:]))


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class SpatialModel:
    """
    Геометрическая модель спортивного объекта.

    Attributes
    ----------
    venue_name       : название объекта
    venue_type       : тип объекта (stadium / hall / court)
    sectors          : list[dict] — зоны наблюдения
    cam_positions    : list[dict] — кандидатные позиции камер
    forbidden_zones  : list[dict] — запретные зоны для БПЛА
    drone_range_m    : максимальная длина маршрута БПЛА (м)
    drone_obs_radius : радиус полосы наблюдения БПЛА (м)
    drone_max_units  : бюджет флота БПЛА (шт.)
    drone_cost       : стоимость одного БПЛА (руб.)
    grid             : np.ndarray shape (N, 2) — координаты точек наблюдения
    coverage_matrix  : np.ndarray shape (N, M) — матрица покрытия
    """

    def __init__(self, xml_path: str, grid_step: float = 5.0):
        self.xml_path = xml_path
        self.grid_step = grid_step

        self.venue_name = ''
        self.venue_type = ''
        self.sectors: list = []
        self.cam_positions: list = []
        self.forbidden_zones: list = []
        self.drone_range_m: float = 50.0
        self.drone_obs_radius: float = 12.0
        self.drone_max_units: int = 4
        self.drone_cost: float = 15000.0

        self.grid: np.ndarray = np.empty((0, 2))
        self.point_priority: np.ndarray = np.empty(0, dtype=int)
        self.coverage_matrix: np.ndarray = np.empty((0, 0), dtype=np.int8)

        self._load_xml()
        self.build_grid()
        self.compute_coverage()

    # ------------------------------------------------------------------
    def _load_xml(self):
        """Парсит XML-файл описания объекта."""
        try:
            tree = ET.parse(self.xml_path)
        except ET.ParseError as e:
            raise ValueError(
                f'Некорректный XML-файл "{self.xml_path}": {e}') from e
        except FileNotFoundError as e:
            raise ValueError(
                f'Файл описания объекта не найден: "{self.xml_path}"') from e

        root = tree.getroot()
        if root.tag != 'venue':
            raise ValueError(
                f'Ожидался корневой элемент <venue>, получен <{root.tag}>.')

        self.venue_name = root.get('name', 'Объект')
        self.venue_type = root.get('type', 'unknown')

        for s in root.findall('.//sector'):
            self.sectors.append({
                'id':       s.get('id'),
                'name':     s.get('name', ''),
                'x':        float(s.get('x', 0)),
                'y':        float(s.get('y', 0)),
                'width':    float(s.get('width', 0)),
                'height':   float(s.get('height', 0)),
                'priority': int(s.get('priority', 2)),
            })

        for c in root.findall('.//position'):
            self.cam_positions.append({
                'id':       c.get('id'),
                'x':        float(c.get('x', 0)),
                'y':        float(c.get('y', 0)),
                'azimuth':  float(c.get('azimuth', 0)),
                'angle_h':  float(c.get('angle_h', 80)),
                'range_m':  float(c.get('range_m', 30)),
                'cost':     float(c.get('cost', 80000)),
            })

        for fz in root.findall('.//zone'):
            self.forbidden_zones.append({
                'id':     fz.get('id'),
                'name':   fz.get('name', ''),
                'x':      float(fz.get('x', 0)),
                'y':      float(fz.get('y', 0)),
                'width':  float(fz.get('width', 0)),
                'height': float(fz.get('height', 0)),
            })

        dp = root.find('.//drone_params')
        if dp is not None:
            self.drone_range_m    = float(dp.get('range_m', 50))
            self.drone_obs_radius = float(dp.get('obs_radius_m', 12))
            self.drone_max_units  = int(dp.get('max_units', 4))
            self.drone_cost       = float(dp.get('cost_per_unit', 15000))

        if not self.sectors:
            raise ValueError('В описании объекта не задано ни одной зоны '
                             'наблюдения (<sector>).')
        if not self.cam_positions:
            raise ValueError('В описании объекта не задано ни одной '
                             'кандидатной позиции камеры (<position>).')

    # ------------------------------------------------------------------
    def build_grid(self):
        """
        Строит равномерную сетку точек наблюдения по всем зонам (секторам).
        Шаг сетки = self.grid_step (м). Точки, попадающие в несколько
        пересекающихся зон, дедуплицируются, чтобы не искажать долю покрытия;
        точке присваивается наивысший (наименьший по номеру) приоритет.
        """
        best: dict = {}
        for s in self.sectors:
            xs = np.arange(s['x'] + self.grid_step / 2,
                           s['x'] + s['width'],
                           self.grid_step)
            ys = np.arange(s['y'] + self.grid_step / 2,
                           s['y'] + s['height'],
                           self.grid_step)
            for x in xs:
                for y in ys:
                    pt = (round(float(x), 6), round(float(y), 6))
                    pr = s['priority']
                    if pt not in best or pr < best[pt]:
                        best[pt] = pr

        if best:
            ordered = sorted(best.items())
            self.grid = np.array([p for p, _ in ordered], dtype=float)
            self.point_priority = np.array([pr for _, pr in ordered], dtype=int)
        else:
            self.grid = np.empty((0, 2))
            self.point_priority = np.empty(0, dtype=int)

    # ------------------------------------------------------------------
    def compute_coverage(self) -> np.ndarray:
        """
        Вычисляет матрицу покрытия a[i][j]:
            a[i][j] = 1, если камера j покрывает точку наблюдения i.

        Реализация векторизована по точкам сетки (numpy).
        """
        N, M = len(self.grid), len(self.cam_positions)
        A = np.zeros((N, M), dtype=np.int8)

        for j, cam in enumerate(self.cam_positions):
            mask = points_in_sector(self.grid,
                                    cam['x'], cam['y'],
                                    cam['azimuth'], cam['angle_h'] / 2.0,
                                    cam['range_m'])
            A[:, j] = mask.astype(np.int8)

        self.coverage_matrix = A
        return A

    # ------------------------------------------------------------------
    def get_uncovered_points(self, selected: np.ndarray) -> np.ndarray:
        """Индексы точек наблюдения, не покрытых выбранными камерами."""
        if len(self.grid) == 0:
            return np.empty(0, dtype=int)
        if len(selected) == 0:
            return np.arange(len(self.grid))
        covered_mask = self.coverage_matrix[:, selected].any(axis=1)
        return np.where(~covered_mask)[0]

    # ------------------------------------------------------------------
    def coverage_pct(self, selected: np.ndarray) -> float:
        """Процент покрытых точек для данного набора камер."""
        if len(self.grid) == 0:
            return 0.0
        unc = self.get_uncovered_points(selected)
        return (1.0 - len(unc) / len(self.grid)) * 100.0
