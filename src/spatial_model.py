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

Зона покрытия камеры моделируется как сектор окружности:
    - центр в точке установки (cx, cy);
    - радиус = range_m;
    - угловой раствор = angle_h (горизонтальный угол обзора);
    - ориентация по азимуту (0° = север = +Y, по часовой стрелке).
"""

import math
import xml.etree.ElementTree as ET
import numpy as np


# ---------------------------------------------------------------------------
# Вспомогательные геометрические функции (без Shapely)
# ---------------------------------------------------------------------------

def _azimuth_to_math(azimuth_deg: float) -> float:
    """
    Перевод азимута (0=север, по часовой) в математический угол
    (0=восток (+X), против часовой стрелки), в радианах.
    """
    math_deg = 90.0 - azimuth_deg
    return math.radians(math_deg)


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
    if dist > range_m or dist < 1e-6:
        return False

    # Угол от камеры до точки (азимут, °)
    # atan2 возвращает угол от +X, переводим в азимут
    point_azimuth = math.degrees(math.atan2(dx, dy))  # atan2(dx,dy): север=0, восток=90
    point_azimuth = point_azimuth % 360.0

    # Разница азимутов с учётом цикличности
    diff = abs(point_azimuth - azimuth_deg) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff

    return diff <= half_angle_deg


def rect_intersects_forbidden(x1, y1, x2, y2,
                               forbidden_zones: list[dict]) -> bool:
    """
    Проверяет, пересекает ли отрезок (x1,y1)-(x2,y2) хотя бы одну
    запретную прямоугольную зону.
    Используется для проверки маршрутов БПЛА.
    """
    for fz in forbidden_zones:
        fx, fy, fw, fh = fz['x'], fz['y'], fz['width'], fz['height']
        # AABB-тест отрезка
        if _segment_intersects_rect(x1, y1, x2, y2, fx, fy, fx + fw, fy + fh):
            return True
    return False


def _segment_intersects_rect(x1, y1, x2, y2,
                               rx_min, ry_min, rx_max, ry_max) -> bool:
    """Cohen-Sutherland clip test — проверка пересечения отрезка с AABB."""
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def code(x, y):
        c = INSIDE
        if x < rx_min: c |= LEFT
        elif x > rx_max: c |= RIGHT
        if y < ry_min: c |= BOTTOM
        elif y > ry_max: c |= TOP
        return c

    c1, c2 = code(x1, y1), code(x2, y2)
    while True:
        if not (c1 | c2):   return True   # оба внутри
        if c1 & c2:         return False  # оба снаружи одной полуплоскости
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


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class SpatialModel:
    """
    Геометрическая модель спортивного объекта.

    Attributes
    ----------
    venue_name      : название объекта
    venue_type      : тип объекта (stadium / hall / court)
    sectors         : list[dict] — зоны наблюдения
    cam_positions   : list[dict] — кандидатные позиции камер
    forbidden_zones : list[dict] — запретные зоны для БПЛА
    drone_range_m   : дальность БПЛА (м)
    drone_cost      : стоимость одного БПЛА (руб.)
    grid            : np.ndarray shape (N, 2) — координаты точек наблюдения
    coverage_matrix : np.ndarray shape (N, M) — матрица покрытия
    """

    def __init__(self, xml_path: str, grid_step: float = 5.0):
        self.xml_path = xml_path
        self.grid_step = grid_step

        self.venue_name = ''
        self.venue_type = ''
        self.sectors: list[dict] = []
        self.cam_positions: list[dict] = []
        self.forbidden_zones: list[dict] = []
        self.drone_range_m: float = 50.0
        self.drone_cost: float = 15000.0

        self.grid: np.ndarray = np.empty((0, 2))
        self.coverage_matrix: np.ndarray = np.empty((0, 0))

        self._load_xml()
        self.build_grid()
        self.compute_coverage()

    # ------------------------------------------------------------------
    def _load_xml(self):
        """Парсит XML-файл описания объекта."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

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
            self.drone_range_m = float(dp.get('range_m', 50))
            self.drone_cost    = float(dp.get('cost_per_unit', 15000))

    # ------------------------------------------------------------------
    def build_grid(self):
        """
        Строит равномерную сетку точек наблюдения по всем зонам (секторам).
        Шаг сетки = self.grid_step (м).
        """
        points = []
        for s in self.sectors:
            xs = np.arange(s['x'] + self.grid_step / 2,
                           s['x'] + s['width'],
                           self.grid_step)
            ys = np.arange(s['y'] + self.grid_step / 2,
                           s['y'] + s['height'],
                           self.grid_step)
            for x in xs:
                for y in ys:
                    points.append([x, y])

        self.grid = np.array(points, dtype=float) if points else np.empty((0, 2))

    # ------------------------------------------------------------------
    def compute_coverage(self):
        """
        Вычисляет матрицу покрытия a[i][j]:
            a[i][j] = 1, если камера j покрывает точку наблюдения i.

        Returns
        -------
        coverage_matrix : np.ndarray shape (N, M), dtype bool
        """
        N = len(self.grid)
        M = len(self.cam_positions)
        A = np.zeros((N, M), dtype=np.int8)

        for j, cam in enumerate(self.cam_positions):
            cx, cy       = cam['x'], cam['y']
            azimuth      = cam['azimuth']
            half_angle   = cam['angle_h'] / 2.0
            range_m      = cam['range_m']

            for i, (px, py) in enumerate(self.grid):
                if point_in_sector(px, py, cx, cy, azimuth, half_angle, range_m):
                    A[i, j] = 1

        self.coverage_matrix = A
        return A

    # ------------------------------------------------------------------
    def get_uncovered_points(self, selected: np.ndarray) -> np.ndarray:
        """
        Возвращает индексы точек наблюдения, не покрытых выбранными камерами.

        Parameters
        ----------
        selected : np.ndarray of int — индексы выбранных позиций камер

        Returns
        -------
        uncovered_indices : np.ndarray of int
        """
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
