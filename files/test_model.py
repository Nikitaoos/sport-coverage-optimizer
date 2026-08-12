"""
test_model.py
-------------
Модульные тесты оптимизационной модели.

Запуск:
    python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import evaluate, size_fleet                         # noqa: E402
from src import SpatialModel, StationaryOptimizer, DroneOptimizer, \
    EconomicBalancer, Configuration                           # noqa: E402
from src.drone_optimizer import plan_leg                       # noqa: E402
from src.spatial_model import (                               # noqa: E402
    point_in_sector, points_in_sector, point_segment_distance,
    points_covered_by_route, segment_intersects_rect,
    rect_intersects_forbidden, route_is_clear, route_length,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'data')
STADIUM = os.path.join(DATA, 'stadium_example.xml')
HALL = os.path.join(DATA, 'hall_example.xml')
COURT = os.path.join(DATA, 'court_example.xml')


class TestSectorGeometry(unittest.TestCase):
    """Геометрия сектора обзора стационарной камеры."""

    def test_point_straight_ahead_is_covered(self):
        # Камера в (0,0), смотрит на север (азимут 0), угол 90°, дальность 10
        self.assertTrue(point_in_sector(0, 5, 0, 0, 0, 45, 10))

    def test_point_beyond_range_is_not_covered(self):
        self.assertFalse(point_in_sector(0, 15, 0, 0, 0, 45, 10))

    def test_point_outside_angle_is_not_covered(self):
        # Точка на востоке (азимут 90) вне сектора ±45° вокруг севера
        self.assertFalse(point_in_sector(5, 0, 0, 0, 0, 45, 10))

    def test_angle_wraparound_at_zero(self):
        # Камера смотрит на азимут 350°, точка на азимуте 5° — разница 15°
        self.assertTrue(point_in_sector(0.87, 9.96, 0, 0, 350, 20, 15))

    def test_azimuth_convention_east_is_90(self):
        # Камера смотрит на восток: точка на востоке покрыта, на западе — нет
        self.assertTrue(point_in_sector(5, 0, 0, 0, 90, 30, 10))
        self.assertFalse(point_in_sector(-5, 0, 0, 0, 90, 30, 10))

    def test_vectorized_matches_scalar(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(-20, 20, size=(500, 2))
        mask = points_in_sector(pts, 1.0, 2.0, 137.0, 35.0, 12.0)
        expected = np.array([
            point_in_sector(p[0], p[1], 1.0, 2.0, 137.0, 35.0, 12.0)
            for p in pts])
        np.testing.assert_array_equal(mask, expected)


class TestSegmentGeometry(unittest.TestCase):
    """Расстояния до отрезка и пересечения с прямоугольниками."""

    def test_distance_to_segment_perpendicular(self):
        pts = np.array([[5.0, 3.0]])
        d = point_segment_distance(pts, (0, 0), (10, 0))
        self.assertAlmostEqual(d[0], 3.0)

    def test_distance_clamped_to_endpoint(self):
        # Проекция вне отрезка — расстояние до ближайшего конца
        pts = np.array([[15.0, 0.0]])
        d = point_segment_distance(pts, (0, 0), (10, 0))
        self.assertAlmostEqual(d[0], 5.0)

    def test_degenerate_segment_is_point(self):
        pts = np.array([[3.0, 4.0]])
        d = point_segment_distance(pts, (0, 0), (0, 0))
        self.assertAlmostEqual(d[0], 5.0)

    def test_route_coverage_band(self):
        pts = np.array([[5.0, 1.0], [5.0, 9.0]])
        mask = points_covered_by_route(pts, [(0, 0), (10, 0)], obs_radius_m=2.0)
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])

    def test_segment_crossing_rect_detected(self):
        self.assertTrue(segment_intersects_rect(-5, 5, 15, 5, 0, 0, 10, 10))

    def test_segment_beside_rect_not_detected(self):
        self.assertFalse(segment_intersects_rect(-5, 20, 15, 20, 0, 0, 10, 10))


class TestDetourPlanning(unittest.TestCase):
    """Обход запретных зон."""

    def setUp(self):
        self.zones = [{'x': 10, 'y': 0, 'width': 10, 'height': 20}]

    def test_clear_leg_returns_direct(self):
        legs = plan_leg((0, 30), (30, 30), self.zones)
        self.assertEqual(legs, [(30, 30)])

    def test_blocked_leg_gets_valid_detour(self):
        legs = plan_leg((0, 10), (30, 10), self.zones)
        self.assertIsNotNone(legs)
        route = [(0, 10)] + list(legs)
        self.assertTrue(route_is_clear(route, self.zones),
                        'обход не должен пересекать запретную зону')

    def test_detour_reaches_target(self):
        legs = plan_leg((0, 10), (30, 10), self.zones)
        self.assertEqual(legs[-1], (30, 10))


class TestStationaryOptimizer(unittest.TestCase):
    """Оптимизация размещения стационарных камер."""

    def setUp(self):
        self.model = SpatialModel(STADIUM, grid_step=5.0)
        self.costs = [c['cost'] for c in self.model.cam_positions]

    def test_budget_is_respected(self):
        for k in (1, 3, 8, 12):
            opt = StationaryOptimizer(self.model.coverage_matrix, self.costs, k)
            sel = opt.solve()
            self.assertLessEqual(len(sel), k)

    def test_budget_capped_by_positions(self):
        opt = StationaryOptimizer(self.model.coverage_matrix, self.costs, 999)
        sel = opt.solve()
        self.assertLessEqual(len(sel), len(self.model.cam_positions))

    def test_coverage_is_monotone_in_budget(self):
        prev = -1.0
        for k in (2, 4, 8, 12):
            opt = StationaryOptimizer(self.model.coverage_matrix, self.costs, k)
            opt.solve()
            self.assertGreaterEqual(opt.coverage_pct + 1e-9, prev,
                                    'покрытие не должно падать при росте бюджета')
            prev = opt.coverage_pct

    def test_greedy_not_better_than_exact(self):
        """Жадное решение не должно превосходить точное решение ЦЛП."""
        import src.stationary_optimizer as so
        if not so._PULP_AVAILABLE:
            self.skipTest('PuLP не установлен — сравнение недоступно')

        opt = StationaryOptimizer(self.model.coverage_matrix, self.costs, 8)
        exact = opt._solve_ilp()
        exact_cov = opt.coverage_pct

        opt2 = StationaryOptimizer(self.model.coverage_matrix, self.costs, 8)
        opt2._solve_greedy()
        self.assertLessEqual(opt2.coverage_pct, exact_cov + 1e-9)
        self.assertGreater(len(exact), 0)

    def test_covered_points_matches_matrix(self):
        opt = StationaryOptimizer(self.model.coverage_matrix, self.costs, 6)
        sel = opt.solve()
        expected = int(self.model.coverage_matrix[:, sel].any(axis=1).sum())
        self.assertEqual(opt.covered_points, expected)


class TestDroneOptimizer(unittest.TestCase):
    """Маршрутизация БПЛА."""

    def setUp(self):
        self.model = SpatialModel(STADIUM, grid_step=5.0)
        costs = [c['cost'] for c in self.model.cam_positions]
        opt = StationaryOptimizer(self.model.coverage_matrix, costs, 12)
        sel = opt.solve()
        self.uncovered = self.model.grid[self.model.get_uncovered_points(sel)]

    def test_fleet_budget_is_respected(self):
        for kd in (1, 2, 4):
            d = DroneOptimizer(self.uncovered, max_drones=kd,
                               drone_range_m=50, obs_radius_m=10,
                               forbidden_zones=self.model.forbidden_zones)
            d.optimize()
            self.assertLessEqual(d.num_drones, kd)

    def test_routes_within_range_limit(self):
        d = DroneOptimizer(self.uncovered, max_drones=2,
                           drone_range_m=30, obs_radius_m=8,
                           forbidden_zones=self.model.forbidden_zones)
        d.optimize()
        for r in d.routes:
            self.assertLessEqual(route_length(r), 30 + 1e-6)

    def test_routes_avoid_forbidden_zones(self):
        d = DroneOptimizer(self.uncovered, max_drones=3,
                           drone_range_m=50, obs_radius_m=8,
                           forbidden_zones=self.model.forbidden_zones)
        d.optimize()
        for r in d.routes:
            self.assertTrue(route_is_clear(r, self.model.forbidden_zones))

    def test_zero_budget_gives_no_routes(self):
        d = DroneOptimizer(self.uncovered, max_drones=0)
        d.optimize()
        self.assertEqual(d.num_drones, 0)
        self.assertEqual(d.covered_count(), 0)

    def test_empty_input_is_handled(self):
        d = DroneOptimizer(np.empty((0, 2)), max_drones=4)
        d.optimize()
        self.assertEqual(d.num_drones, 0)

    def test_coverage_is_not_assumed(self):
        """
        Ключевой тест: при жёстких ограничениях часть точек ОБЯЗАНА остаться
        ненаблюдаемой. Ранее покрытие БПЛА постулировалось (все непокрытые
        точки объявлялись покрытыми), из-за чего совокупное покрытие всегда
        равнялось 100 %.
        """
        d = DroneOptimizer(self.uncovered, max_drones=1,
                           drone_range_m=5, obs_radius_m=2,
                           forbidden_zones=self.model.forbidden_zones)
        d.optimize()
        self.assertLess(d.covered_count(), len(self.uncovered))

    def test_local_search_never_reduces_observability(self):
        """
        Локальный поиск оптимизирует наблюдаемость, а не длину: ход
        принимается только если число наблюдаемых точек не уменьшается.
        Чистый 2-opt, минимизирующий длину, этого не гарантирует.
        """
        d = DroneOptimizer(self.uncovered, max_drones=1,
                           drone_range_m=60, obs_radius_m=4,
                           forbidden_zones=self.model.forbidden_zones)
        d.optimize()
        cluster = d.pts
        base = np.zeros(len(cluster), dtype=bool)
        targets = list(range(min(6, len(cluster))))
        route_before, len_before = d._build_route(targets, cluster)
        prize_before = d._prize(route_before, cluster, base)

        improved, new_len = d._local_search(targets, len_before, cluster, base)
        route_after, _ = d._build_route(improved, cluster)
        prize_after = d._prize(route_after, cluster, base)

        self.assertGreaterEqual(prize_after, prize_before)
        if prize_after == prize_before:
            self.assertLessEqual(new_len, len_before + 1e-6)

    def test_local_search_respects_range_limit(self):
        d = DroneOptimizer(self.uncovered, max_drones=1,
                           drone_range_m=30, obs_radius_m=4,
                           forbidden_zones=self.model.forbidden_zones)
        d.optimize()
        cluster = d.pts
        base = np.zeros(len(cluster), dtype=bool)
        targets = list(range(min(5, len(cluster))))
        route, ln = d._build_route(targets, cluster)
        improved, new_len = d._local_search(targets, ln, cluster, base)
        if improved != targets:
            self.assertLessEqual(new_len, 30 + 1e-6)

    def test_routing_is_exercised_at_narrow_swath(self):
        """
        При узкой полосе наблюдения аппарат обязан именно облетать точки,
        а не зависать: маршрут должен иметь ненулевую длину.
        """
        d = DroneOptimizer(self.uncovered, max_drones=1,
                           drone_range_m=50, obs_radius_m=4,
                           forbidden_zones=self.model.forbidden_zones)
        d.optimize()
        self.assertGreater(d.max_route_length(), 0.0)

    def test_more_budget_never_hurts(self):
        cov = []
        for kd in (1, 2, 4):
            d = DroneOptimizer(self.uncovered, max_drones=kd,
                               drone_range_m=40, obs_radius_m=6,
                               forbidden_zones=self.model.forbidden_zones)
            d.optimize()
            cov.append(d.covered_count())
        self.assertLessEqual(cov[0], cov[-1])


class TestSpatialModel(unittest.TestCase):
    """Загрузка описания объекта и построение сетки."""

    def test_grid_points_are_unique(self):
        model = SpatialModel(STADIUM, grid_step=5.0)
        uniq = np.unique(model.grid, axis=0)
        self.assertEqual(len(uniq), len(model.grid),
                         'точки пересекающихся зон не должны дублироваться')

    def test_grid_refines_with_smaller_step(self):
        coarse = SpatialModel(STADIUM, grid_step=5.0)
        fine = SpatialModel(STADIUM, grid_step=2.5)
        self.assertGreater(len(fine.grid), len(coarse.grid))

    def test_coverage_matrix_shape(self):
        model = SpatialModel(STADIUM, grid_step=5.0)
        self.assertEqual(model.coverage_matrix.shape,
                         (len(model.grid), len(model.cam_positions)))

    def test_invalid_xml_raises_value_error(self):
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False,
                                         encoding='utf-8') as f:
            f.write('<venue name="x"><sectors></sectors></venue>')
            path = f.name
        try:
            with self.assertRaises(ValueError):
                SpatialModel(path)
        finally:
            os.unlink(path)

    def test_missing_file_raises_value_error(self):
        with self.assertRaises(ValueError):
            SpatialModel(os.path.join(DATA, 'no_such_file.xml'))


class TestEconomicBalancer(unittest.TestCase):
    """Технико-экономическая балансировка."""

    def _cfg(self, label, cams, cam_cost, drones, cov):
        return Configuration(label=label, num_cameras=cams,
                             camera_cost=cam_cost, num_drones=drones,
                             drone_cost_per=15000, coverage_cameras=cov,
                             coverage_total=cov)

    def test_tco_is_sum_of_parts(self):
        c = self._cfg('a', 10, 800000, 3, 95.0)
        self.assertEqual(c.tco, 800000 + 3 * 15000)

    def test_cheaper_config_wins_at_equal_coverage(self):
        b = EconomicBalancer(0.7, 0.3)
        b.add(self._cfg('дорогая', 12, 960000, 4, 100.0))
        b.add(self._cfg('дешёвая', 6, 540000, 4, 100.0))
        self.assertEqual(b.best().label, 'дешёвая')

    def test_score_is_not_constant(self):
        """
        Раньше нормировка покрытия по максимуму выборки давала лучшей
        конфигурации Score = w_cov - w_cost независимо от показателей.
        """
        b = EconomicBalancer(0.7, 0.3)
        b.add(self._cfg('полная', 12, 900000, 4, 100.0))
        b.add(self._cfg('частичная', 12, 900000, 4, 80.0))
        scores = [c.score for c in b.rank()]
        self.assertNotAlmostEqual(scores[0], scores[1])

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(AssertionError):
            EconomicBalancer(0.7, 0.7)


class TestEndToEnd(unittest.TestCase):
    """Сквозные инварианты полного расчёта."""

    def test_coverage_within_bounds(self):
        model = SpatialModel(STADIUM, grid_step=5.0)
        res = evaluate(model, 12, 50, 12, max_drones=4, use_drones=True)
        self.assertGreaterEqual(res['coverage_total'], 0.0)
        self.assertLessEqual(res['coverage_total'], 100.0)

    def test_drones_never_reduce_coverage(self):
        model = SpatialModel(STADIUM, grid_step=5.0)
        without = evaluate(model, 8, 50, 12, max_drones=0, use_drones=False)
        with_dr = evaluate(model, 8, 50, 12, max_drones=4, use_drones=True)
        self.assertGreaterEqual(with_dr['coverage_total'],
                                without['coverage_total'])

    def test_partial_coverage_is_reachable(self):
        """Совокупное покрытие может быть строго меньше 100 %."""
        model = SpatialModel(STADIUM, grid_step=5.0)
        res = evaluate(model, 4, 10, 3, max_drones=1, use_drones=True)
        self.assertLess(res['coverage_total'], 100.0)

    def test_point_accounting_is_consistent(self):
        model = SpatialModel(STADIUM, grid_step=5.0)
        res = evaluate(model, 8, 50, 10, max_drones=3, use_drones=True)
        N = len(model.grid)
        covered = res['stat_cfg']['covered_points'] + res['observed_by_drones']
        self.assertEqual(covered + res['uncovered_after'], N)

    def test_fleet_sizing_finds_minimal_fleet(self):
        """Подбор флота должен возвращать минимальный достаточный состав."""
        model = SpatialModel(STADIUM, grid_step=5.0)
        res = size_fleet(model, 12, 50, 5, target_pct=95.0, fleet_cap=6)
        self.assertTrue(res['target_met'])
        kd = res['fleet_required']
        self.assertGreaterEqual(res['coverage_total'], 95.0)
        if kd > 0:
            smaller = evaluate(model, 12, 50, 5,
                               max_drones=kd - 1, use_drones=kd - 1 > 0)
            self.assertLess(smaller['coverage_total'], 95.0,
                            'меньший флот не должен достигать цели')

    def test_fleet_sizing_reports_unreachable_target(self):
        model = SpatialModel(STADIUM, grid_step=5.0)
        res = size_fleet(model, 2, 10, 2, target_pct=99.0, fleet_cap=2)
        self.assertFalse(res['target_met'])

    def test_all_venues_load_and_solve(self):
        for path in (STADIUM, HALL, COURT):
            model = SpatialModel(path, grid_step=2.5)
            res = evaluate(model, 6, 40, 8, max_drones=2, use_drones=True)
            self.assertGreater(res['coverage_total'], 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
