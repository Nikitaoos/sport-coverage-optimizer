"""
main.py
-------
Точка входа оптимизационной модели покрытия спортивного объекта
устройствами видеорегистрации.

Использование:
    python main.py [--input PATH] [--budget K] [--drone-range M]
                   [--grid-step S] [--output PATH]
"""

import argparse
import sys
import os
import numpy as np

# Добавляем корень проекта в путь для импорта
sys.path.insert(0, os.path.dirname(__file__))

from src import (
    SpatialModel,
    StationaryOptimizer,
    DroneOptimizer,
    EconomicBalancer,
    Configuration,
    ReportGenerator,
)


# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Оптимизационная модель покрытия спортивного объекта '
                    'устройствами видеорегистрации.'
    )
    parser.add_argument('--input',       default='data/stadium_example.xml',
                        help='Путь к XML-описанию объекта')
    parser.add_argument('--budget',      type=int,   default=12,
                        help='Максимальное число стационарных камер')
    parser.add_argument('--drone-range', type=float, default=50.0,
                        help='Дальность БПЛА, м')
    parser.add_argument('--grid-step',   type=float, default=5.0,
                        help='Шаг сетки точек наблюдения, м')
    parser.add_argument('--output',      default='output_config.xml',
                        help='Путь к выходному XML-отчёту')
    parser.add_argument('--w-coverage',  type=float, default=0.7,
                        help='Вес критерия покрытия [0..1]')
    return parser.parse_args()


# ---------------------------------------------------------------------------
def print_separator(char='─', width=55):
    print(char * width)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print_separator('═')
    print('  Оптимизационная модель покрытия спортивного объекта')
    print_separator('═')

    # ── Шаг 1: загрузка и геометрическая модель ────────────────────────────
    print(f'\n[1/4] Загрузка объекта и вычисление матрицы покрытия...')
    print(f'      Файл: {args.input}')

    model = SpatialModel(args.input, grid_step=args.grid_step)

    print(f'      Объект: {model.venue_name} (тип: {model.venue_type})')
    print(f'      Зоны наблюдения: {len(model.sectors)}')
    print(f'      Точек сетки (N): {len(model.grid)}')
    print(f'      Кандидатных позиций (M): {len(model.cam_positions)}')

    if len(model.grid) == 0:
        print('\nОШИБКА: Сетка точек пуста. Проверьте XML-описание объекта.')
        sys.exit(1)

    # ── Шаг 2: оптимизация стационарных камер ─────────────────────────────
    print(f'\n[2/4] Оптимизация стационарных камер (бюджет K={args.budget})...')

    costs = [cam['cost'] for cam in model.cam_positions]
    stat_opt = StationaryOptimizer(model.coverage_matrix, costs, args.budget)
    selected = stat_opt.solve()
    cfg = stat_opt.get_config()

    print(f'      Решатель: {cfg["solver"]}')
    print(f'      Установлено камер: {cfg["num_cameras"]}')
    print(f'      Покрытие стационарными камерами: {cfg["coverage_pct"]:.1f}%  '
          f'({int(cfg["coverage_pct"]*len(model.grid)/100)}/{len(model.grid)} точек)')
    print(f'      Стоимость камер: {int(cfg["total_cost"]):,} руб.')

    # ── Шаг 3: оптимизация БПЛА ───────────────────────────────────────────
    print(f'\n[3/4] Оптимизация БПЛА (дальность {args.drone_range} м)...')

    uncov_idx = model.get_uncovered_points(selected)
    uncov_pts = model.grid[uncov_idx] if len(uncov_idx) > 0 else np.empty((0, 2))

    drone_opt = DroneOptimizer(
        uncov_pts,
        drone_range_m=args.drone_range,
        grid_step=args.grid_step,
        forbidden_zones=model.forbidden_zones,
    )
    routes = drone_opt.optimize()
    drone_cfg = drone_opt.get_config()

    if drone_cfg['num_drones'] == 0:
        print('      Все зоны покрыты стационарными камерами. БПЛА не требуются.')
        coverage_total = cfg['coverage_pct']
    else:
        # Приближённая оценка дополнительного покрытия БПЛА
        # (считаем, что каждый маршрут покрывает точки кластера)
        drones_cover_pts = len(uncov_idx)
        coverage_total = min(100.0,
            cfg['coverage_pct'] + drones_cover_pts / len(model.grid) * 100.0)
        print(f'      Непокрытых точек: {len(uncov_idx)}')
        print(f'      Кластеров / маршрутов БПЛА: {drone_cfg["num_drones"]}')
        print(f'      Совокупное покрытие: {coverage_total:.1f}%')

    # ── Шаг 4: технико-экономический анализ ───────────────────────────────
    print(f'\n[4/4] Технико-экономический анализ...')

    w_cov = min(1.0, max(0.0, args.w_coverage))
    w_cost = 1.0 - w_cov

    balancer = EconomicBalancer(w_coverage=w_cov, w_cost=w_cost)

    # Основная конфигурация
    main_config = Configuration(
        label='Основная (оптимальная)',
        num_cameras=cfg['num_cameras'],
        camera_cost=cfg['total_cost'],
        num_drones=drone_cfg['num_drones'],
        drone_cost_per=model.drone_cost,
        coverage_cameras=cfg['coverage_pct'],
        coverage_total=coverage_total,
    )
    balancer.add(main_config)

    # Альтернатива 1: меньше камер (бюджет K//2), без ограничения БПЛА
    if args.budget >= 4:
        alt_k = max(2, args.budget // 2)
        alt_opt = StationaryOptimizer(model.coverage_matrix, costs, alt_k)
        alt_sel = alt_opt.solve()
        alt_cfg = alt_opt.get_config()
        alt_uncov = model.get_uncovered_points(alt_sel)
        alt_drone_opt = DroneOptimizer(
            model.grid[alt_uncov] if len(alt_uncov) else np.empty((0,2)),
            drone_range_m=args.drone_range,
            grid_step=args.grid_step,
            forbidden_zones=model.forbidden_zones,
        )
        alt_drone_opt.optimize()
        alt_drone_n = alt_drone_opt.num_drones
        alt_total_cov = min(100.0,
            alt_cfg['coverage_pct'] + len(alt_uncov) / len(model.grid) * 100.0)

        balancer.add(Configuration(
            label=f'Экономичная (K={alt_k} камер)',
            num_cameras=alt_cfg['num_cameras'],
            camera_cost=alt_cfg['total_cost'],
            num_drones=alt_drone_n,
            drone_cost_per=model.drone_cost,
            coverage_cameras=alt_cfg['coverage_pct'],
            coverage_total=alt_total_cov,
        ))

    ranked = balancer.rank()
    best   = ranked[0]

    print(f'      TCO основной конфигурации: {int(main_config.tco):,} руб.')
    print(f'      Score: {main_config.score:.4f}  '
          f'(веса: покрытие={w_cov:.1f}, стоимость={w_cost:.1f})')

    # ── Итог ──────────────────────────────────────────────────────────────
    print_separator()
    print('  ИТОГ')
    print_separator()
    print(f'  Стационарные камеры : {cfg["num_cameras"]:2d}  '
          f'(покрытие {cfg["coverage_pct"]:.1f}%)')
    print(f'  БПЛА                : {drone_cfg["num_drones"]:2d}  '
          f'(совокупное покрытие {coverage_total:.1f}%)')
    print(f'  TCO                 : {int(main_config.tco):,} руб.')

    if len(ranked) > 1:
        print()
        print('  Сравнение конфигураций:')
        print(f'  {"Конфигурация":<30} {"Камер":>5} {"БПЛА":>4} {"Покр.%":>7} '
              f'{"TCO, руб.":>12} {"Score":>7}')
        print('  ' + '─' * 70)
        for c in ranked:
            mark = ' ◀ рекомендовано' if c is best else ''
            print(f'  {c.label:<30} {c.num_cameras:>5} {c.num_drones:>4} '
                  f'{c.coverage_total:>6.1f}% {int(c.tco):>12,} {c.score:>7.4f}{mark}')

    print()

    # ── Генерация отчёта ───────────────────────────────────────────────────
    report = ReportGenerator(
        venue_name=model.venue_name,
        venue_type=model.venue_type,
        cam_positions=model.cam_positions,
        selected_idx=selected.tolist(),
        drone_routes=routes,
        coverage_cameras=cfg['coverage_pct'],
        coverage_total=coverage_total,
        tco=main_config.tco,
        alternatives=balancer.summary(),
    )
    report.save(args.output)
    print(f'  Отчёт сохранён: {args.output}')
    print_separator('═')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
