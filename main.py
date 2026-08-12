"""
main.py
-------
Точка входа оптимизационной модели покрытия спортивного объекта
устройствами видеорегистрации.

Использование:
    python main.py [--input PATH] [--budget K] [--drone-range M]
                   [--obs-radius M] [--grid-step S] [--output PATH]
"""

import argparse
import sys
import os
import numpy as np

# Добавляем корень проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    parser.add_argument('--drones',      type=int,   default=None,
                        help='Бюджет флота БПЛА, шт. (по умолчанию — из XML)')
    parser.add_argument('--drone-range', type=float, default=None,
                        help='Максимальная длина маршрута БПЛА, м '
                             '(по умолчанию — из XML)')
    parser.add_argument('--obs-radius',  type=float, default=None,
                        help='Радиус зоны наблюдения БПЛА, м '
                             '(по умолчанию — из XML)')
    parser.add_argument('--target-coverage', type=float, default=None,
                        help='Целевое совокупное покрытие, %%. Если задано, '
                             'подбирается минимальный флот БПЛА, '
                             'обеспечивающий этот уровень')
    parser.add_argument('--no-drones',   action='store_true',
                        help='Отключить подсистему БПЛА')
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
FLEET_SEARCH_CAP = 6


def size_fleet(model, budget_k, drone_range, obs_radius,
               target_pct, fleet_cap):
    """
    Определяет минимальный состав флота БПЛА, обеспечивающий заданный
    уровень совокупного покрытия.

    Перебор ведётся от нуля аппаратов вверх; возвращается первая
    конфигурация, достигшая цели, либо конфигурация с максимальным
    допустимым флотом, если цель недостижима.
    """
    last = None
    for kd in range(0, fleet_cap + 1):
        res = evaluate(model, budget_k, drone_range, obs_radius,
                       max_drones=kd, use_drones=kd > 0)
        last = res
        if res['coverage_total'] >= target_pct - 1e-9:
            res['target_met'] = True
            res['fleet_required'] = kd
            return res
    last['target_met'] = False
    last['fleet_required'] = fleet_cap
    return last


def evaluate(model, budget_k, drone_range, obs_radius,
             max_drones=0, use_drones=True):
    """
    Считает одну конфигурацию: размещение камер, маршруты БПЛА и фактическое
    совокупное покрытие.

    Совокупное покрытие вычисляется как доля точек сетки, наблюдаемых
    стационарными камерами ИЛИ попадающих в полосу наблюдения хотя бы
    одного маршрута БПЛА. Точки, не попавшие ни туда, ни туда, остаются
    непокрытыми — покрытие БПЛА не постулируется.

    Returns
    -------
    dict с результатами конфигурации.
    """
    costs = [cam['cost'] for cam in model.cam_positions]
    stat_opt = StationaryOptimizer(model.coverage_matrix, costs, budget_k)
    selected = stat_opt.solve()
    cfg = stat_opt.get_config()

    N = len(model.grid)
    uncov_idx = model.get_uncovered_points(selected)
    uncov_pts = model.grid[uncov_idx] if len(uncov_idx) else np.empty((0, 2))

    drone_cfg = {'num_drones': 0, 'routes': [], 'observed_pts_cnt': 0,
                 'max_route_len_m': 0.0, 'routes_valid': True,
                 'uncovered_pts_cnt': len(uncov_idx)}
    routes = []
    observed_by_drones = 0

    if use_drones and len(uncov_pts) > 0:
        drone_opt = DroneOptimizer(
            uncov_pts,
            max_drones=max_drones,
            drone_range_m=drone_range,
            obs_radius_m=obs_radius,
            forbidden_zones=model.forbidden_zones,
        )
        routes = drone_opt.optimize()
        drone_cfg = drone_opt.get_config()
        observed_by_drones = drone_opt.covered_count()

    covered_total = cfg['covered_points'] + observed_by_drones
    coverage_total = covered_total / N * 100.0 if N else 0.0

    return {
        'selected':          selected,
        'stat_cfg':          cfg,
        'drone_cfg':         drone_cfg,
        'routes':            routes,
        'uncovered_before':  len(uncov_idx),
        'observed_by_drones': observed_by_drones,
        'uncovered_after':   N - covered_total,
        'coverage_total':    coverage_total,
    }


# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print_separator('═')
    print('  Оптимизационная модель покрытия спортивного объекта')
    print_separator('═')

    # ── Шаг 1: загрузка и геометрическая модель ────────────────────────────
    print('\n[1/4] Загрузка объекта и вычисление матрицы покрытия...')
    print(f'      Файл: {args.input}')

    try:
        model = SpatialModel(args.input, grid_step=args.grid_step)
    except ValueError as e:
        print(f'\nОШИБКА: {e}')
        sys.exit(1)

    drone_range = args.drone_range if args.drone_range is not None \
        else model.drone_range_m
    max_drones = args.drones if args.drones is not None \
        else model.drone_max_units
    obs_radius = args.obs_radius if args.obs_radius is not None \
        else model.drone_obs_radius

    print(f'      Объект: {model.venue_name} (тип: {model.venue_type})')
    print(f'      Зоны наблюдения: {len(model.sectors)}')
    print(f'      Точек сетки (N): {len(model.grid)}')
    print(f'      Кандидатных позиций (M): {len(model.cam_positions)}')

    if len(model.grid) == 0:
        print('\nОШИБКА: Сетка точек пуста. Проверьте XML-описание объекта.')
        sys.exit(1)

    # ── Шаг 2-3: основная конфигурация ────────────────────────────────────
    print(f'\n[2/4] Оптимизация стационарных камер (бюджет K={args.budget})...')

    if args.target_coverage is not None and not args.no_drones:
        res = size_fleet(model, args.budget, drone_range, obs_radius,
                         args.target_coverage, max_drones)
        max_drones = res['fleet_required']
    else:
        res = evaluate(model, args.budget, drone_range, obs_radius,
                       max_drones=max_drones, use_drones=not args.no_drones)
    cfg, drone_cfg = res['stat_cfg'], res['drone_cfg']

    print(f'      Решатель: {cfg["solver"]}')
    print(f'      Установлено камер: {cfg["num_cameras"]}')
    print(f'      Покрытие стационарными камерами: {cfg["coverage_pct"]:.1f}%  '
          f'({cfg["covered_points"]}/{len(model.grid)} точек)')
    print(f'      Стоимость камер: {int(cfg["total_cost"]):,} руб.')

    print(f'\n[3/4] Оптимизация БПЛА (бюджет флота {max_drones} шт., '
          f'маршрут ≤ {drone_range:g} м, полоса наблюдения {obs_radius:g} м)...')

    if args.no_drones:
        print('      Подсистема БПЛА отключена (--no-drones).')
    elif res['uncovered_before'] == 0:
        print('      Все зоны покрыты стационарными камерами. БПЛА не требуются.')
    else:
        print(f'      Непокрытых камерами точек: {res["uncovered_before"]}')
        print(f'      Аппаратов / маршрутов: {drone_cfg["num_drones"]}')
        print(f'      Наблюдается с маршрутов: {res["observed_by_drones"]} '
              f'из {res["uncovered_before"]}')
        print(f'      Длина самого длинного маршрута: '
              f'{drone_cfg["max_route_len_m"]:g} м (лимит {drone_range:g} м)')
        if not drone_cfg['routes_valid']:
            print('      ВНИМАНИЕ: построены недопустимые маршруты '
                  '(превышение дальности или пересечение запретной зоны).')
        if res['uncovered_after'] > 0:
            print(f'      Остаётся непокрытыми: {res["uncovered_after"]} точек')

    print(f'      Совокупное покрытие: {res["coverage_total"]:.1f}%')
    if args.target_coverage is not None and not args.no_drones:
        if res.get('target_met'):
            print(f'      Минимальный флот для {args.target_coverage:g}%: '
                  f'{res["fleet_required"]} шт.')
        else:
            print(f'      Цель {args.target_coverage:g}% не достигнута при '
                  f'флоте до {res["fleet_required"]} шт.')

    # ── Шаг 4: технико-экономический анализ ───────────────────────────────
    print('\n[4/4] Технико-экономический анализ...')

    w_cov = min(1.0, max(0.0, args.w_coverage))
    w_cost = 1.0 - w_cov
    balancer = EconomicBalancer(w_coverage=w_cov, w_cost=w_cost)

    main_config = Configuration(
        label=f'K={args.budget}, флот {drone_cfg["num_drones"]}',
        num_cameras=cfg['num_cameras'],
        camera_cost=cfg['total_cost'],
        num_drones=drone_cfg['num_drones'],
        drone_cost_per=model.drone_cost,
        coverage_cameras=cfg['coverage_pct'],
        coverage_total=res['coverage_total'],
    )
    balancer.add(main_config)

    # Альтернативы с сокращённым бюджетом камер.
    # Если задан целевой уровень покрытия, для каждой альтернативы
    # подбирается свой минимальный флот, а конфигурации, не достигшие цели,
    # в сравнение не включаются: сопоставлять по стоимости имеет смысл
    # только те варианты, которые решают задачу.
    rejected = []
    if not args.no_drones:
        for alt_k in sorted({max(1, args.budget // 2),
                             max(1, (2 * args.budget) // 3)}):
            if alt_k >= args.budget:
                continue
            if args.target_coverage is not None:
                alt = size_fleet(model, alt_k, drone_range, obs_radius,
                                 args.target_coverage, FLEET_SEARCH_CAP)
                if not alt['target_met']:
                    rejected.append((alt_k, alt['coverage_total']))
                    continue
            else:
                alt = evaluate(model, alt_k, drone_range, obs_radius,
                               max_drones=max_drones, use_drones=True)
            balancer.add(Configuration(
                label=f'K={alt_k}, флот {alt["drone_cfg"]["num_drones"]}',
                num_cameras=alt['stat_cfg']['num_cameras'],
                camera_cost=alt['stat_cfg']['total_cost'],
                num_drones=alt['drone_cfg']['num_drones'],
                drone_cost_per=model.drone_cost,
                coverage_cameras=alt['stat_cfg']['coverage_pct'],
                coverage_total=alt['coverage_total'],
            ))

    ranked = balancer.rank()
    best = ranked[0]

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
          f'(совокупное покрытие {res["coverage_total"]:.1f}%)')
    print(f'  TCO                 : {int(main_config.tco):,} руб.')

    if len(ranked) > 1:
        print()
        print('  Сравнение конфигураций:')
        print(f'  {"Конфигурация":<24} {"Камер":>5} {"БПЛА":>5} {"Покр.%":>8} '
              f'{"TCO, руб.":>12} {"Score":>8}')
        print('  ' + '─' * 68)
        for c in ranked:
            mark = ' ◀' if c is best else ''
            print(f'  {c.label:<24} {c.num_cameras:>5} {c.num_drones:>5} '
                  f'{c.coverage_total:>7.1f}% {int(c.tco):>12,} '
                  f'{c.score:>8.4f}{mark}')
        print(f'\n  Рекомендуется: {best.label}')
        for alt_k, cov in rejected:
            print(f'  Не рассматривалась: K={alt_k} — покрытие {cov:.1f}% '
                  f'ниже целевого уровня')

    print()

    # ── Генерация отчёта ───────────────────────────────────────────────────
    report = ReportGenerator(
        venue_name=model.venue_name,
        venue_type=model.venue_type,
        cam_positions=model.cam_positions,
        selected_idx=res['selected'].tolist(),
        drone_routes=res['routes'],
        coverage_cameras=cfg['coverage_pct'],
        coverage_total=res['coverage_total'],
        tco=main_config.tco,
        alternatives=balancer.summary(),
    )
    report.save(args.output)
    print(f'  Отчёт сохранён: {args.output}')
    print_separator('═')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
