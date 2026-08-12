"""
run_scenarios.py
----------------
Прогон тест-сценариев оптимизационной модели и построение сводной таблицы
результатов.

Для каждого сценария подбирается минимальный состав флота БПЛА, при котором
совокупное покрытие достигает SUCCESS_THRESHOLD. Сценарий считается успешным,
если целевой уровень достигнут в пределах FLEET_CAP аппаратов и при этом все
маршруты допустимы: длина каждого не превышает дальности, ни одно звено не
пересекает запретную зону.

Запуск:
    python tests/run_scenarios.py            # таблица в консоль
    python tests/run_scenarios.py --csv results.csv
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import evaluate, size_fleet                        # noqa: E402
from src import SpatialModel                                 # noqa: E402
from src.spatial_model import route_is_clear, route_length   # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'data')

SUCCESS_THRESHOLD = 95.0
FLEET_CAP = 6

# (№, объект, xml, δ, K, R_max, r_obs, БПЛА?, комментарий)
SCENARIOS = [
    (1,  'Стадион',  'stadium_example.xml', 5.0, 12, 50, 5, True,  'базовый сценарий'),
    (2,  'Стадион',  'stadium_example.xml', 5.0, 10, 50, 5, True,  'сокращённый бюджет камер'),
    (3,  'Стадион',  'stadium_example.xml', 5.0,  8, 50, 5, True,  'бюджет камер уменьшен на треть'),
    (4,  'Стадион',  'stadium_example.xml', 5.0,  4, 50, 5, True,  'дефицит камер'),
    (5,  'Стадион',  'stadium_example.xml', 5.0, 12, 25, 4, True,  'короткий маршрут, узкая полоса'),
    (6,  'Стадион',  'stadium_example.xml', 5.0, 12,  0, 0, False, 'без БПЛА'),
    (7,  'Зал',      'hall_example.xml',    2.5,  6,  0, 0, False, 'без БПЛА (закрытый объект)'),
    (8,  'Зал',      'hall_example.xml',    2.5,  4,  0, 0, False, 'сокращённый бюджет'),
    (9,  'Зал',      'hall_example.xml',    2.5,  3,  0, 0, False, 'дефицит камер, без БПЛА'),
    (10, 'Зал',      'hall_example.xml',    2.5,  3, 40, 5, True,  'дефицит камер, с БПЛА'),
    (11, 'Зал',      'hall_example.xml',    2.5,  2, 40, 5, True,  'жёсткий дефицит камер'),
    (12, 'Площадка', 'court_example.xml',   2.5,  4, 45, 4, True,  'базовый сценарий'),
    (13, 'Площадка', 'court_example.xml',   2.5,  2, 45, 4, True,  'минимум камер'),
    (14, 'Площадка', 'court_example.xml',   2.5,  1, 45, 4, True,  'дефицит камер'),
    (15, 'Площадка', 'court_example.xml',   2.5,  2, 15, 3, True,  'короткий маршрут, узкая полоса'),
]


def run_all() -> list:
    """Выполняет все сценарии и возвращает список словарей с результатами."""
    rows = []
    cache = {}

    for (num, venue, xml, step, k, rng, obs, use_drones, note) in SCENARIOS:
        key = (xml, step)
        if key not in cache:
            cache[key] = SpatialModel(os.path.join(DATA, xml), grid_step=step)
        model = cache[key]

        t0 = time.perf_counter()
        if use_drones:
            res = size_fleet(model, k, rng, obs, SUCCESS_THRESHOLD, FLEET_CAP)
            fleet = res['fleet_required']
            target_met = res['target_met']
        else:
            res = evaluate(model, k, rng, obs, max_drones=0, use_drones=False)
            fleet = 0
            target_met = res['coverage_total'] >= SUCCESS_THRESHOLD
        elapsed = time.perf_counter() - t0

        routes = res['routes']
        routes_clear = all(route_is_clear(r, model.forbidden_zones)
                           for r in routes)
        routes_fit = all(route_length(r) <= rng + 1e-6 for r in routes)
        max_len = max((route_length(r) for r in routes), default=0.0)

        ok = target_met and routes_clear and routes_fit

        rows.append({
            'num':        num,
            'venue':      venue,
            'step':       step,
            'N':          len(model.grid),
            'K':          k,
            'Kd':         fleet if use_drones else None,
            'range':      rng if use_drones else None,
            'obs':        obs if use_drones else None,
            'cov_cam':    res['stat_cfg']['coverage_pct'],
            'drones':     res['drone_cfg']['num_drones'],
            'cov_total':  res['coverage_total'],
            'max_len':    max_len,
            'clear':      routes_clear,
            'fit':        routes_fit,
            'ok':         ok,
            'time_s':     elapsed,
            'note':       note,
        })
    return rows


def print_table(rows: list):
    """Печатает сводную таблицу результатов."""
    head = (f'{"№":>2}  {"Объект":<10} {"δ":>4} {"N":>4} {"K":>3} {"Kd":>3} '
            f'{"R,м":>5} {"r,м":>4} {"Камеры":>8} {"Флот":>5} '
            f'{"Итого":>8} {"Lmax":>7} {"t,с":>6}  Рез.  Примечание')
    print(head)
    print('─' * len(head))

    for r in rows:
        rng = f'{r["range"]:g}' if r['range'] is not None else '—'
        obs = f'{r["obs"]:g}' if r['obs'] is not None else '—'
        kd = f'{r["Kd"]:d}' if r['Kd'] is not None else '—'
        mark = '+' if r['ok'] else '–'
        print(f'{r["num"]:>2}  {r["venue"]:<10} {r["step"]:>4g} {r["N"]:>4} '
              f'{r["K"]:>3} {kd:>3} {rng:>5} {obs:>4} {r["cov_cam"]:>7.1f}% '
              f'{r["drones"]:>5} {r["cov_total"]:>7.1f}% '
              f'{r["max_len"]:>6.1f}м {r["time_s"]:>6.2f}   {mark}   {r["note"]}')

    total = len(rows)
    good = sum(1 for r in rows if r['ok'])
    covs = [r['cov_total'] for r in rows]
    print('─' * len(head))
    print(f'Целевой уровень {SUCCESS_THRESHOLD:g}% достигнут (флот ≤ '
          f'{FLEET_CAP}, маршруты допустимы): {good}/{total}')
    print(f'Диапазон совокупного покрытия: {min(covs):.1f}–{max(covs):.1f}%')
    print(f'Максимальное время расчёта: {max(r["time_s"] for r in rows):.2f} с')

    bad_routes = [r['num'] for r in rows if not (r['clear'] and r['fit'])]
    if bad_routes:
        print(f'ВНИМАНИЕ: недопустимые маршруты в сценариях: {bad_routes}')
    else:
        print('Все построенные маршруты допустимы '
              '(в пределах дальности, без пересечения запретных зон).')


def save_csv(rows: list, path: str):
    """Сохраняет результаты в CSV."""
    import csv
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=';')
        writer.writeheader()
        writer.writerows(rows)
    print(f'CSV сохранён: {path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None, help='Путь к CSV с результатами')
    a = ap.parse_args()

    results = run_all()
    print_table(results)
    if a.csv:
        save_csv(results, a.csv)
