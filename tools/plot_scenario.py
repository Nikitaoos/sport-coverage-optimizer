"""
plot_scenario.py
----------------
Построение иллюстраций непосредственно из модели, чтобы рисунки
соответствовали числам, которые выдаёт программа.

Запуск:
    python tools/plot_scenario.py --out-dir figures
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Rectangle, Wedge, Circle      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import evaluate                                    # noqa: E402
from src import SpatialModel, StationaryOptimizer, DroneOptimizer  # noqa: E402
from src.spatial_model import points_in_sector, route_length  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'data')

BLUE, GREEN, RED, ORANGE = '#2b5fa8', '#2e9e5b', '#cf3b3b', '#e08a1e'


def _draw_sectors(ax, model):
    """Рисует зоны наблюдения и запретные зоны."""
    for s in model.sectors:
        ax.add_patch(Rectangle((s['x'], s['y']), s['width'], s['height'],
                               facecolor='#dfe7f2', edgecolor='#5a6b85',
                               lw=0.8, zorder=1))
        ax.text(s['x'] + s['width'] / 2, s['y'] + s['height'] / 2,
                s['name'], ha='center', va='center', fontsize=8,
                color='#33445e', zorder=6)
    for fz in model.forbidden_zones:
        ax.add_patch(Rectangle((fz['x'], fz['y']), fz['width'], fz['height'],
                               facecolor='#f2e3e3', edgecolor=RED,
                               lw=0.8, ls='--', zorder=1))
        ax.text(fz['x'] + fz['width'] / 2, fz['y'] + fz['height'] / 2,
                f'{fz["name"]}\n(запретная зона)', ha='center', va='center',
                fontsize=8, color='#8a3b3b', zorder=6)


def figure_grid(model, cam_idx, path):
    """
    Рисунок 1: сетка точек наблюдения и зоны обзора выбранных камер.
    Число точек и разбиение на покрытые/непокрытые берутся из модели.
    """
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    _draw_sectors(ax, model)

    covered = np.zeros(len(model.grid), dtype=bool)
    for j in cam_idx:
        cam = model.cam_positions[j]
        covered |= points_in_sector(model.grid, cam['x'], cam['y'],
                                    cam['azimuth'], cam['angle_h'] / 2.0,
                                    cam['range_m'])
        # Сектор обзора: matplotlib отсчитывает угол от +X против часовой,
        # азимут — от +Y по часовой, поэтому theta = 90 - azimuth
        theta = 90.0 - cam['azimuth']
        ax.add_patch(Wedge((cam['x'], cam['y']), cam['range_m'],
                           theta - cam['angle_h'] / 2,
                           theta + cam['angle_h'] / 2,
                           facecolor=BLUE, alpha=0.16, edgecolor=BLUE,
                           lw=0.7, zorder=2))
        ax.plot(cam['x'], cam['y'], marker='s', ms=8, color=BLUE, zorder=7)

    ax.scatter(model.grid[covered, 0], model.grid[covered, 1], s=16,
               color=GREEN, zorder=5,
               label=f'покрытые точки ({covered.sum()})')
    ax.scatter(model.grid[~covered, 0], model.grid[~covered, 1], s=16,
               color=RED, zorder=5,
               label=f'непокрытые точки ({(~covered).sum()})')
    ax.plot([], [], marker='s', ls='none', color=BLUE,
            label='позиция камеры')
    ax.fill_between([], [], color=BLUE, alpha=0.16, label='сектор обзора')

    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')
    ax.set_title(f'Сетка точек наблюдения T (δ = {model.grid_step:g} м, '
                 f'N = {len(model.grid)})', fontsize=11)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.margins(0.05)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return covered


def figure_drones(model, budget_k, max_drones, drone_range, obs_radius, path):
    """
    Рисунок 2: результат работы подсистемы БПЛА — маршруты, полосы наблюдения
    и точки, оставшиеся непокрытыми при заданном бюджете флота.
    """
    costs = [c['cost'] for c in model.cam_positions]
    opt = StationaryOptimizer(model.coverage_matrix, costs, budget_k)
    selected = opt.solve()
    unc_idx = model.get_uncovered_points(selected)
    unc_pts = model.grid[unc_idx]

    dopt = DroneOptimizer(unc_pts, max_drones=max_drones,
                          drone_range_m=drone_range, obs_radius_m=obs_radius,
                          forbidden_zones=model.forbidden_zones)
    routes = dopt.optimize()
    observed = dopt.covered_mask

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    _draw_sectors(ax, model)

    cam_covered = np.ones(len(model.grid), dtype=bool)
    cam_covered[unc_idx] = False
    ax.scatter(model.grid[cam_covered, 0], model.grid[cam_covered, 1],
               s=14, color='#9bbf9b', zorder=4,
               label=f'покрыто камерами ({cam_covered.sum()})')

    for ri, route in enumerate(routes):
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        for (x, y) in route:
            ax.add_patch(Circle((x, y), obs_radius, facecolor=ORANGE,
                                alpha=0.10, edgecolor='none', zorder=2))
        ax.plot(xs, ys, '-o', ms=4, lw=1.6, color=ORANGE, zorder=6,
                label='маршруты БПЛА' if ri == 0 else None)

    ax.scatter(unc_pts[observed, 0], unc_pts[observed, 1], s=26,
               color=GREEN, marker='^', zorder=7,
               label=f'наблюдается с маршрутов ({int(observed.sum())})')
    ax.scatter(unc_pts[~observed, 0], unc_pts[~observed, 1], s=30,
               color=RED, marker='x', zorder=7,
               label=f'осталось непокрытым ({int((~observed).sum())})')

    total = (cam_covered.sum() + observed.sum()) / len(model.grid) * 100
    lmax = max((route_length(r) for r in routes), default=0.0)
    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')
    ax.set_title(f'Подсистема БПЛА: K = {budget_k}, $K_d$ = {max_drones}, '
                 f'$R_{{max}}$ = {drone_range:g} м, $r_{{obs}}$ = {obs_radius:g} м\n'
                 f'совокупное покрытие {total:.1f} %, '
                 f'длина наибольшего маршрута {lmax:.1f} м',
                 fontsize=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.margins(0.05)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return total


def figure_tradeoff(model, drone_range, obs_radius, path):
    """
    Рисунок 3: зависимость совокупного покрытия от бюджетов K и K_d.
    """
    ks = [2, 4, 6, 8, 10, 12]
    kds = [0, 1, 2, 3]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    markers = ['o', 's', '^', 'D']
    for m, kd in zip(markers, kds):
        ys = []
        for k in ks:
            res = evaluate(model, k, drone_range, obs_radius,
                           max_drones=kd, use_drones=kd > 0)
            ys.append(res['coverage_total'])
        ax.plot(ks, ys, marker=m, lw=1.6, ms=5,
                label=f'$K_d$ = {kd}' + (' (без БПЛА)' if kd == 0 else ''))

    ax.axhline(95, color=RED, ls='--', lw=1.0)
    ax.text(ks[0], 95.6, 'целевой уровень 95 %', color=RED, fontsize=8)
    ax.set_xlabel('Бюджет стационарных камер K, шт.')
    ax.set_ylabel('Совокупное покрытие, %')
    ax.set_title('Зависимость совокупного покрытия от бюджетов '
                 'стационарной и мобильной подсистем', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='figures')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stadium = SpatialModel(os.path.join(DATA, 'stadium_example.xml'),
                           grid_step=5.0)

    figure_grid(stadium, cam_idx=[1, 4],
                path=os.path.join(args.out_dir, 'fig1_grid.png'))
    figure_drones(stadium, budget_k=12, max_drones=1, drone_range=50,
                  obs_radius=5,
                  path=os.path.join(args.out_dir, 'fig2_drones.png'))
    figure_tradeoff(stadium, drone_range=50, obs_radius=5,
                    path=os.path.join(args.out_dir, 'fig3_tradeoff.png'))
    print(f'Рисунки сохранены в {args.out_dir}/')
