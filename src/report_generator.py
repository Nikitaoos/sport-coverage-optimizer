"""
report_generator.py
-------------------
Генерация XML-отчёта с рекомендуемой конфигурацией системы видеонаблюдения.

Структура выходного XML:
<coverage_report>
    <venue .../>
    <summary coverage_total="..." coverage_cameras="..." tco="..." score="..."/>
    <stationary_cameras count="...">
        <camera id="..." x="..." y="..." azimuth="..." range_m="..." cost="..."/>
        ...
    </stationary_cameras>
    <drone_routes count="...">
        <route id="..." waypoints_count="...">
            <waypoint seq="..." x="..." y="..."/>
            ...
        </route>
        ...
    </drone_routes>
    <economic_alternatives>
        <configuration label="..." cameras="..." drones="..."
                       coverage_total="..." tco="..." score="..."/>
        ...
    </economic_alternatives>
</coverage_report>
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import date


class ReportGenerator:
    """
    Формирует XML-отчёт по результатам оптимизации.

    Parameters
    ----------
    venue_name      : название объекта
    venue_type      : тип объекта
    cam_positions   : list[dict] — все кандидатные позиции (из SpatialModel)
    selected_idx    : list[int]  — индексы выбранных позиций
    drone_routes    : list[list[tuple]] — маршруты БПЛА
    coverage_cameras: float — % покрытия стационарными камерами
    coverage_total  : float — итоговый % покрытия
    tco             : float — суммарная стоимость
    alternatives    : list[dict] — сводная таблица конфигураций (из EconomicBalancer)
    """

    def __init__(self,
                 venue_name: str,
                 venue_type: str,
                 cam_positions: list[dict],
                 selected_idx: list[int],
                 drone_routes: list,
                 coverage_cameras: float,
                 coverage_total: float,
                 tco: float,
                 alternatives: list[dict] = None):
        self.venue_name       = venue_name
        self.venue_type       = venue_type
        self.cam_positions    = cam_positions
        self.selected_idx     = selected_idx
        self.drone_routes     = drone_routes or []
        self.coverage_cameras = coverage_cameras
        self.coverage_total   = coverage_total
        self.tco              = tco
        self.alternatives     = alternatives or []

    # ------------------------------------------------------------------
    def build_xml(self) -> ET.Element:
        """Строит дерево XML и возвращает корневой элемент."""
        root = ET.Element('coverage_report',
                          date=str(date.today()),
                          generator='sport-coverage-optimizer')

        # Объект
        ET.SubElement(root, 'venue',
                      name=self.venue_name,
                      type=self.venue_type)

        # Сводка
        ET.SubElement(root, 'summary',
                      coverage_total=f"{self.coverage_total:.1f}",
                      coverage_cameras=f"{self.coverage_cameras:.1f}",
                      tco=str(int(self.tco)),
                      num_cameras=str(len(self.selected_idx)),
                      num_drones=str(len(self.drone_routes)))

        # Стационарные камеры
        cams_el = ET.SubElement(root, 'stationary_cameras',
                                count=str(len(self.selected_idx)))
        for idx in self.selected_idx:
            cam = self.cam_positions[idx]
            ET.SubElement(cams_el, 'camera',
                          id=cam['id'],
                          x=str(cam['x']),
                          y=str(cam['y']),
                          azimuth=str(cam['azimuth']),
                          angle_h=str(cam['angle_h']),
                          range_m=str(cam['range_m']),
                          cost=str(int(cam['cost'])))

        # Маршруты БПЛА
        drones_el = ET.SubElement(root, 'drone_routes',
                                  count=str(len(self.drone_routes)))
        for ri, route in enumerate(self.drone_routes):
            route_el = ET.SubElement(drones_el, 'route',
                                     id=str(ri + 1),
                                     waypoints_count=str(len(route)))
            for si, (wx, wy) in enumerate(route):
                ET.SubElement(route_el, 'waypoint',
                              seq=str(si + 1),
                              x=f"{wx:.1f}",
                              y=f"{wy:.1f}")

        # Альтернативные конфигурации
        if self.alternatives:
            alts_el = ET.SubElement(root, 'economic_alternatives')
            for alt in self.alternatives:
                ET.SubElement(alts_el, 'configuration',
                              label=alt.get('label', ''),
                              cameras=str(alt.get('num_cameras', 0)),
                              drones=str(alt.get('num_drones', 0)),
                              coverage_cameras=str(alt.get('coverage_cameras', 0)),
                              coverage_total=str(alt.get('coverage_total', 0)),
                              tco=str(alt.get('tco', 0)),
                              score=str(alt.get('score', 0)))

        return root

    # ------------------------------------------------------------------
    def save(self, output_path: str):
        """Сохраняет отчёт в XML-файл с pretty-print форматированием."""
        root = self.build_xml()
        raw = ET.tostring(root, encoding='unicode')
        pretty = minidom.parseString(raw).toprettyxml(indent='    ')
        # убираем лишнюю XML-декларацию от minidom (оставляем одну)
        lines = pretty.split('\n')
        if lines[0].startswith('<?xml'):
            lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return output_path
