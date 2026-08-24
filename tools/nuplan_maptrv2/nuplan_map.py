"""nuPlan GPKG 地图适配器。

将 nuPlan HD Map（EPSG:4326 经纬度）转换为 MapTRV2 所需的局部 ego 坐标矢量真值：
- divider:       被 >=2 条车道引用的内部共享车道边界
- boundary:      可行驶区域（road_segments 并集）外边界
- ped_crossing:  crosswalk 多边形的两条长边
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.strtree import STRtree
from shapely import wkb


def _read_geom(blob: bytes):
    blob = bytes(blob)
    off = 40 if (blob[:4] == b"GPKG" or blob[:2] == b"GP") else 0
    return wkb.loads(blob[off:])


def _reproj_coords(coords, tf, has_z: bool):
    arr = np.asarray(coords, dtype=np.float64)
    x, y = tf.transform(arr[:, 0], arr[:, 1])
    if has_z:
        z = arr[:, 2]
        return list(map(tuple, np.stack([x, y, z], axis=-1)))
    return list(map(tuple, np.stack([x, y], axis=-1)))


def _reproj_geom(geom, tf):
    t = geom.geom_type
    has_z = geom.has_z
    if t == "Polygon":
        ext = _reproj_coords(geom.exterior.coords, tf, has_z)
        ints = [_reproj_coords(r.coords, tf, has_z) for r in geom.interiors]
        return Polygon(ext, ints)
    if t == "MultiPolygon":
        return MultiPolygon([_reproj_geom(p, tf) for p in geom.geoms])
    if t == "LineString":
        return LineString(_reproj_coords(geom.coords, tf, has_z))
    if t == "LinearRing":
        return LineString(_reproj_coords(geom.coords, tf, has_z))
    if t == "MultiLineString":
        return MultiLineString([_reproj_geom(l, tf) for l in geom.geoms])
    if t == "Point":
        x, y = tf.transform(geom.x, geom.y)
        return type(geom)(x, y)
    return geom


def _split_lines(geom) -> List[LineString]:
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return [l for l in geom.geoms if not l.is_empty]
    if geom.geom_type == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(_split_lines(g))
        return out
    return []


def _open_closed_ring(line: LineString, tol: float = 1e-3) -> List[LineString]:
    """把闭合线（首尾同点）从最远两点处切开成两条开放线。

    当可行驶区域（road_segments 并集）完整落在 patch 内时，
    polygon.exterior 与 patch 求交会返回整个闭合环（首尾同点，span≈0）。
    nuScenes 的 boundary 是开放线段，闭合环会导致：
      - GT 线首尾重叠（span=0），采样退化成绕圈线
      - 视觉上横跨道路，看起来像 divider
    这里检测闭合并切开：找到环上相距最远的一对顶点，把环切成两条开放的 LineString。
    """
    coords = np.asarray(line.coords)
    if coords.shape[0] < 4:
        return [line]
    # 首尾是否重合（闭合）
    if not np.allclose(coords[0], coords[-1], atol=tol):
        return [line]
    # 找相距最远的一对顶点（用首点开始的累积弧长近似避免 O(N^2)）
    n = coords.shape[0] - 1  # 去掉重复的收尾点
    ring = coords[:n]
    # 两两找最远点（n 通常几十，O(N^2) 可接受）
    best = (0, 0, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(ring[i] - ring[j]))
            if d > best[2]:
                best = (i, j, d)
    i, j, _ = best
    if j < i:
        i, j = j, i
    # 切两条开放线：A: i->j；B: j->...->i（绕过环）
    seg_a = ring[i:j + 1]
    seg_b = np.concatenate([ring[j:], ring[:i + 1]], axis=0)
    out = []
    for s in (seg_a, seg_b):
        if len(s) >= 2:
            out.append(LineString(s))
    return out


def _merge_overlap(lines: List[LineString], buffer_m: float = 1.0,
                   iou_thresh: float = 0.9) -> List[LineString]:
    if len(lines) < 2:
        return lines
    buf = [l.buffer(buffer_m, cap_style=2, join_style=2) for l in lines]
    tree = STRtree(buf)
    idx_by_id = {id(b): i for i, b in enumerate(buf)}
    final_idx: List[int] = []
    removed = set()
    for i in range(len(lines)):
        if i in removed:
            continue
        final_idx.append(i)
        for o in tree.query(buf[i]):
            j = idx_by_id.get(id(o))
            if j is None or j == i or j in removed:
                continue
            inter = o.intersection(buf[i]).area
            union = o.union(buf[i]).area
            if union > 0 and inter / union >= iou_thresh:
                removed.add(j)
    return [lines[i] for i in final_idx]


def _crosswalk_long_edges(poly: Polygon) -> List[LineString]:
    if poly.is_empty:
        return []
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:4]
    if len(coords) < 4:
        return [LineString(list(poly.exterior.coords))]
    p0, p1, p2, p3 = [np.array(c) for c in coords]
    d01 = np.linalg.norm(p1 - p0)
    d12 = np.linalg.norm(p2 - p1)
    if d01 >= d12:
        return [LineString([p0, p1]), LineString([p2, p3])]
    return [LineString([p1, p2]), LineString([p3, p0])]


class NuPlanMap:
    """单个 location 的 nuPlan 矢量地图（已投影到 target_epsg 全局坐标）。"""

    def __init__(self, gpkg_path: str, target_epsg: int):
        self.gpkg_path = gpkg_path
        self.target_epsg = target_epsg
        self._layers, self._lane_attrs = self._load_and_project(target_epsg)
        self._build_index()

    def _load_and_project(self, epsg: int):
        from pyproj import Transformer

        tf = Transformer.from_crs(4326, epsg, always_xy=True)
        layers: Dict[str, list] = {}
        lane_attrs: Dict[int, dict] = {}
        con = sqlite3.connect(f"file:{self.gpkg_path}?mode=ro", uri=True)
        for table in (
            "lanes_polygons",
            "lane_connectors",
            "boundaries",
            "crosswalks",
            "road_segments",
            "walkways",
            "carpark_areas",
            "generic_drivable_areas",
            "intersections",
            "stop_polygons",
        ):
            try:
                cur = con.execute(f"SELECT * FROM {table}")
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
            except sqlite3.Error:
                continue
            geoms = []
            for r in rows:
                rec = dict(zip(cols, r))
                blob = rec.get("geom")
                if blob is None:
                    continue
                g = _read_geom(blob)
                if g.is_empty:
                    continue
                g = _reproj_geom(g, tf)
                geoms.append((rec.get("fid"), g))
                if table == "lanes_polygons":
                    lane_attrs[rec.get("fid")] = rec
            layers[table] = geoms
        con.close()
        return layers, lane_attrs

    def _build_index(self):
        self._trees: Dict[str, Optional[STRtree]] = {}
        for key in ("lanes_polygons", "road_segments", "crosswalks"):
            geoms = [g for _, g in self._layers.get(key, [])]
            self._trees[key] = STRtree(geoms) if geoms else None

    def get_local_map(self, ego_xy: Tuple[float, float], patch_xy: Tuple[float, float],
                      angle_deg: float, source: Optional[Dict] = None) -> Dict[str, list]:
        """返回局部 ego 坐标的矢量真值。

        ego_xy: ego 在全局（target_epsg）坐标的 (x, y)。
        patch_xy: (h, w)。
        angle_deg: ego 朝向（度）。
        """
        half_h, half_w = patch_xy[0] / 2, patch_xy[1] / 2
        global_patch = box(ego_xy[0] - half_w - 0.5, ego_xy[1] - half_h - 0.5,
                           ego_xy[0] + half_w + 0.5, ego_xy[1] + half_h + 0.5)

        dividers = self._extract_dividers(global_patch)
        boundaries = self._extract_boundary(global_patch)
        ped = self._extract_ped_crossing(global_patch)

        def to_local(lines):
            out = []
            for ln in lines:
                ln = affinity.rotate(ln, -angle_deg, origin=(ego_xy[0], ego_xy[1]))
                ln = affinity.translate(ln, -ego_xy[0], -ego_xy[1])
                out.append(np.asarray(ln.coords))
            return out

        result = {
            "divider": to_local(dividers),
            "ped_crossing": to_local(ped),
            "boundary": to_local(boundaries),
        }
        if source is not None:
            result["_source"] = source
        return result

    # ---------- divider ----------
    def _extract_dividers(self, global_patch: Polygon) -> List[LineString]:
        ref_count = {}
        for rec in self._lane_attrs.values():
            lb = rec.get("left_boundary_fid")
            rb = rec.get("right_boundary_fid")
            if lb:
                ref_count[lb] = ref_count.get(lb, 0) + 1
            if rb:
                ref_count[rb] = ref_count.get(rb, 0) + 1
        boundary_geom = {fid: g for fid, g in self._layers.get("boundaries", [])}
        lines = []
        for bfid, cnt in ref_count.items():
            if cnt < 2:
                continue
            g = boundary_geom.get(bfid)
            if g is None:
                continue
            seg = g.intersection(global_patch)
            lines.extend(_split_lines(seg))
        return _merge_overlap(lines)

    # ---------- boundary ----------
    def _extract_boundary(self, global_patch: Polygon) -> List[LineString]:
        polys = [g for _, g in self._layers.get("road_segments", []) if g.intersects(global_patch)]
        if not polys:
            return []
        union = ops.unary_union(polys)
        if union.geom_type == "Polygon":
            union = MultiPolygon([union])
        lines = []
        for poly in union.geoms:
            ext = poly.exterior
            if ext.is_ccw:
                ext = LineString(list(ext.coords)[::-1])
            seg = ext.intersection(global_patch)
            for part in _split_lines(seg):
                lines.extend(_open_closed_ring(part))
            for interior in poly.interiors:
                if not interior.is_ccw:
                    interior = LineString(list(interior.coords)[::-1])
                seg2 = interior.intersection(global_patch)
                for part in _split_lines(seg2):
                    lines.extend(_open_closed_ring(part))
        return _merge_overlap(lines)

    # ---------- ped_crossing ----------
    def _extract_ped_crossing(self, global_patch: Polygon) -> List[LineString]:
        polys = [g for _, g in self._layers.get("crosswalks", []) if g.intersects(global_patch)]
        out = []
        for p in polys:
            seg = p.intersection(global_patch)
            if seg.is_empty:
                continue
            if seg.geom_type == "MultiPolygon":
                for sp in seg.geoms:
                    out.extend(_crosswalk_long_edges(sp))
            elif seg.geom_type == "Polygon":
                out.extend(_crosswalk_long_edges(seg))
        return out
