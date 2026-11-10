import numpy as np
import pandas as pd
from scipy.interpolate import griddata

import data
import room

RES = 0.02
WALL_HEIGHTS = (("0100", 0.1), ("1100", 1.1), ("2200", 2.2))
GLAZING_HEIGHTS = {"0600": 0.6, "1100": 1.1, "1700": 1.7}
SCREEN_HEIGHTS = {"0100": 0.1, "0600": 0.6, "1100": 1.1, "1700": 1.7}


def _interpolate(pts, vals, g1, g2):
    pts, vals = np.asarray(pts, float), np.asarray(vals, float)
    ok = np.isfinite(vals)
    pts, vals = pts[ok], vals[ok]
    c1 = np.clip(g1, pts[:, 0].min(), pts[:, 0].max())
    c2 = np.clip(g2, pts[:, 1].min(), pts[:, 1].max())
    lin = griddata(pts, vals, (c1, c2), method="linear")
    near = griddata(pts, vals, (c1, c2), method="nearest")
    return np.where(np.isfinite(lin), lin, near)


def surface_fields(row):
    get = lambda name: row.get(name, np.nan)
    xs, ys, zs = room.face_axes(RES)
    out = {}
    gx, gy = np.meshgrid(xs, ys)
    for face, prefix in (("floor", "床表面温度"), ("ceiling", "天井表面温度")):
        pts = [(x, y) for x in room.COLUMN_X.values() for y in room.ROW_Y.values()]
        vals = [get(f"{prefix}{c}{r}") for c in room.COLUMN_X for r in room.ROW_Y]
        out[face] = _interpolate(pts, vals, gx, gy)
    gx, gz = np.meshgrid(xs, zs)
    for face, prefix in (("wall2", "西壁表面温度"), ("wall4", "東壁表面温度")):
        pts = [(room.COLUMN_X[c], z) for c in "ABCDE" for _, z in WALL_HEIGHTS]
        vals = [get(f"{prefix}{c}{code}") for c in "ABCDE" for code, _ in WALL_HEIGHTS]
        out[face] = _interpolate(pts, vals, gx, gz)
    yv = [room.ROW_Y[r] for r in "①③⑤"]
    line = np.interp(ys, yv, [get(f"北壁表面温度{r}") for r in "①③⑤"])
    out["wall3"] = np.tile(line[None, :], (len(zs), 1))
    yv = [room.ROW_Y["①"], room.ROW_Y["⑤"]]
    grid = np.tile(np.interp(ys, yv, [get("南壁表面温度①"), get("南壁表面温度⑤")])[None, :], (len(zs), 1))
    if np.isfinite(get("ｽｸﾘｰﾝ表面温度d1100")):
        heights = list(SCREEN_HEIGHTS.values())
        vals = [np.nanmean([get(f"ｽｸﾘｰﾝ表面温度b{k}"), get(f"ｽｸﾘｰﾝ表面温度d{k}")]) for k in SCREEN_HEIGHTS]
    else:
        heights = list(GLAZING_HEIGHTS.values())
        vals = [np.nanmean([get(f"室内側ｶﾞﾗｽ表面温度b{k}"), get(f"室内側ｶﾞﾗｽ表面温度d{k}")]) for k in GLAZING_HEIGHTS]
    column = np.interp(zs, heights, vals)
    out["wall1"] = np.where(room.window_mask(RES), column[:, None], grid)
    return out


def ten_minute_fields(day):
    s = data.ten_minute(data.surface_temperatures(day))
    for t, row in s.iterrows():
        if not row.isna().all():
            yield t, surface_fields(row)


def surface_means(day):
    win = room.window_mask(RES)
    rows = {}
    for t, f in ten_minute_fields(day):
        r = {k: float(np.nanmean(f[k])) for k in ("wall2", "wall4", "wall3", "floor", "ceiling")}
        r["glazing"] = float(np.nanmean(f["wall1"][win]))
        r["wall1"] = float(np.nanmean(f["wall1"][~win]))
        r["wall1_all"] = float(np.nanmean(f["wall1"]))
        rows[t] = r
    return pd.DataFrame(rows).T
