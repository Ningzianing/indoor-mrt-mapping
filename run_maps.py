import os
import sys

import numpy as np
import pandas as pd

import config as cfg
import m2_reference as m2
import m3_thermal_camera as m3
import m4_surface_model as m4
import room
import shortwave as sw
import window_model as wm

CHUNK = 60


def ten_minute_irradiance(fn, day, points_w):
    daylight = np.where(wm.horizontal_irradiance(day).to_numpy() > 0)[0]
    k = np.zeros((1440, len(points_w)))
    for c in range(0, len(daylight), CHUNK):
        rows = daylight[c:c + CHUNK]
        k[rows] = fn(rows)
    return k.reshape(144, 10, len(points_w)).mean(axis=1)


def main(day):
    cal = wm.load()
    xs, ys, points = room.plane_grid()
    points_w = [room.window_frame(p) for p in points]
    shape = (144, len(xs), len(ys))
    k_ref = ten_minute_irradiance(lambda r: m2.reference_irradiance(day, points_w, r), day, points_w)
    k_mod = ten_minute_irradiance(lambda r: wm.predicted_irradiance(day, cal, points_w, r), day, points_w)
    index = pd.date_range(pd.Timestamp(day), periods=144, freq="10min")
    lw_ref = m2.longwave_temperature(day, points, cfg.RAYS_GRID).reindex(index).to_numpy()
    lw_cam = m3.longwave_temperature(day, points, cfg.RAYS_GRID).reindex(index).to_numpy()
    lw_est = m4.longwave_temperature(m4.predict(day, cal), points, cfg.RAYS_GRID).reindex(index).to_numpy()
    out = {"xs": xs, "ys": ys,
           "M2_1": sw.add_shortwave(lw_ref, k_ref).reshape(shape), "M2_2": lw_ref.reshape(shape),
           "M3": sw.add_shortwave(lw_cam, k_mod).reshape(shape), "M4": sw.add_shortwave(lw_est, k_mod).reshape(shape)}
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(cfg.OUT_DIR, f"mrt_maps_{day}.npz"), **out)
    for m in ("M3", "M4"):
        e = out[m] - out["M2_1"]
        print(day, m, "RMSE vs M2_1 = %.3f K, mean difference = %.3f K" % (np.sqrt(np.nanmean(e ** 2)), np.nanmean(e)))


if __name__ == "__main__":
    for day in sys.argv[1:] or cfg.VALIDATION_DAYS:
        main(day)
