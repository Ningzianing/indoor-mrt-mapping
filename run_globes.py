import os
import sys

import numpy as np
import pandas as pd

import config as cfg
import data
import m1_globe
import m2_reference as m2
import m3_thermal_camera as m3
import m4_surface_model as m4
import room
import shortwave as sw
import window_model as wm


def ten_minute(k, day):
    index = pd.date_range(pd.Timestamp(day), periods=len(k), freq="min")
    return data.ten_minute(pd.DataFrame(k, index=index))


def main(day):
    cal = wm.load()
    names = list(room.GLOBES)
    points = list(room.GLOBES.values())
    points_w = [room.window_frame(p) for p in points]
    k_ref = ten_minute(m2.reference_irradiance(day, points_w), day)
    k_mod = ten_minute(wm.predicted_irradiance(day, cal, points_w), day)
    lw_ref = m2.longwave_temperature(day, points, cfg.RAYS_GLOBE)
    lw_cam = m3.longwave_temperature(day, points, cfg.RAYS_GLOBE)
    lw_est = m4.longwave_temperature(m4.predict(day, cal), points, cfg.RAYS_GLOBE)
    m1 = m1_globe.compute(day)
    ta = data.ten_minute(data.air(day)["Ta"])
    rows = []
    for j, g in enumerate(names):
        t = lw_ref.index
        rows.append(pd.DataFrame({
            "time": t, "globe": g, "Ta": ta.reindex(t).to_numpy(),
            "M1": m1[g].reindex(t).to_numpy(),
            "M2_1": sw.add_shortwave(lw_ref[j], k_ref[j].reindex(t)),
            "M2_2": lw_ref[j].to_numpy(),
            "M3": sw.add_shortwave(lw_cam[j].reindex(t), k_mod[j].reindex(t)),
            "M4": sw.add_shortwave(lw_est[j].reindex(t), k_mod[j].reindex(t)),
            "K_reference": k_ref[j].reindex(t).to_numpy(), "K_model": k_mod[j].reindex(t).to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(cfg.OUT_DIR, f"mrt_globes_{day}.csv"), index=False)
    for m in ("M1", "M3", "M4"):
        print(day, m, "RMSE vs M2_1 = %.3f K" % np.sqrt(np.nanmean((out[m] - out["M2_1"]) ** 2)))


if __name__ == "__main__":
    for day in sys.argv[1:] or cfg.VALIDATION_DAYS:
        main(day)
