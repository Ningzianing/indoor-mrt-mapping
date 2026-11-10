import os
import sys

import numpy as np
import pandas as pd

import config as cfg
import m4_surface_model as m4
import room
import thermal_camera
import thermocouples
import window_model as wm


def six_faces(means, window_fraction=None):
    out = means[["floor", "ceiling", "wall2", "wall4", "wall3"]].copy()
    if window_fraction is None:
        out["wall1"] = means["wall1_all"]
    else:
        out["wall1"] = window_fraction * means["glazing"] + (1 - window_fraction) * means["wall1"]
    return out[room.FACES]


def main(day):
    cal = wm.load()
    fw = float(room.window_mask(thermal_camera.RES).mean())
    m2 = six_faces(thermocouples.surface_means(day)).add_prefix("M2_")
    m3 = six_faces(thermal_camera.surface_means(day)).add_prefix("M3_")
    m4_ = six_faces(m4.predict(day, cal), fw).add_prefix("M4_")
    out = pd.concat([m2, m3.reindex(m2.index), m4_.reindex(m2.index)], axis=1)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(cfg.OUT_DIR, f"surface_temperatures_{day}.csv"), index_label="time")
    for m in ("M3", "M4"):
        e = np.concatenate([(out[f"{m}_{f}"] - out[f"M2_{f}"]).to_numpy() for f in room.FACES])
        print(day, m, "RMSE vs M2 = %.3f K" % np.sqrt(np.nanmean(e ** 2)))


if __name__ == "__main__":
    for day in sys.argv[1:] or cfg.VALIDATION_DAYS:
        main(day)
