import numpy as np
import pandas as pd

import longwave
import thermal_camera


def longwave_temperature(day, points, n_rays):
    V = longwave.view_matrix(points, n_rays, thermal_camera.RES)
    rows = {}
    for t0 in pd.date_range(pd.Timestamp(day), periods=144, freq="10min"):
        if thermal_camera.available(t0):
            rows[t0] = longwave.radiant_temperature(V, thermal_camera.surface_fields(t0))
        else:
            rows[t0] = np.full(len(points), np.nan)
    return pd.DataFrame(rows).T
