import json
import os

import numpy as np
import pandas as pd

import config as cfg
import data
import room
import shortwave as sw

UP = (0, 0, 1)


def horizontal_irradiance(day):
    return data.irradiance(day)["ghi"].clip(lower=0)


def separated(day):
    ghi = horizontal_irradiance(day)
    return sw.separate(data.mid_minute(ghi.to_frame())["ghi"], room.site(day))


def fit_tau0(day, patch_model=5.0, patch_measured=50.0):
    site = room.site(day)
    p = separated(day)
    y = data.irradiance(day)["floor_B"].to_numpy()
    ok = ((p["zenith"] < 80) & (p["ghi"] > 20) & (p["cos_inc"] > 0)).to_numpy() & np.isfinite(y)
    p, y = p[ok], y[ok]
    unit = sw.WindowParams(1.0, cfg.GLAZING_Q, cfg.GLAZING_PANES)
    total = sw.indoor_irradiance(p, site, unit, room.WINDOWS, [room.FLOOR_PYRANOMETER_B], normal=UP)[:, 0]
    p_diffuse = p.copy()
    p_diffuse["beam_n"] = 0.0
    diffuse = sw.indoor_irradiance(p_diffuse, site, unit, room.WINDOWS, [room.FLOOR_PYRANOMETER_B], normal=UP)[:, 0]
    use = ~((total - diffuse > patch_model) | (y > patch_measured))
    return float(np.clip(total[use] @ y[use] / (total[use] @ total[use]), 0.02, 0.98))


def window_exitance(kind, tau_s, p, params):
    if kind == "interior":
        return tau_s * np.clip(sw.transmitted_irradiance(p, params, cfg.GROUND_ALBEDO), 0, None)
    return tau_s * params.tau_diffuse * np.clip(sw.window_plane_irradiance(p, cfg.GROUND_ALBEDO), 0, None)


def fit_tau_s(kind, day, params):
    p = separated(day)
    source = window_exitance(kind, 1.0, p, params)
    model = source * sw.lambert_factor(room.WINDOWS, [room.FLOOR_PYRANOMETER_B], normal=UP)[0]
    y = data.irradiance(day)["floor_B"].to_numpy()
    ok = (horizontal_irradiance(day).to_numpy() > 20) & np.isfinite(y) & np.isfinite(model)
    return max(float(np.sum(model[ok] * y[ok]) / np.sum(model[ok] ** 2)), 0.0)


def calibrate():
    tau0 = fit_tau0(cfg.CALIBRATION_DAY)
    params = sw.WindowParams(tau0, cfg.GLAZING_Q, cfg.GLAZING_PANES)
    return {"tau0": tau0, "tau_d": params.tau_diffuse,
            "tau_s": {k: fit_tau_s(k, d, params) for k, d in cfg.SCREEN_CALIBRATION_DAY.items()}}


def load():
    with open(os.path.join(cfg.OUT_DIR, "calibration.json"), encoding="utf-8") as f:
        return json.load(f)


def glazing(cal):
    return sw.WindowParams(cal["tau0"], cfg.GLAZING_Q, cfg.GLAZING_PANES)


def predicted_irradiance(day, cal, points, rows=None):
    p = separated(day)
    if rows is not None:
        p = p.iloc[rows]
    params = glazing(cal)
    kind = cfg.SHADING[day]
    if kind is None:
        k = sw.indoor_irradiance(p, room.site(day), params, room.WINDOWS, points)
    else:
        m = window_exitance(kind, cal["tau_s"][kind], p, params)
        k = m[:, None] * sw.lambert_factor(room.WINDOWS, points)[None, :]
    return np.clip(k, 0, None)
