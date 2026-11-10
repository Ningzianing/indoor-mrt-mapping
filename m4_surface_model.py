import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import config as cfg
import data
import longwave
import room
import shortwave as sw
import thermal_camera
import thermocouples
import window_model as wm

OUTER_SURFACE = {"wall4": -90.0, "wall2": 90.0, "wall3": 180.0}


def inputs(day, cal):
    ghi = wm.horizontal_irradiance(day)
    d = data.ten_minute(pd.concat([data.air(day), ghi], axis=1)).interpolate(limit=6).dropna()
    g = d["ghi"].copy()
    g.index = g.index.tz_localize(cfg.TZ) + pd.Timedelta(minutes=cfg.TIME_STEP_MIN / 2)
    site, params, alb = room.site(day), wm.glazing(cal), cfg.GROUND_ALBEDO
    p = sw.separate(g, site)
    outside = sw.window_plane_irradiance(p, alb)
    inside = np.clip(sw.transmitted_irradiance(p, params, alb), 0, None)
    g_floor = float(sw.lambert_factor(room.WINDOWS, [room.FLOOR_CENTRE], normal=wm.UP)[0])
    kind = cfg.SHADING[day]
    S = {"wall1": outside}
    if kind is None:
        S["glazing"] = outside
        S["floor"] = np.clip(sw.indoor_irradiance(p, site, params, room.WINDOWS, [room.FLOOR_CENTRE], normal=wm.UP)[:, 0], 0, None)
    elif kind == "interior":
        tau_s = cal["tau_s"]["interior"]
        S["glazing"] = inside
        S["floor"] = tau_s * inside * g_floor
    else:
        tau_s = cal["tau_s"]["exterior"]
        S["glazing"] = tau_s * outside
        S["floor"] = np.where(inside > 0.5, tau_s * params.tau_diffuse * outside, 0.0) * g_floor
    for s, offset in OUTER_SURFACE.items():
        q = sw.separate(g, room.site(day, offset))
        S[s] = np.clip(sw.window_plane_irradiance(q, alb), 0, None)
    S["ceiling"] = np.clip(p["beam_n"].to_numpy() * np.clip(p["sz"].to_numpy(), 0, None) + p["dhi_iso"].to_numpy(), 0, None)
    return d, pd.DataFrame(S, index=d.index)[room.SURFACES]


def simulate(par, ta, to, s, t0):
    a, b, tau_h = par
    target = a * ta + (1 - a) * to + b * s
    if tau_h < 0.001:
        return target
    k = np.exp(-cfg.TIME_STEP_MIN * 60.0 / (tau_h * 3600.0))
    t = np.empty_like(target)
    t[0] = t0
    for i in range(1, len(t)):
        t[i] = target[i - 1] + (t[i - 1] - target[i - 1]) * k
    return t


def calibrate(cal):
    day = cfg.CALIBRATION_DAY
    d, S = inputs(day, cal)
    obs = thermocouples.surface_means(day)
    ta, to = d["Ta"].to_numpy(), d["To"].to_numpy()
    out = {}
    for s in room.SURFACES:
        t0 = float(obs[s].dropna().iloc[0])
        sim = lambda par: simulate(par, ta, to, S[s].to_numpy(), t0)
        res = lambda par: np.nan_to_num((pd.Series(sim(par), index=d.index).reindex(obs.index) - obs[s]).to_numpy())
        best = None
        for tau in (0.0, 1.0, 6.0):
            sol = least_squares(res, [0.8, 0.01, tau], bounds=([0, 0.0, 0], [1.0, 10.0, cfg.TAU_MAX_HOURS]))
            if best is None or sol.cost < best.cost:
                best = sol
        out[s] = dict(zip(("a", "b", "tau_h"), map(float, best.x)))
    return out


def predict(day, cal):
    d, S = inputs(day, cal)
    obs = thermocouples.surface_means(day)
    ta, to = d["Ta"].to_numpy(), d["To"].to_numpy()
    out = {}
    for s in room.SURFACES:
        par = [cal["m4"][s][k] for k in ("a", "b", "tau_h")]
        out[s] = simulate(par, ta, to, S[s].to_numpy(), float(obs[s].dropna().iloc[0]))
    return pd.DataFrame(out, index=d.index)


def longwave_temperature(surface_temperatures, points, n_rays):
    shares = longwave.surface_shares(points, n_rays, thermal_camera.RES)
    t4 = (surface_temperatures[room.SURFACES].to_numpy() + 273.15) ** 4
    return pd.DataFrame((t4 @ shares.T) ** 0.25 - 273.15, index=surface_temperatures.index)
