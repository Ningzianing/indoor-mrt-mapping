import numpy as np
import pandas as pd

import config as cfg
import data
import longwave
import room
import shortwave as sw
import thermocouples

UNIT_GLAZING = sw.WindowParams(1.0, 4.0, cfg.GLAZING_PANES)
MIN_VERTICAL = 20.0


def measured_inputs(day):
    d = data.irradiance(day)
    site = room.site(day)
    p, _ = sw.sun_geometry(data.mid_minute(d).index, site)
    cz = p["sz"].to_numpy()
    dni = np.where(cz > 0, d["dni"].clip(lower=0).to_numpy(), 0.0)
    dhi = np.clip(d["ghi"].to_numpy() - dni * np.clip(cz, 0, None), 0, None)
    p["ghi"] = np.where(cz > 0, d["ghi"].clip(lower=0).to_numpy(), 0.0)
    p["beam_n"] = dni
    p["dhi_iso"] = dhi
    beam_v = p["beam_n"] * p["cos_inc"].clip(lower=0)
    diffuse_measured = (d["vertical"].clip(lower=0).to_numpy() - beam_v).clip(lower=0)
    diffuse_model = p["dhi_iso"] / 2 + site.ground_albedo * p["ghi"] / 2
    f = np.where(diffuse_model > 1.0, diffuse_measured / diffuse_model.clip(lower=1.0), 1.0)
    f = np.clip(f, 0, 3)
    p["dhi_iso"] = p["dhi_iso"] * f
    p["ghi"] = p["ghi"] * f
    return p, d


def measured_transmittance(d):
    e_in = ((d["window_upper"] + d["window_lower"]) / 2).clip(lower=0)
    e_v = d["vertical"].clip(lower=0)
    tau = (e_in / e_v).where(e_v >= MIN_VERTICAL)
    return tau.fillna(float(tau.median())).clip(0, 1)


def reference_irradiance(day, points, rows=None):
    p, d = measured_inputs(day)
    if cfg.SHADING[day] is None:
        tau = measured_transmittance(d).to_numpy()
        tb = np.maximum(UNIT_GLAZING.tau(p["theta"].to_numpy()), 0.05)
        p["beam_n"] = p["beam_n"] * tau / tb
        p["dhi_iso"] = p["dhi_iso"] * tau / UNIT_GLAZING.tau_diffuse
        p["ghi"] = p["ghi"] * tau / UNIT_GLAZING.tau_diffuse
        if rows is not None:
            p = p.iloc[rows]
        k = sw.indoor_irradiance(p, room.site(day), UNIT_GLAZING, room.WINDOWS, points)
    else:
        m = d["window_lower"].clip(lower=0).to_numpy()
        if rows is not None:
            m = m[rows]
        k = m[:, None] * sw.lambert_factor(room.WINDOWS, points)[None, :]
    return np.clip(k, 0, None)


def longwave_temperature(day, points, n_rays):
    V = longwave.view_matrix(points, n_rays, thermocouples.RES)
    rows = {t: longwave.radiant_temperature(V, f) for t, f in thermocouples.ten_minute_fields(day)}
    return pd.DataFrame(rows).T
