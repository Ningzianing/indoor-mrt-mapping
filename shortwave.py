from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as cfg

SIGMA = 5.670374419e-08
SOLAR_CONSTANT = 1361.0


@dataclass
class Site:
    lat: float
    lon: float
    window_azimuth: float
    ground_albedo: float = 0.2


@dataclass
class WindowParams:
    tau0: float
    q: float = 2.0
    panes: int = 2

    def tau(self, theta_deg):
        return self.tau0 * tau_ratio(theta_deg, self.q, self.panes)

    @property
    def tau_diffuse(self):
        return self.tau0 * diffuse_ratio(self.q, self.panes)


@dataclass
class Window:
    x0: float
    x1: float
    z0: float
    z1: float


def _julian_terms(times):
    days = np.asarray((times.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1), dtype=float)
    jd = days + 2440587.5
    t = (jd - 2451545.0) / 36525.0
    l0 = np.radians((280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360.0)
    m = np.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    e = 0.016708634 - t * (4.2037e-05 + 1.267e-07 * t)
    omega = np.radians(125.04 - 1934.136 * t)
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = np.radians(eps0 + 0.00256 * np.cos(omega))
    y = np.tan(eps / 2.0) ** 2
    eot = 4.0 * np.degrees(y * np.sin(2 * l0) - 2 * e * np.sin(m) + 4 * e * y * np.sin(m) * np.cos(2 * l0)
                           - 0.5 * y * y * np.sin(4 * l0) - 1.25 * e * e * np.sin(2 * m))
    utc_min = (jd + 0.5) % 1.0 * 1440.0
    return t, l0, m, e, omega, eps, eot, utc_min


def solar_position(times, lat, lon):
    t, l0, m, e, omega, eps, eot, utc_min = _julian_terms(times)
    c = np.radians(np.sin(m) * (1.914602 - t * (0.004817 + 1.4e-05 * t)) + np.sin(2 * m) * (0.019993 - 0.000101 * t)
                   + np.sin(3 * m) * 0.000289)
    lam = l0 + c - np.radians(0.00569 + 0.00478 * np.sin(omega))
    decl = np.arcsin(np.sin(eps) * np.sin(lam))
    tst = (utc_min + eot + 4.0 * lon) % 1440.0
    h = np.radians(tst / 4.0 - 180.0)
    phi = np.radians(lat)
    cosz = np.clip(np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(h), -1, 1)
    zenith = np.degrees(np.arccos(cosz))
    azimuth = (np.degrees(np.arctan2(np.sin(h), np.cos(h) * np.sin(phi) - np.tan(decl) * np.cos(phi))) + 180.0) % 360.0
    r = 1.000001018 * (1 - e * e) / (1 + e * np.cos(m + c))
    return zenith, azimuth, SOLAR_CONSTANT / r ** 2


def true_solar_time_hours(times, lon):
    _, _, _, _, _, _, eot, utc_min = _julian_terms(times)
    return (utc_min + eot + 4.0 * lon) % 1440.0 / 60.0


ENGERER2_1MIN = dict(C=0.10562, B0=-4.1332, B1=8.2578, B2=0.010087, B3=0.00088801, B4=-4.9302, B5=0.44378)


def engerer2(ghi, zenith_deg, i0n, ast_hours, doy):
    ghi = np.asarray(ghi, dtype=float)
    zen = np.asarray(zenith_deg, dtype=float)
    cz = np.cos(np.radians(zen))
    up = cz > 0
    e_exth = np.where(up, i0n * cz, np.nan)
    kt = ghi / e_exth
    d = np.asarray(doy, dtype=float)
    a = 1160.0 + 75.0 * np.sin(np.radians(360.0 * (d - 275.0) / 365.0))
    k = 0.174 + 0.035 * np.sin(np.radians(360.0 * (d - 100.0) / 365.0))
    c = 0.095 + 0.04 * np.sin(np.radians(360.0 * (d - 100.0) / 365.0))
    dni_cs = np.where(up, a * np.exp(-k / np.maximum(cz, 0.001)), 0.0)
    ghi_cs = dni_cs * np.clip(cz, 0, None) + c * dni_cs
    ktc = ghi_cs / e_exth
    dktc = ktc - kt
    with np.errstate(divide="ignore", invalid="ignore"):
        kde = np.where(ghi - ghi_cs > 0.015, (ghi - ghi_cs) / ghi, 0.0)
    p = ENGERER2_1MIN
    with np.errstate(over="ignore", invalid="ignore"):
        kd = p["C"] + (1 - p["C"]) / (1 + np.exp(p["B0"] + p["B1"] * kt + p["B2"] * ast_hours + p["B3"] * zen + p["B4"] * dktc)) + p["B5"] * kde
    kd = np.clip(np.nan_to_num(kd, nan=1.0), 0.0, 1.0)
    dhi = kd * ghi
    low = zen >= 87.0
    dni = np.where(low | ~up, 0.0, (ghi - dhi) / np.maximum(cz, 0.001))
    dni = np.clip(np.minimum(dni, i0n), 0.0, None)
    dhi = np.where(low | ~up, np.clip(ghi, 0, None), dhi)
    return dni, np.clip(dhi, 0.0, None)


PEREZ_1990 = np.array([[-0.008, 0.588, -0.062, -0.06, 0.072, -0.022], [0.13, 0.683, -0.151, -0.019, 0.066, -0.029],
                       [0.33, 0.487, -0.221, 0.055, -0.064, -0.026], [0.568, 0.187, -0.295, 0.109, -0.152, -0.014],
                       [0.873, -0.392, -0.362, 0.226, -0.462, 0.001], [1.132, -1.237, -0.412, 0.288, -0.823, 0.056],
                       [1.06, -1.6, -0.359, 0.264, -1.127, 0.131], [0.678, -0.327, -0.25, 0.156, -1.377, 0.251]])


def perez_f1f2(dhi, dni, zenith_deg, i0n):
    z = np.radians(np.minimum(np.asarray(zenith_deg, dtype=float), 89.9))
    zd = np.degrees(z)
    am = 1.0 / (np.cos(z) + 0.50572 * (96.07995 - zd) ** (-1.6364))
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = ((dhi + dni) / dhi + 1.041 * z ** 3) / (1 + 1.041 * z ** 3)
    delta = dhi * am / i0n
    b = np.clip(np.digitize(np.nan_to_num(eps, nan=1.0), (1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2)), 0, 7)
    c = PEREZ_1990[b]
    f1 = np.maximum(c[:, 0] + c[:, 1] * delta + c[:, 2] * z, 0.0)
    f2 = c[:, 3] + c[:, 4] * delta + c[:, 5] * z
    ok = (dhi > 0) & np.isfinite(eps)
    return np.where(ok, f1, 0.0), np.where(ok, f2, 0.0)


def tau_ratio(theta_deg, q, panes=2):
    z = np.clip(np.asarray(theta_deg, dtype=float) / 90.0, 0.0, 1.0)
    a = 8.0
    b = 0.25 / q
    c = 1.0 - a - b
    alpha = 5.2 + 0.7 * q
    gamma = 5.26 + 0.06 * panes + (0.73 + 0.04 * panes) * q
    return np.clip(1.0 - a * z ** alpha - b * z ** 2 - c * z ** gamma, 0.0, 1.0)


def diffuse_ratio(q, panes=2):
    th = np.linspace(0.0, 90.0, 361)
    r = np.radians(th)
    f = tau_ratio(th, q, panes) * np.cos(r) * np.sin(r)
    return float(2.0 * np.sum((f[1:] + f[:-1]) * 0.5 * np.diff(r)))


def sun_geometry(index, site):
    zen, az, i0n = solar_position(index, site.lat, site.lon)
    zr, ar, aw = np.radians(zen), np.radians(az), np.radians(site.window_azimuth)
    cosz = np.cos(zr)
    s_e, s_n = np.sin(zr) * np.sin(ar), np.sin(zr) * np.cos(ar)
    sx = s_e * -np.cos(aw) + s_n * np.sin(aw)
    sy = s_e * -np.sin(aw) + s_n * -np.cos(aw)
    cos_inc = -sy
    out = pd.DataFrame(index=index)
    out["zenith"], out["azimuth"] = zen, az
    out["cos_inc"] = cos_inc
    out["theta"] = np.degrees(np.arccos(np.clip(cos_inc, -1, 1)))
    out["sx"], out["sy"], out["sz"] = sx, sy, cosz
    return out, i0n


def separate(ghi, site):
    out, i0n = sun_geometry(ghi.index, site)
    zen = out["zenith"].to_numpy()
    cosz = out["sz"].to_numpy()
    g = np.clip(ghi.to_numpy(dtype=float), 0.0, None)
    g = np.where(cosz > 0, g, 0.0)
    ast = true_solar_time_hours(ghi.index, site.lon)
    doy = ghi.index.dayofyear.to_numpy()
    dni, dhi = engerer2(g, zen, i0n, ast, doy)
    low_sun = zen > 90.0 - cfg.LOW_SUN_ALTITUDE
    dni = np.where(low_sun, 0.0, dni)
    dhi = np.where(low_sun, g, dhi)
    cz_safe = np.maximum(cosz, np.cos(np.radians(85.0)))
    f1, f2 = perez_f1f2(dhi, dni, zen, i0n)
    beam_n = np.where(cosz > 0, dni + f1 * dhi / cz_safe, 0.0)
    dhi_iso = np.clip(dhi * (1.0 - f1 + 2.0 * f2), 0.0, None)
    beam_n = np.where(low_sun, 0.0, beam_n)
    dhi_iso = np.where(low_sun, dhi, dhi_iso)
    out["ghi"], out["beam_n"], out["dhi_iso"] = g, beam_n, dhi_iso
    return out


def window_plane_irradiance(p, albedo):
    cosi = np.clip(p["cos_inc"].to_numpy(), 0.0, None)
    return p["beam_n"].to_numpy() * cosi + 0.5 * (p["dhi_iso"].to_numpy() + albedo * p["ghi"].to_numpy())


def transmitted_irradiance(p, params, albedo):
    cosi = np.clip(p["cos_inc"].to_numpy(), 0.0, None)
    direct = params.tau(p["theta"].to_numpy()) * p["beam_n"].to_numpy() * cosi
    diffuse = params.tau_diffuse * 0.5 * (p["dhi_iso"].to_numpy() + albedo * p["ghi"].to_numpy())
    return direct + diffuse


def window_patches(w, n):
    xs = w.x0 + (np.arange(n) + 0.5) * (w.x1 - w.x0) / n
    zs = w.z0 + (np.arange(n) + 0.5) * (w.z1 - w.z0) / n
    gx, gz = np.meshgrid(xs, zs)
    pts = np.column_stack([gx.ravel(), np.zeros(gx.size), gz.ravel()])
    return pts, (w.x1 - w.x0) * (w.z1 - w.z0) / (n * n)


def indoor_irradiance(p, site, params, windows, points, normal=None, n_sub=16):
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    sphere = normal is None
    if not sphere:
        nrm = np.asarray(normal, dtype=float)
        nrm = nrm / np.linalg.norm(nrm)
    s = p[["sx", "sy", "sz"]].to_numpy()
    theta = p["theta"].to_numpy()
    beam_n, dhi_iso, ghi = p["beam_n"].to_numpy(), p["dhi_iso"].to_numpy(), p["ghi"].to_numpy()
    tau_b = params.tau(theta)
    n_t, n_p = len(p), len(pts)
    sun_in = (s[:, 1] < 0) & (s[:, 2] > 0) & (beam_n > 0)
    sy = np.where(s[:, 1] < 0, s[:, 1], -1.0)[:, None]
    tt = -pts[None, :, 1] / sy
    hx = pts[None, :, 0] + tt * s[:, 0][:, None]
    hz = pts[None, :, 2] + tt * s[:, 2][:, None]
    lit = np.zeros((n_t, n_p), dtype=bool)
    for w in windows:
        lit |= (hx >= w.x0) & (hx <= w.x1) & (hz >= w.z0) & (hz <= w.z1)
    lit &= sun_in[:, None]
    cos_face = np.full(n_t, 0.25) if sphere else np.clip(s @ nrm, 0.0, None)
    direct = lit * (tau_b * beam_n * cos_face)[:, None]
    g_sky = np.zeros(n_p)
    g_gnd = np.zeros(n_p)
    for w in windows:
        wp, d_a = window_patches(w, n_sub)
        d = wp[None, :, :] - pts[:, None, :]
        r2 = np.sum(d * d, axis=2)
        dh = d / np.sqrt(r2)[:, :, None]
        cos_w = np.clip(-dh[:, :, 1], 0.0, 1.0)
        tr = tau_ratio(np.degrees(np.arccos(cos_w)), params.q, params.panes)
        cos_i = np.full(r2.shape, 0.25) if sphere else np.clip(dh @ nrm, 0.0, None)
        k = tr * cos_w * d_a / (np.pi * r2)
        up = dh[:, :, 2] > 0
        g_sky += np.sum(k * up * cos_i, axis=1)
        g_gnd += np.sum(k * ~up * cos_i, axis=1)
    diffuse = params.tau0 * (g_sky[None, :] * dhi_iso[:, None] + g_gnd[None, :] * (site.ground_albedo * ghi)[:, None])
    return direct + diffuse


def lambert_factor(windows, points, normal=None, n_sub=24):
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    out = np.zeros(len(pts))
    for k, pt in enumerate(pts):
        for w in windows:
            wp, d_a = window_patches(w, n_sub)
            d = wp - pt
            r2 = np.sum(d * d, axis=1)
            dh = d / np.sqrt(r2)[:, None]
            cos_w = np.clip(-dh[:, 1], 0, 1)
            cos_i = 0.25 if normal is None else np.clip(dh @ np.asarray(normal, dtype=float), 0, None)
            out[k] += float(np.sum(cos_w * cos_i * d_a / (np.pi * r2)))
    return out


def add_shortwave(t_lw, k):
    return ((np.asarray(t_lw) + 273.15) ** 4 + cfg.GLOBE_ABSORPTANCE * np.asarray(k, dtype=float) / (cfg.GLOBE_EMISSIVITY * SIGMA)) ** 0.25 - 273.15
