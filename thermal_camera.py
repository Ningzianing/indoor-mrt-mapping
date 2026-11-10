import os

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, map_coordinates

import config as cfg
import room

RES = 0.05
FOCAL = 99.2
CAMERAS = {"03": dict(pos=(3.87, 0.1, 1.5), ypr=(135.8, -11.0, -3.3)),
           "00": dict(pos=(0.1, 3.87, 0.8), ypr=(-43.3, 10.4, 0.1))}
FACE_CAMERA = {"floor": "03", "ceiling": "03", "wall1": "03", "wall4": "03", "wall2": "00", "wall3": "00"}
MAX_OFF_AXIS = 48.0
MAX_INCIDENCE = 88.0
GLOBE_MASKS = {"wall2": [(1.05, 0.87, 0.33), (0.84, 2.21, 0.28)]}
MASK_RING = 0.1


def _world(x, y, z):
    return np.stack([np.asarray(y, float), np.asarray(x, float), np.asarray(z, float)], axis=-1)


def _axes(yaw, pitch, roll):
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    f = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    r = np.cross(f, [0, 0, 1])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c, s = np.cos(roll), np.sin(roll)
    return f, c * r + s * u, -s * r + c * u


def project(P, cam):
    pos = _world(*cam["pos"])
    f, r, u = _axes(*cam["ypr"])
    d = P - pos
    d = d / np.linalg.norm(d, axis=-1)[..., None]
    th = np.arccos(np.clip(d @ f, -1, 1))
    ph = np.arctan2(d @ u, d @ r)
    rr = FOCAL * th
    return 80 + rr * np.cos(ph) - 0.5, 60 - rr * np.sin(ph) - 0.5, np.degrees(th), d


def faces():
    xs, ys, zs = room.face_axes(RES)
    F = {}
    gx, gy = np.meshgrid(xs, ys)
    F["floor"] = (_world(gx, gy, 0 * gx), np.array([0, 0, 1.0]), xs, ys)
    F["ceiling"] = (_world(gx, gy, room.HEIGHT + 0 * gx), np.array([0, 0, -1.0]), xs, ys)
    gx, gz = np.meshgrid(xs, zs)
    F["wall2"] = (_world(gx, 0 * gx, gz), np.array([1.0, 0, 0]), xs, zs)
    F["wall4"] = (_world(gx, room.WIDTH + 0 * gx, gz), np.array([-1.0, 0, 0]), xs, zs)
    gy, gz = np.meshgrid(ys, zs)
    F["wall1"] = (_world(0 * gy, gy, gz), np.array([0, 1.0, 0]), ys, zs)
    F["wall3"] = (_world(room.DEPTH + 0 * gy, gy, gz), np.array([0, -1.0, 0]), ys, zs)
    return F


def image_path(camera, t0):
    t1 = t0 + pd.Timedelta(minutes=10)
    tag = t0.strftime("%y%m%d")
    return os.path.join(cfg.THERMAL_DIR, camera, tag, f"{t0:%y%m%d_%H%M}_{t1:%y%m%d_%H%M}_avg10min_temp_c_matrix.csv")


def load_image(camera, t0):
    p = image_path(camera, t0)
    return np.loadtxt(p, delimiter=",") if os.path.exists(p) else None


def available(t0):
    return os.path.exists(image_path("03", t0))


def camera_map(img, cam, P, n):
    if img is None:
        return np.full(P.shape[:-1], np.nan)
    u, v, th, d = project(P, cam)
    cosi = -(d @ n)
    ok = (th < MAX_OFF_AXIS) & (cosi > np.cos(np.radians(MAX_INCIDENCE))) & (u >= 0) & (u <= 159) & (v >= 0) & (v <= 119)
    val = map_coordinates(img, [v, u], order=1, mode="nearest")
    return np.where(ok, val, np.nan)


def remove_globes(face, a1, a2, g):
    if face not in GLOBE_MASKS or not np.isfinite(g).any():
        return g
    g = g.copy()
    A1, A2 = np.meshgrid(a1, a2)
    for c1, c2, r in GLOBE_MASKS[face]:
        d = np.hypot(A1 - c1, A2 - c2)
        inside, ring = d <= r, (d > r) & (d <= r + MASK_RING)
        if np.isfinite(g[ring]).any():
            g[inside] = np.nanmean(g[ring])
    return g


def surface_fields(t0):
    imgs = {c: load_image(c, t0) for c in CAMERAS}
    out = {}
    for face, (P, n, a1, a2) in faces().items():
        c = FACE_CAMERA[face]
        g = camera_map(imgs[c], CAMERAS[c], P, n)
        if np.isnan(g).any() and np.isfinite(g).any():
            idx = distance_transform_edt(np.isnan(g), return_distances=False, return_indices=True)
            g = g[tuple(idx)]
        out[face] = remove_globes(face, a1, a2, g)
    return out


def surface_means(day):
    win = room.window_mask(RES, lower_edge=False)
    rows = {}
    for t0 in pd.date_range(pd.Timestamp(day), periods=144, freq="10min"):
        f = surface_fields(t0)
        r = {k: np.nanmean(f[k]) for k in ("wall2", "wall4", "wall3", "floor", "ceiling")}
        r["glazing"], r["wall1"] = np.nanmean(f["wall1"][win]), np.nanmean(f["wall1"][~win])
        r["wall1_all"] = np.nanmean(f["wall1"])
        rows[t0] = r
    return pd.DataFrame(rows).T
