import numpy as np
from scipy.sparse import csr_matrix

import room


def sphere_directions(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def cast_rays(p, dirs, res):
    D, W, H = room.DEPTH, room.WIDTH, room.HEIGHT
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dirs[:, 0] > 0, (D - p[0]) / dirs[:, 0], -p[0] / dirs[:, 0])
        ty = np.where(dirs[:, 1] > 0, (W - p[1]) / dirs[:, 1], -p[1] / dirs[:, 1])
        tz = np.where(dirs[:, 2] > 0, (H - p[2]) / dirs[:, 2], -p[2] / dirs[:, 2])
    t = np.stack([tx, ty, tz], 1)
    t[~np.isfinite(t) | (t <= 0)] = np.inf
    axis = t.argmin(1)
    hit = p + dirs * t.min(1)[:, None]
    xs, ys, zs = room.face_axes(res)
    nx, ny, nz = len(xs), len(ys), len(zs)
    ix = np.clip((hit[:, 0] / res).astype(int), 0, nx - 1)
    iy = np.clip((hit[:, 1] / res).astype(int), 0, ny - 1)
    iz = np.clip((hit[:, 2] / res).astype(int), 0, nz - 1)
    out = {}
    for a, pos, neg, i, j in ((0, "wall3", "wall1", iz, iy), (1, "wall4", "wall2", iz, ix), (2, "ceiling", "floor", iy, ix)):
        for face, m in ((pos, (axis == a) & (dirs[:, a] > 0)), (neg, (axis == a) & ~(dirs[:, a] > 0))):
            if m.any():
                out[face] = (i[m], j[m])
    return out


def view_matrix(points, n_rays, res):
    dirs = sphere_directions(n_rays)
    xs, ys, zs = room.face_axes(res)
    shape = {"floor": (len(ys), len(xs)), "ceiling": (len(ys), len(xs)), "wall2": (len(zs), len(xs)),
             "wall4": (len(zs), len(xs)), "wall3": (len(zs), len(ys)), "wall1": (len(zs), len(ys))}
    offset, o = {}, 0
    for f in room.FACES:
        offset[f] = o
        o += shape[f][0] * shape[f][1]
    rows, cols, vals = [], [], []
    for r, p in enumerate(points):
        hits = cast_rays(p, dirs, res)
        n = sum(len(i) for i, _ in hits.values())
        for f, (i, j) in hits.items():
            rows.append(np.full(len(i), r))
            cols.append(offset[f] + i * shape[f][1] + j)
            vals.append(np.full(len(i), 1.0 / n))
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    return csr_matrix((vals, (rows, cols)), shape=(len(points), o))


def radiant_temperature(V, fields):
    t4 = np.concatenate([(np.ravel(fields[f]) + 273.15) ** 4 for f in room.FACES])
    return np.asarray(V @ t4) ** 0.25 - 273.15


def surface_shares(points, n_rays, res):
    V = view_matrix(points, n_rays, res)
    xs, ys, zs = room.face_axes(res)
    win = room.window_mask(res, lower_edge=False)
    ind = {"floor": {"floor": np.ones((len(ys), len(xs)))}, "ceiling": {"ceiling": np.ones((len(ys), len(xs)))},
           "wall2": {"wall2": np.ones((len(zs), len(xs)))}, "wall4": {"wall4": np.ones((len(zs), len(xs)))},
           "wall3": {"wall3": np.ones((len(zs), len(ys)))}, "glazing": {"wall1": win.astype(float)},
           "wall1": {"wall1": (~win).astype(float)}}
    zero = {"floor": (len(ys), len(xs)), "ceiling": (len(ys), len(xs)), "wall2": (len(zs), len(xs)),
            "wall4": (len(zs), len(xs)), "wall3": (len(zs), len(ys)), "wall1": (len(zs), len(ys))}
    S = np.zeros((len(points), len(room.SURFACES)))
    for k, s in enumerate(room.SURFACES):
        vec = np.concatenate([np.ravel(ind[s].get(f, np.zeros(zero[f]))) for f in room.FACES])
        S[:, k] = np.asarray(V @ vec).ravel()
    return S
