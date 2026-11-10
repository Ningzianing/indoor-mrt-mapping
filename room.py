import numpy as np

import config as cfg
import shortwave as sw

DEPTH = 3.97
WIDTH = 3.97
HEIGHT = 2.4
SETBACK = 0.12

WINDOW_Y = (1.15, 2.81)
WINDOW_Z = (0.075, 2.125)
WINDOWS = [sw.Window(x0=WINDOW_Y[0], x1=WINDOW_Y[1], z0=WINDOW_Z[0], z1=WINDOW_Z[1])]

COLUMN_X = {"A": 0.1, "B": 0.6, "C": 1.41, "D": 2.22, "E": 3.03, "F": 3.84}
ROW_Y = {"①": 0.1, "②": 1.235, "③": 1.985, "④": 2.735, "⑤": 3.87}
Y_MID = 1.985

GLOBES = {"AB③1100": (0.35, Y_MID, 1.1), "B③0600": (0.6, Y_MID, 0.6), "B③1100": (0.6, Y_MID, 1.1),
          "B③1700": (0.6, Y_MID, 1.7), "C③1100": (1.41, Y_MID, 1.1), "D③1100": (2.22, Y_MID, 1.1),
          "E③1100": (3.03, Y_MID, 1.1)}

FLOOR_PYRANOMETER_B = (Y_MID, 0.83 + SETBACK, 0.0)
FLOOR_CENTRE = (Y_MID, 2.45 + SETBACK, 0.001)

SURFACES = ["glazing", "wall1", "wall2", "wall4", "wall3", "floor", "ceiling"]
FACES = ["floor", "ceiling", "wall2", "wall4", "wall3", "wall1"]


def window_frame(p):
    return (p[1], p[0] + SETBACK, p[2])


def site(day, azimuth_offset=0.0):
    return sw.Site(cfg.LATITUDE, cfg.LONGITUDE, (cfg.WINDOW_AZIMUTH[day] + azimuth_offset) % 360, cfg.GROUND_ALBEDO)


def face_axes(res):
    xs = np.arange(res / 2, DEPTH, res)
    ys = np.arange(res / 2, WIDTH, res)
    zs = np.arange(res / 2, HEIGHT, res)
    return xs, ys, zs


def window_mask(res, lower_edge=True):
    _, ys, zs = face_axes(res)
    gy, gz = np.meshgrid(ys, zs)
    m = (gy >= WINDOW_Y[0]) & (gy <= WINDOW_Y[1]) & (gz <= WINDOW_Z[1])
    return m & (gz >= WINDOW_Z[0]) if lower_edge else m


def plane_grid():
    xs = np.arange(cfg.GRID_STEP / 2, DEPTH, cfg.GRID_STEP)
    ys = np.arange(cfg.GRID_STEP / 2, WIDTH, cfg.GRID_STEP)
    return xs, ys, [(x, y, cfg.PLANE_HEIGHT) for x in xs for y in ys]
