import numpy as np
import pandas as pd

import config as cfg
import data
import room
import shortwave as sw


def globe_mrt(tg, ta):
    d, eps, v = cfg.GLOBE_DIAMETER, cfg.GLOBE_EMISSIVITY, cfg.AIR_SPEED
    dt = tg - ta
    h = np.maximum(6.3 * v ** 0.6 / d ** 0.4, 1.4 * (np.abs(dt) / d) ** 0.25)
    return ((tg + 273.15) ** 4 + h / (eps * sw.SIGMA) * dt) ** 0.25 - 273.15


def compute(day):
    tg = data.ten_minute(data.globe_temperatures(day))
    ta = data.ten_minute(data.air(day)["Ta"])
    return pd.DataFrame({g: globe_mrt(tg[f"ｸﾞﾛｰﾌﾞ温度{g}"], ta) for g in room.GLOBES})
