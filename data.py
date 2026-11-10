import os
from functools import lru_cache

import numpy as np
import pandas as pd

import config as cfg

NIGHT = ("00:00", "03:00")
BROKEN = {"床表面温度D④", "床表面温度F③", "天井表面温度E⑤"}


@lru_cache(maxsize=None)
def sheet(day, name):
    return pd.read_excel(os.path.join(cfg.DATA_DIR, f"{day}.xlsx"), sheet_name=name, index_col=0)


def _zeroed(d):
    return d - d.between_time(*NIGHT).mean()


@lru_cache(maxsize=None)
def irradiance(day):
    m, s = sheet(day, "気象データ"), sheet(day, "日射計")
    d = pd.DataFrame({"ghi": m["屋根上全天日射計"], "dni": m["直達日射計"], "vertical": m["鉛直面日射計"],
                      "window_upper": s["窓室内側上半分鉛直日射計"], "window_lower": s["窓室内側下半分鉛直日射計"],
                      "floor_B": s["床日射計B③"]}).astype(float)
    return _zeroed(d)


@lru_cache(maxsize=None)
def air(day):
    m, a = sheet(day, "気象データ"), sheet(day, "上下温度")
    return pd.DataFrame({"Ta": a["上下温度_室温"], "To": m["外気温2"]}).astype(float)


@lru_cache(maxsize=None)
def globe_temperatures(day):
    g = sheet(day, "ｸﾞﾛｰﾌﾞ放射温度")
    return g[[c for c in g.columns if str(c).startswith("ｸﾞﾛｰﾌﾞ温度")]].astype(float)


def mid_minute(df):
    df = df.copy()
    df.index = df.index.tz_localize(cfg.TZ) + pd.Timedelta(seconds=30)
    return df


@lru_cache(maxsize=None)
def surface_temperatures(day):
    parts = []
    for name in ("床表面温度", "天井表面温度", "壁表面温度", "室内側ｶﾞﾗｽ"):
        d = sheet(day, name).apply(pd.to_numeric, errors="coerce")
        d = d[[c for c in d.columns if "温度" in str(c) and c not in BROKEN]]
        parts.append(d.where(d < 1000))
    s = pd.concat(parts, axis=1)
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated()]
    full = pd.date_range(s.index[0].normalize(), periods=1440, freq="min")
    s = s.reindex(full).interpolate(limit_direction="both").round(2)
    if cfg.SHADING[day] == "interior":
        x = sheet(day, "ｽｸﾘｰﾝ").apply(pd.to_numeric, errors="coerce")
        x = x.where(x < 1000)
        for z in ("0100", "0600", "1100", "1700"):
            for p in "bd":
                c = f"ｽｸﾘｰﾝ表面温度{p}{z}"
                v = x[c] if c in x and x[c].notna().mean() > 0.9 else x[f"ｽｸﾘｰﾝ表面温度d{z}"]
                s[c] = v.reindex(s.index)
        s = s.interpolate(limit_direction="both")
    return s


def ten_minute(df):
    return df.resample(f"{cfg.TIME_STEP_MIN}min").mean()
