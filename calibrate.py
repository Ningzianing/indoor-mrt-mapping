import json
import os

import config as cfg
import m4_surface_model as m4
import window_model as wm


def main():
    cal = wm.calibrate()
    cal["m4"] = m4.calibrate(cal)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUT_DIR, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
