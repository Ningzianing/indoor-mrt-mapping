# MRT reconstruction (M.1–M.4)

Requirements: Python 3, numpy, pandas, scipy, openpyxl.

| File | Content |
|---|---|
| config.py | days, shading condition, window orientation, constants |
| room.py | room geometry, sensor and globe positions |
| data.py | reading of the measurement files |
| shortwave.py | solar geometry, separation, sky model, glazing transmittance, indoor shortwave irradiance |
| longwave.py | ray casting and longwave radiant temperature |
| thermocouples.py | surface-temperature fields from the thermocouples |
| thermal_camera.py | surface-temperature fields from the thermal cameras |
| window_model.py | one-day calibrated window model |
| m1_globe.py | M.1 |
| m2_reference.py | M.2-1 and M.2-2 |
| m3_thermal_camera.py | M.3 |
| m4_surface_model.py | M.4 |
