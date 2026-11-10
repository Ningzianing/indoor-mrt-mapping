# MRT reconstruction (M.1–M.4)

Requirements: Python 3, numpy, pandas, scipy, openpyxl.

Input layout (relative to the working directory):

    rooftop-data/<YYYYMMDD>.xlsx
    thermal/<camera>/<yymmdd>/<yymmdd_HHMM>_<yymmdd_HHMM>_avg10min_temp_c_matrix.csv

Run order:

    python calibrate.py                 -> output/calibration.json
    python run_globes.py [days]         -> output/mrt_globes_<day>.csv
    python run_maps.py [days]           -> output/mrt_maps_<day>.npz
    python run_surfaces.py [days]       -> output/surface_temperatures_<day>.csv

Without arguments the three run scripts process all validation days listed in config.py.

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
