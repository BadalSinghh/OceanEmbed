import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd

d = Path('data/raw/test')
# Wind files: the primary ones used by download script are sample_wind_u.nc / sample_wind_v.nc
# But sample_l4_wind_3d.nc may be the 3-day combined file
files = {
    'sample_wind_u.nc': 'eastward_wind',
    'sample_wind_v.nc': 'northward_wind',
    'sample_l4_wind_3d.nc': 'eastward_wind',
}
for f, var in files.items():
    fp = d / f
    if not fp.exists():
        print(f"MISSING: {f}")
        continue
    ds = xr.open_dataset(fp)
    size = fp.stat().st_size
    vars_ = list(ds.data_vars)
    t = len(ds['time'].values) if 'time' in ds.coords else 'N/A'
    t0 = str(ds['time'].values[0])[:22] if 'time' in ds.coords and t != 'N/A' else 'N/A'
    t1 = str(ds['time'].values[-1])[:22] if 'time' in ds.coords and t != 'N/A' else 'N/A'
    print(f"FILE: {f}  size={size:,}")
    print(f"  vars={vars_}  N_times={t}")
    print(f"  time_range: {t0} to {t1}")
    if var in ds:
        da = ds[var]
        vals = da.values
        nan_pct = 100.0 * np.sum(np.isnan(vals)) / vals.size
        lats = ds['latitude'].values if 'latitude' in ds.coords else None
        lons = ds['longitude'].values if 'longitude' in ds.coords else None
        print(f"  shape={dict(da.sizes)}  units={da.attrs.get('units','?')}  NaN%={nan_pct:.2f}")
        if lats is not None:
            lat_res = float(np.median(np.diff(np.sort(np.unique(lats)))))
            print(f"  lat [{lats.min():.4f}, {lats.max():.4f}]  res={lat_res:.4f}")
        if lons is not None:
            lon_res = float(np.median(np.diff(np.sort(np.unique(lons)))))
            print(f"  lon [{lons.min():.4f}, {lons.max():.4f}]  res={lon_res:.4f}")
        # Check if daily timestamps (if hourly, needs aggregation)
        if t != 'N/A' and t > 1:
            times = pd.to_datetime(ds['time'].values)
            delta = times[1] - times[0]
            print(f"  time step: {delta}  => {'HOURLY - needs daily aggregation' if delta.total_seconds() < 86400 else 'DAILY - ok'}")
    ds.close()
    print()
