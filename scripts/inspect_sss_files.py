import xarray as xr
from pathlib import Path
import numpy as np

d = Path('data/raw/test')
files = ['sample_sss.nc', 'sample_sss_(1).nc', 'sample_sss_(2).nc', 'sample_multiobs_sss_3d.nc']
for f in files:
    fp = d / f
    if not fp.exists():
        continue
    ds = xr.open_dataset(fp)
    size = fp.stat().st_size
    vars_ = list(ds.data_vars)
    coords = list(ds.coords)
    t = len(ds['time'].values) if 'time' in ds.coords else 'N/A'
    t0 = str(ds['time'].values[0])[:19] if 'time' in ds.coords and t != 'N/A' else 'N/A'
    t1 = str(ds['time'].values[-1])[:19] if 'time' in ds.coords and t != 'N/A' else 'N/A'
    print(f"FILE: {f}  size={size:,}")
    print(f"  vars={vars_}  coords={coords}  N_times={t}")
    print(f"  time_range: {t0} to {t1}")
    for v in vars_:
        da = ds[v]
        vals = da.values
        nan_pct = 100.0 * np.sum(np.isnan(vals)) / vals.size
        print(f"  var={v}  shape={dict(da.sizes)}  units={da.attrs.get('units','?')}  NaN%={nan_pct:.2f}")
    ds.close()
    print()
