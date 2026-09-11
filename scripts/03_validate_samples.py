"""
03_validate_samples.py

Validates the 5 successfully downloaded NetCDF sample files in data/raw/test/.
For each dataset checks:
- Variable names and dimensions
- Coordinate names (lat, lon, time)
- Latitude ordering, longitude convention
- Time range and daily timestamps
- Spatial extent
- Native resolution
- Units
- Missing values / NaN percentage
- Min / Max / Mean
- Whether data can be mapped onto a common 0.25 x 0.25 daily grid

For GLORYS thetao specifically also:
- Inspects native depth coordinate
- Maps 15 requested SIH depths to nearest native depths
- Saves mapping to results/depth_mapping.csv
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

DATA_DIR = Path("data/raw/test")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DEPTHS_M = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

DATASET_MAP = {
    "sample_sst.nc":       {"key": "SST",       "var": "analysed_sst",  "units_note": "K (needs K->C conversion)"},
    "sample_multiobs_sss_3d.nc": {"key": "SSS",       "var": "sos",  "units_note": "PSU (pss-78) — daily multiobs analysis"},
    "sample_sla.nc":       {"key": "SLA",       "var": "sla",           "units_note": "m"},
    "sample_current_u.nc": {"key": "CURRENT_U", "var": "ugos",          "units_note": "m/s"},
    "sample_current_v.nc": {"key": "CURRENT_V", "var": "vgos",          "units_note": "m/s"},
    "sample_wind_u.nc":    {"key": "WIND_U",    "var": "eastward_wind", "units_note": "m/s"},
    "sample_wind_v.nc":    {"key": "WIND_V",    "var": "northward_wind","units_note": "m/s"},
    "sample_glorys_thetao.nc": {"key": "GLORYS_THETAO", "var": "thetao", "units_note": "degC"},
}

def latlon_resolution(coord):
    vals = np.sort(np.unique(coord.values))
    if len(vals) < 2:
        return float("nan")
    diffs = np.diff(vals)
    return float(np.median(diffs))

def validate_dataset(filepath, var_name):
    print("\n" + "=" * 70)
    print(f"FILE: {filepath.name}")
    print("=" * 70)
    try:
        ds = xr.open_dataset(filepath)
    except Exception as e:
        print(f"  [ERROR] Cannot open file: {e}")
        return None

    # Variables
    print(f"  All variables: {list(ds.data_vars)}")
    print(f"  Coordinates:   {list(ds.coords)}")

    if var_name not in ds.data_vars:
        print(f"  [ERROR] Expected variable '{var_name}' NOT FOUND. Available: {list(ds.data_vars)}")
        ds.close()
        return None

    da = ds[var_name]
    dims = dict(da.sizes)
    print(f"  Dimensions:    {dims}")
    print(f"  dtype:         {da.dtype}")
    print(f"  attrs/units:   {da.attrs.get('units', 'N/A')}")

    # Detect coordinate names
    lat_name = next((c for c in da.dims if "lat" in c.lower()), None)
    lon_name = next((c for c in da.dims if "lon" in c.lower()), None)
    time_name = next((c for c in da.dims if "time" in c.lower()), None)
    depth_name = next((c for c in da.dims if c.lower() in ("depth", "elevation", "level", "plev", "z")), None)

    print(f"  Lat dim:       {lat_name}")
    print(f"  Lon dim:       {lon_name}")
    print(f"  Time dim:      {time_name}")
    print(f"  Depth dim:     {depth_name}")

    # Lat/Lon extent and ordering
    if lat_name and lat_name in ds.coords:
        lats = ds[lat_name].values
        print(f"  Lat range:     [{lats.min():.4f}, {lats.max():.4f}]  ordering={'ascending' if lats[0] < lats[-1] else 'descending'}")
        lat_res = latlon_resolution(ds[lat_name])
        print(f"  Lat resolution: {lat_res:.4f}°")
    else:
        print("  [WARN] No lat coordinate found")

    if lon_name and lon_name in ds.coords:
        lons = ds[lon_name].values
        convention = "0-360" if lons.max() > 180 else "-180 to 180"
        print(f"  Lon range:     [{lons.min():.4f}, {lons.max():.4f}]  convention={convention}")
        lon_res = latlon_resolution(ds[lon_name])
        print(f"  Lon resolution: {lon_res:.4f}°")
    else:
        print("  [WARN] No lon coordinate found")

    # Time range
    if time_name and time_name in ds.coords:
        times = ds[time_name].values
        print(f"  Time range:    {str(times[0])[:19]} to {str(times[-1])[:19]}")
        print(f"  N timesteps:   {len(times)}")
        if len(times) > 1:
            delta = pd.to_datetime(times[1]) - pd.to_datetime(times[0])
            print(f"  Time step:     {delta}")
    else:
        print("  [WARN] No time coordinate found")

    # Missing values
    vals = da.values
    total = vals.size
    nan_count = int(np.sum(np.isnan(vals)))
    nan_pct = 100.0 * nan_count / total if total > 0 else 0.0
    finite_vals = vals[np.isfinite(vals)]
    print(f"  Total elements: {total}")
    print(f"  NaN count:     {nan_count}  ({nan_pct:.2f}%)")
    if finite_vals.size > 0:
        print(f"  Min:           {float(np.min(finite_vals)):.4f}")
        print(f"  Max:           {float(np.max(finite_vals)):.4f}")
        print(f"  Mean:          {float(np.mean(finite_vals)):.4f}")
    else:
        print("  [WARN] All values are NaN or masked")

    # Grid mapping check
    if lat_name and lon_name and lat_name in ds.coords and lon_name in ds.coords:
        lat_res = latlon_resolution(ds[lat_name])
        lon_res = latlon_resolution(ds[lon_name])
        needs_regrid = not (abs(lat_res - 0.25) < 0.01 and abs(lon_res - 0.25) < 0.01)
        print(f"  Needs regrid to 0.25°: {needs_regrid}")

    ds.close()
    return da


def map_glorys_depths(filepath):
    print("\n" + "=" * 70)
    print("GLORYS NATIVE DEPTH MAPPING")
    print("=" * 70)
    try:
        ds = xr.open_dataset(filepath)
    except Exception as e:
        print(f"  [ERROR] Cannot open GLORYS file: {e}")
        return

    depth_name = next((c for c in ds.coords if c.lower() in ("depth", "elevation", "level", "z")), None)
    if depth_name is None:
        print("  [ERROR] No depth coordinate found in GLORYS file!")
        ds.close()
        return

    native_depths = ds[depth_name].values
    print(f"  Native depth coordinate: '{depth_name}'")
    print(f"  N native levels: {len(native_depths)}")
    print(f"  Native depth range: [{native_depths.min():.4f}, {native_depths.max():.4f}] m")
    print(f"  First 25 native depths: {[round(float(d), 2) for d in native_depths[:25]]}")

    records = []
    print(f"\n  {'Requested (m)':<18} {'Selected native (m)':<22} {'Native index':<14} {'Abs diff (m)':<14}")
    print("  " + "-" * 68)
    for req_depth in TARGET_DEPTHS_M:
        idx = int(np.argmin(np.abs(native_depths - req_depth)))
        native_val = float(native_depths[idx])
        diff = abs(native_val - req_depth)
        print(f"  {req_depth:<18} {native_val:<22.4f} {idx:<14} {diff:<14.4f}")
        records.append({
            "requested_depth_m": req_depth,
            "selected_native_depth_m": round(native_val, 4),
            "native_depth_index": idx,
            "abs_diff_m": round(diff, 4)
        })

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "depth_mapping.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  [SAVED] Depth mapping saved to: {out_path}")
    ds.close()


def main():
    print("=" * 70)
    print("OCEAEMBED SAMPLE DATASET VALIDATION")
    print("=" * 70)

    available = list(DATA_DIR.glob("*.nc"))
    print(f"Files found in {DATA_DIR}: {[f.name for f in available]}")

    # Validate each successfully downloaded file
    for fname, meta in DATASET_MAP.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"\n[MISSING] {fname} — was not downloaded. SKIP.")
            continue
        validate_dataset(fpath, meta["var"])

    # GLORYS depth mapping
    glorys_path = DATA_DIR / "sample_glorys_thetao.nc"
    if glorys_path.exists():
        map_glorys_depths(glorys_path)
    else:
        print("\n[MISSING] sample_glorys_thetao.nc — cannot perform depth mapping.")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
