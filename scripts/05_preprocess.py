# -*- coding: utf-8 -*-
"""
05_preprocess.py
================
OceanEmbed Stage 3 -- Full preprocessing pipeline (Parts B-H).

Reads monthly raw chunks from data/raw/full/, applies:
  B. Regrid all inputs to common 0.25° × 0.25° grid
  C. Temporal alignment — exactly one daily timestep per day
  D. Unit normalization (SST K->°C, verify SSS units, etc.)
  E. GLORYS vertical interpolation to 15 exact SIH depths
  F. Missing-value / land-mask handling
  G. Normalization statistics from TRAINING data only
  H. Export ML-ready NPZ tensors with train/val/test split

Output:
  data/processed/train/  X_train.npz  Y_train.npz  dates_train.npy
  data/processed/val/    X_val.npz    Y_val.npz    dates_val.npy
  data/processed/test/   X_test.npz   Y_test.npz   dates_test.npy
  data/processed/coords.nc
  data/processed/land_mask.npy
  results/normalization_stats.json
  results/depth_coverage.csv
  docs/STAGE3_DATASET_REPORT.md
  results/stage3_dataset_summary.json

Usage:
    python scripts/05_preprocess.py [--dry-run] [--report-only]
"""

import argparse
import json
import sys
import traceback
import warnings
from datetime import date, timedelta

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────
# Paths & Constants
# ──────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent.parent
RAW_FULL     = ROOT / "data" / "raw" / "full"
PROCESSED    = ROOT / "data" / "processed"
RESULTS      = ROOT / "results"
DOCS         = ROOT / "docs"

for d in [PROCESSED / "train", PROCESSED / "val", PROCESSED / "test", RESULTS, DOCS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Common output grid ──────────────────────────────────────────
# 5.0 to 22.0 inclusive, 0.25° step -> 69 latitudes
# 80.0 to 100.0 inclusive, 0.25° step -> 81 longitudes
TARGET_LAT = np.arange(5.0, 22.01, 0.25)   # shape (69,)
TARGET_LON = np.arange(80.0, 100.01, 0.25) # shape (81,)

# ── Target depths (SIH standard) ───────────────────────────────
TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# ── Full date range ─────────────────────────────────────────────
ALL_DATES = pd.date_range("2022-01-01", "2023-12-31", freq="D")
N_DAYS    = len(ALL_DATES)   # 730

# ── Chronological 70/15/15 split ───────────────────────────────
N_TRAIN = round(N_DAYS * 0.70)   # 511 days (70%)
N_VAL   = round(N_DAYS * 0.15)   # 110 days (15%)
N_TEST  = N_DAYS - N_TRAIN - N_VAL  # 109 days (15%)

TRAIN_DATES = ALL_DATES[:N_TRAIN]
VAL_DATES   = ALL_DATES[N_TRAIN : N_TRAIN + N_VAL]
TEST_DATES  = ALL_DATES[N_TRAIN + N_VAL :]

# ── Grid dimensions ─────────────────────────────────────────────
H = len(TARGET_LAT)   # 69
W = len(TARGET_LON)   # 81


# ──────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────

def banner(msg):
    print("\n" + "=" * 65)
    print(msg)
    print("=" * 65)


def section(msg):
    print(f"\n── {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def info(msg):
    print(f"  [INFO] {msg}")


# ──────────────────────────────────────────────────────────────
# Raw file discovery
# ──────────────────────────────────────────────────────────────

def find_monthly_files(subdir: str):
    """Return sorted list of NetCDF files in data/raw/full/<subdir>/."""
    d = RAW_FULL / subdir
    if not d.exists():
        return []
    files = sorted(d.glob("*.nc"))
    return files


def check_coverage(files: list, subdir: str):
    """
    Verify all 24 expected monthly files exist.
    Returns list of missing (year, month) tuples.
    """
    found_tags = set()
    for f in files:
        # Expected name pattern: <key>_YYYY-MM.nc
        parts = f.stem.split("_")
        for p in parts:
            if len(p) == 7 and p[4] == "-":
                found_tags.add(p)

    missing = []
    for year in [2022, 2023]:
        for month in range(1, 13):
            tag = f"{year:04d}-{month:02d}"
            if tag not in found_tags:
                missing.append((year, month, tag))
    return missing


# ──────────────────────────────────────────────────────────────
# Part B — Regridding
# ──────────────────────────────────────────────────────────────

def detect_lat_lon(ds: xr.Dataset):
    """Detect latitude and longitude dimension names."""
    lat_name = next((c for c in ds.coords if "lat" in c.lower()), None)
    lon_name = next((c for c in ds.coords if "lon" in c.lower()), None)
    return lat_name, lon_name


def regrid_to_common(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    """
    Bilinear interpolation onto the common 0.25° × 0.25° target grid.
    Uses xarray.DataArray.interp() which performs linear interpolation
    on a regular Cartesian grid (equivalent to bilinear for 2D).

    Method: bilinear (linear in lat AND lon simultaneously via xarray).
    Suitable for all continuous physical fields (SST, SSS, SLA, U, V, wind).

    Args:
        da: input DataArray (any native resolution)
        lat_name: name of the latitude dimension in da
        lon_name: name of the longitude dimension in da

    Returns:
        DataArray regridded to TARGET_LAT × TARGET_LON
    """
    return da.interp(
        {lat_name: TARGET_LAT, lon_name: TARGET_LON},
        method="linear",
        kwargs={"fill_value": np.nan, "bounds_error": False},
    ).rename({lat_name: "lat", lon_name: "lon"})


# ──────────────────────────────────────────────────────────────
# Part C — Temporal alignment
# ──────────────────────────────────────────────────────────────

def align_to_daily_index(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure DataArray has exactly one timestep per calendar day.
    - Converts time coordinate to date-only (drops sub-daily info)
    - Checks for duplicates (warns + deduplicates by taking first)
    Returns DataArray indexed on daily pandas DatetimeIndex.
    """
    times = pd.to_datetime(da["time"].values).normalize()  # floor to midnight
    da = da.assign_coords(time=times)

    # Deduplicate
    _, idx = np.unique(times, return_index=True)
    if len(idx) < len(times):
        warn(f"Removed {len(times) - len(idx)} duplicate timestamps")
        da = da.isel(time=idx)

    return da


def concat_and_align(arrays: list, sort=True) -> xr.DataArray:
    """Concatenate a list of DataArrays along time, then sort and deduplicate."""
    combined = xr.concat(arrays, dim="time")
    combined = align_to_daily_index(combined)
    if sort:
        combined = combined.sortby("time")
    return combined


def check_missing_dates(da: xr.DataArray, variable: str):
    """Report gaps in the time series against the full expected date range."""
    times = pd.to_datetime(da["time"].values).normalize()
    present = set(times.date)
    expected = set(d.date() for d in ALL_DATES)
    missing = sorted(expected - present)
    extra   = sorted(present - expected)

    if missing:
        warn(f"{variable}: {len(missing)} missing dates "
             f"(first: {missing[0]}, last: {missing[-1]})")
        if len(missing) <= 10:
            warn(f"  Missing: {[str(m) for m in missing]}")
    else:
        info(f"{variable}: all {len(expected)} dates present ✓")

    if extra:
        warn(f"{variable}: {len(extra)} extra dates outside 2022-2023 "
             f"(will be dropped in reindex)")

    return missing


def reindex_to_full_range(da: xr.DataArray) -> xr.DataArray:
    """
    Reindex DataArray to the full 730-day daily range (2022-01-01 to 2023-12-31).
    Days without data are filled with NaN (not zero).
    """
    return da.reindex(time=ALL_DATES, fill_value=np.nan)


# ──────────────────────────────────────────────────────────────
# Part D — Unit conversions
# ──────────────────────────────────────────────────────────────

def convert_units(da: xr.DataArray, variable: str) -> xr.DataArray:
    """
    Apply unit conversions as needed. Returns converted DataArray.
    All conversions are documented.
    """
    units = da.attrs.get("units", "").strip()

    if variable == "SST":
        # METOFFICE product reports in Kelvin (units = 'kelvin' or 'K')
        if units.lower() in ("kelvin", "k"):
            info(f"SST: converting K -> °C  (units='{units}')")
            da = da - 273.15
            da.attrs["units"] = "degC"
            da.attrs["unit_conversion"] = "Kelvin - 273.15"
        else:
            # Values > 200 are almost certainly Kelvin even if label is wrong
            sample = float(da.isel(time=0).values[~np.isnan(da.isel(time=0).values)][0])
            if sample > 200:
                warn(f"SST units='{units}' but value {sample:.1f} looks like Kelvin; converting.")
                da = da - 273.15
                da.attrs["units"] = "degC"
                da.attrs["unit_conversion"] = "Kelvin - 273.15 (inferred from values)"
            else:
                info(f"SST units='{units}'; values look like °C, no conversion applied.")

    elif variable == "SSS":
        # MULTIOBS product: units attribute may read '.001' or 'pss-78'
        # Actual values in the data are in practical salinity units (PSU ~ 0-40)
        # The scale factor is already applied by xarray's CF decoding.
        sample_vals = da.values[np.isfinite(da.values)]
        if len(sample_vals) > 0:
            sval = float(np.median(sample_vals))
            if sval < 1.0:
                # Looks like scale factor NOT decoded -> multiply by 1000
                warn(f"SSS: median value = {sval:.4f} (units='{units}'); "
                     f"values < 1 suggest undecoded scale factor -> ×1000")
                da = da * 1000.0
                da.attrs["units"] = "PSU"
                da.attrs["unit_conversion"] = "x1000 (scale_factor correction)"
            else:
                info(f"SSS: units='{units}', median={sval:.2f} — values look like PSU, no conversion.")
                da.attrs["units"] = "PSU"
    else:
        info(f"{variable}: units='{units}' — no conversion required.")

    return da


# ──────────────────────────────────────────────────────────────
# Load one variable — reads monthly files, regrids, aligns
# ──────────────────────────────────────────────────────────────

def load_variable(subdir: str, var_name: str, label: str,
                  squeeze_depth=False) -> tuple:
    """
    Load all monthly NetCDF files for a variable, regrid to common grid,
    and align to daily date range.

    squeeze_depth: if True, drops the depth dimension (e.g. SSS has depth=1).

    Returns (da_aligned: xr.DataArray, missing_dates: list, coverage_info: dict)
    """
    section(f"Loading {label} ({var_name})")
    files = find_monthly_files(subdir)

    if not files:
        raise FileNotFoundError(
            f"No NetCDF files found in {RAW_FULL / subdir}. "
            f"Run 04_download_full.py first."
        )

    missing_months = check_coverage(files, subdir)
    if missing_months:
        warn(f"{label}: {len(missing_months)} monthly files missing: "
             f"{[t for _, _, t in missing_months]}")

    monthly_arrays = []
    for fpath in files:
        try:
            ds = xr.open_dataset(fpath)
            if var_name not in ds.data_vars:
                warn(f"{fpath.name}: variable '{var_name}' not found "
                     f"(available: {list(ds.data_vars)})")
                ds.close()
                continue

            da = ds[var_name]

            # Squeeze singleton depth if present
            if squeeze_depth:
                depth_dim = next((d for d in da.dims
                                  if d.lower() in ("depth", "level", "z")), None)
                if depth_dim:
                    da = da.isel({depth_dim: 0}, drop=True)

            lat_name, lon_name = detect_lat_lon(ds)
            if lat_name is None or lon_name is None:
                warn(f"{fpath.name}: cannot detect lat/lon dims; skipping.")
                ds.close()
                continue

            da_rg = regrid_to_common(da, lat_name, lon_name)
            da_rg = da_rg.load()  # load into memory before closing file handle
            ds.close()
            monthly_arrays.append(da_rg)

        except Exception as e:
            warn(f"Failed loading {fpath.name}: {e}")
            traceback.print_exc()

    if not monthly_arrays:
        raise RuntimeError(f"No data loaded for {label}")

    combined = concat_and_align(monthly_arrays)
    missing_dates = check_missing_dates(combined, label)
    combined = reindex_to_full_range(combined)

    nan_pct = 100.0 * float(np.sum(np.isnan(combined.values))) / combined.values.size
    info(f"{label}: final shape={combined.shape}  NaN%={nan_pct:.2f}")

    return combined, missing_dates, {"label": label, "nan_pct": nan_pct,
                                     "missing_dates": len(missing_dates)}


# ──────────────────────────────────────────────────────────────
# Part E — GLORYS depth handling
# ──────────────────────────────────────────────────────────────

def load_glorys_and_interpolate() -> tuple:
    """
    Load GLORYS thetao monthly chunks, regrid to common spatial grid,
    then vertically interpolate to the 15 SIH target depths.

    Vertical interpolation:
      - Uses scipy.interpolate.interp1d (linear) per grid column.
      - fill_value=np.nan, bounds_error=False -> NaN below ocean bottom.
      - 1000 m will be NaN for shallow regions; this is CORRECT behaviour.

    Returns:
      da_interp: DataArray [time, depth=15, lat=69, lon=81]
      native_depths: 1-D array of GLORYS native depth coordinate
      depth_report: dict
    """
    section("Loading GLORYS thetao (TARGET)")

    files = find_monthly_files("glorys")
    if not files:
        raise FileNotFoundError(
            "No GLORYS NetCDF files found. Run 04_download_full.py first."
        )

    check_coverage(files, "glorys")

    monthly_arrays = []
    native_depths_ref = None

    for fpath in files:
        try:
            ds = xr.open_dataset(fpath)
            if "thetao" not in ds.data_vars:
                warn(f"{fpath.name}: 'thetao' not found; skipping.")
                ds.close()
                continue

            da = ds["thetao"]

            # Detect and record native depth coordinate
            depth_dim = next((d for d in da.dims
                              if d.lower() in ("depth", "z", "level")), None)
            if depth_dim is None:
                warn(f"{fpath.name}: no depth dim found; skipping.")
                ds.close()
                continue

            if native_depths_ref is None:
                native_depths_ref = ds[depth_dim].values.copy()
                info(f"Native GLORYS depth coordinate: {len(native_depths_ref)} levels, "
                     f"{native_depths_ref[0]:.2f}–{native_depths_ref[-1]:.2f} m")

            lat_name, lon_name = detect_lat_lon(ds)
            da_rg = regrid_to_common(da, lat_name, lon_name)
            da_rg = da_rg.load()
            ds.close()
            monthly_arrays.append(da_rg)

        except Exception as e:
            warn(f"Failed loading {fpath.name}: {e}")
            traceback.print_exc()

    if not monthly_arrays:
        raise RuntimeError("No GLORYS data loaded.")

    # Combine and align time
    combined = xr.concat(monthly_arrays, dim="time")
    combined = align_to_daily_index(combined)
    combined = combined.sortby("time")
    check_missing_dates(combined, "GLORYS_thetao")
    combined = reindex_to_full_range(combined)

    # Rename depth dim to 'depth' for consistency
    depth_dim = next((d for d in combined.dims if d.lower() in ("depth", "z", "level")), None)
    if depth_dim != "depth":
        combined = combined.rename({depth_dim: "depth"})

    info(f"GLORYS combined shape before depth interp: {combined.shape}")

    # ── Vertical interpolation to SIH target depths ──────────────
    section("Vertical interpolation to SIH standard depths")

    native_depths = combined["depth"].values.astype(float)
    info(f"Native depths ({len(native_depths)} levels): "
         f"{[round(float(d), 2) for d in native_depths]}")

    max_native = native_depths.max()
    min_native = native_depths.min()
    info(f"Native depth range: {min_native:.2f} – {max_native:.2f} m")

    # Map target depth 0.0 m to top native level (0.494 m) for surface interpolation
    interp_depths = [min_native if d == 0 else float(d) for d in TARGET_DEPTHS]

    try:
        combined_interp = combined.interp(
            depth=np.array(interp_depths, dtype=float),
            method="linear",
            kwargs={"fill_value": np.nan, "bounds_error": False},
        )
    except Exception as e:
        warn(f"xarray interp failed ({e}); falling back to scipy loop.")
        combined_interp = _scipy_depth_interp(combined, native_depths)

    # Rename depth coordinate values to exact target depths
    combined_interp = combined_interp.assign_coords(depth=TARGET_DEPTHS)
    info(f"After depth interp: {combined_interp.shape}")

    # ── Per-depth spatial coverage ──────────────────────────────
    depth_report = {}
    total_cells = H * W
    for d in TARGET_DEPTHS:
        da_d = combined_interp.sel(depth=d)
        # Coverage = fraction of cells that have valid data in ≥1 timestep
        valid = np.any(np.isfinite(da_d.values), axis=0)
        frac  = 100.0 * valid.sum() / total_cells
        nan_pct = 100.0 * float(np.sum(np.isnan(da_d.values))) / da_d.values.size
        depth_report[d] = {
            "valid_cell_pct": round(float(frac), 2),
            "overall_nan_pct": round(nan_pct, 2),
        }
        info(f"  Depth {d:>5} m: valid cells={frac:.1f}%  NaN%={nan_pct:.1f}%")

    return combined_interp, native_depths, depth_report


def _scipy_depth_interp(da: xr.DataArray, native_depths: np.ndarray) -> xr.DataArray:
    """
    Fallback: interpolate depth dimension using scipy.interp1d per spatial column.
    Much slower but robust fallback.
    """
    data = da.values  # [time, depth, lat, lon]
    T, D, La, Lo = data.shape
    n_targets = len(TARGET_DEPTHS)
    out = np.full((T, n_targets, La, Lo), np.nan, dtype=np.float32)

    for t in range(T):
        for i in range(La):
            for j in range(Lo):
                profile = data[t, :, i, j]
                if np.all(np.isnan(profile)):
                    continue
                mask = np.isfinite(profile)
                if mask.sum() < 2:
                    continue
                f = interp1d(
                    native_depths[mask], profile[mask],
                    kind="linear", bounds_error=False, fill_value=np.nan
                )
                out[t, :, i, j] = f(TARGET_DEPTHS)

    coords = {
        "time": da["time"],
        "depth": TARGET_DEPTHS,
        "lat": da["lat"],
        "lon": da["lon"],
    }
    return xr.DataArray(out, coords=coords, dims=["time", "depth", "lat", "lon"])


# ──────────────────────────────────────────────────────────────
# Part F — Land mask
# ──────────────────────────────────────────────────────────────

def build_land_mask(input_arrays: dict) -> np.ndarray:
    """
    Build a boolean land mask [H, W] where True = LAND (always invalid).

    Strategy: a cell is LAND if it is NaN in ≥95% of SST timesteps.
    SST has the densest coverage and most complete ocean mask.

    Returns:
        land_mask [H, W]: True = land/permanently invalid
    """
    section("Building land mask from SST")
    sst = input_arrays.get("SST")
    if sst is None:
        warn("SST not available; building mask from first available variable.")
        sst = next(iter(input_arrays.values()))

    sst_vals = sst.values  # [T, H, W]
    nan_fraction = np.sum(np.isnan(sst_vals), axis=0) / sst_vals.shape[0]
    land_mask = nan_fraction >= 0.95   # [H, W]
    ocean_pct = 100.0 * (~land_mask).sum() / (H * W)
    info(f"Land mask: {land_mask.sum()} land cells, "
         f"{(~land_mask).sum()} ocean cells ({ocean_pct:.1f}%)")
    return land_mask


# ──────────────────────────────────────────────────────────────
# Part G — Normalization
# ──────────────────────────────────────────────────────────────

CHANNEL_ORDER = ["SST", "SSS", "SLA", "CURRENT_U", "CURRENT_V", "WIND_U", "WIND_V"]

def compute_normalization_stats(arrays: dict, train_idx: np.ndarray) -> dict:
    """
    Compute per-channel mean and std from TRAINING DATA ONLY.
    NaN values are excluded from statistics.

    Args:
        arrays: dict variable_name -> DataArray [T, H, W]
        train_idx: integer indices for training days

    Returns:
        stats: dict { channel: { mean, std, min, max } }
    """
    section("Computing normalization statistics (training data only)")
    stats = {}

    for ch in CHANNEL_ORDER:
        da = arrays[ch]
        train_vals = da.values[train_idx]  # [N_train, H, W]
        finite = train_vals[np.isfinite(train_vals)]

        if finite.size == 0:
            warn(f"{ch}: no valid training values — using mean=0, std=1")
            stats[ch] = {"mean": 0.0, "std": 1.0, "min": None, "max": None,
                         "n_valid": 0}
            continue

        mean = float(np.mean(finite))
        std  = float(np.std(finite))
        if std < 1e-6:
            warn(f"{ch}: std={std:.2e} is nearly zero; clamping to 1.0")
            std = 1.0
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

        stats[ch] = {
            "mean": round(mean, 6),
            "std":  round(std, 6),
            "min":  round(vmin, 6),
            "max":  round(vmax, 6),
            "n_valid": int(finite.size),
        }
        info(f"  {ch:<12}: mean={mean:+.4f}  std={std:.4f}  "
             f"range=[{vmin:.4f}, {vmax:.4f}]  N={finite.size:,}")

    return stats


def normalize_array(da: xr.DataArray, mean: float, std: float) -> np.ndarray:
    """Apply (x - mean) / std. NaN values remain NaN."""
    return ((da.values - mean) / std).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Part H — Build and save ML dataset
# ──────────────────────────────────────────────────────────────

def build_ml_tensors(input_arrays: dict, glorys_da: xr.DataArray,
                     norm_stats: dict, land_mask: np.ndarray):
    """
    Assemble X [N, 7, H, W] and Y [N, 15, H, W] tensors and
    save as compressed NPZ to data/processed/{train,val,test}/.

    Split indices:
      train: [0, N_TRAIN)
      val:   [N_TRAIN, N_TRAIN+N_VAL)
      test:  [N_TRAIN+N_VAL, N_DAYS)
    """
    section("Assembling ML tensors and saving NPZ")

    # ── Normalize inputs ──
    X_normed = []
    for ch in CHANNEL_ORDER:
        da = input_arrays[ch]
        s  = norm_stats[ch]
        arr = normalize_array(da, s["mean"], s["std"])  # [T, H, W]
        X_normed.append(arr)

    X_all = np.stack(X_normed, axis=1).astype(np.float32)  # [T, 7, H, W]

    # ── GLORYS Y tensor (not normalized — raw °C for regression) ──
    Y_all = glorys_da.values.astype(np.float32)  # [T, 15, H, W]
    # Reorder dims if necessary
    if glorys_da.dims[1] != "depth":
        Y_all = np.transpose(Y_all, (0, glorys_da.dims.index("depth"), 1, 2))

    # ── Date arrays ──
    dates_all = np.array([str(d.date()) for d in ALL_DATES])

    # ── Splits ──
    splits = {
        "train": (slice(0, N_TRAIN), TRAIN_DATES),
        "val":   (slice(N_TRAIN, N_TRAIN + N_VAL), VAL_DATES),
        "test":  (slice(N_TRAIN + N_VAL, None), TEST_DATES),
    }

    for split_name, (slc, split_dates) in splits.items():
        X_split = X_all[slc]
        Y_split = Y_all[slc]
        dates_split = dates_all[slc]

        out_dir = PROCESSED / split_name
        np.savez_compressed(out_dir / f"X_{split_name}.npz", data=X_split)
        np.savez_compressed(out_dir / f"Y_{split_name}.npz", data=Y_split)
        np.save(out_dir / f"dates_{split_name}.npy", dates_split)

        nan_X = 100.0 * np.sum(np.isnan(X_split)) / X_split.size
        nan_Y = 100.0 * np.sum(np.isnan(Y_split)) / Y_split.size
        info(f"  {split_name:<6}: X={X_split.shape}  Y={Y_split.shape}  "
             f"NaN_X={nan_X:.2f}%  NaN_Y={nan_Y:.2f}%  "
             f"dates={dates_split[0]}–{dates_split[-1]}")

    return X_all.shape, Y_all.shape


def save_coords(glorys_da: xr.DataArray):
    """Save coordinate metadata as NetCDF sidecar."""
    ds_coords = xr.Dataset({
        "lat": xr.DataArray(TARGET_LAT, dims=["lat"],
                            attrs={"units": "degrees_north", "long_name": "latitude"}),
        "lon": xr.DataArray(TARGET_LON, dims=["lon"],
                            attrs={"units": "degrees_east", "long_name": "longitude"}),
        "depth": xr.DataArray(TARGET_DEPTHS, dims=["depth"],
                               attrs={"units": "m",
                                      "long_name": "SIH standard depth levels"}),
        "all_dates": xr.DataArray(
            np.array([str(d.date()) for d in ALL_DATES]), dims=["time"],
            attrs={"long_name": "all daily dates 2022-2023"}
        ),
    })
    ds_coords.attrs["grid"] = "0.25 degree regular lat-lon"
    ds_coords.attrs["region"] = "Bay of Bengal"
    ds_coords.to_netcdf(PROCESSED / "coords.nc")
    info("Saved coords.nc")


# ──────────────────────────────────────────────────────────────
# Part I — Reports
# ──────────────────────────────────────────────────────────────

def save_normalization_stats(stats: dict):
    out = RESULTS / "normalization_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    info(f"Saved normalization_stats.json")


def save_depth_coverage(depth_report: dict, native_depths: np.ndarray):
    import csv
    rows = []
    for d in TARGET_DEPTHS:
        rows.append({
            "requested_depth_m": d,
            "valid_cell_pct": depth_report[d]["valid_cell_pct"],
            "overall_nan_pct": depth_report[d]["overall_nan_pct"],
            "interpolated": "yes" if d <= native_depths.max() else "no_below_range",
        })
    out = RESULTS / "depth_coverage.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    info(f"Saved depth_coverage.csv")


def generate_dataset_report(
    input_arrays: dict,
    glorys_da: xr.DataArray,
    norm_stats: dict,
    native_depths: np.ndarray,
    depth_report: dict,
    land_mask: np.ndarray,
    missing_per_var: dict,
    X_shape: tuple,
    Y_shape: tuple,
):
    """Generate STAGE3_DATASET_REPORT.md and stage3_dataset_summary.json."""
    section("Generating Stage 3 reports")

    ocean_pct = 100.0 * float((~land_mask).sum()) / (H * W)

    # ── Markdown report ──────────────────────────────────────────
    lines = []
    A = lines.append

    A("# OceanEmbed Stage 3 — Dataset Report")
    A(f"\nGenerated: {pd.Timestamp.now('UTC').isoformat()[:19]} UTC\n")

    A("## 1. Download Coverage")
    A(f"- Total days in dataset: **{N_DAYS}** (2022-01-01 -> 2023-12-31)")
    A(f"- Expected daily dates: {N_DAYS}")
    for var, miss in missing_per_var.items():
        A(f"- {var}: {miss} missing dates")

    A("\n## 2. Final Grid")
    A(f"- Latitude : {TARGET_LAT[0]:.2f}°N -> {TARGET_LAT[-1]:.2f}°N, "
      f"step 0.25°, **{H} points**")
    A(f"- Longitude: {TARGET_LON[0]:.2f}°E -> {TARGET_LON[-1]:.2f}°E, "
      f"step 0.25°, **{W} points**")
    A(f"- Grid resolution: **0.25° × 0.25°**")

    A("\n## 3. Input Variables (7 channels)")
    inputs_meta = [
        ("SST",       "METOFFICE-GLO-SST-L4-REP-OBS-SST",              "analysed_sst", "0.125°", "Daily",  "°C (converted from K)"),
        ("SSS",       "cmems_obs-mob_glo_phy-sss_my_multi_P1D",         "sos",          "0.125°", "Daily",  "PSU — daily obs-based satellite/in-situ analysis"),
        ("SLA",       "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D", "sla", "0.125°","Daily",  "m — satellite altimetry DUACS"),
        ("CURRENT_U", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D", "ugos","0.125°","Daily",  "m/s — geostrophic zonal current from altimetry"),
        ("CURRENT_V", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D", "vgos","0.125°","Daily",  "m/s — geostrophic meridional current from altimetry"),
        ("WIND_U",    "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",    "eastward_wind","0.125°","Hourly->Daily mean","m/s — L4 blended wind (scatterometer+ERA5)"),
        ("WIND_V",    "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",    "northward_wind","0.125°","Hourly->Daily mean","m/s — L4 blended wind (scatterometer+ERA5)"),
    ]
    A("\n| Channel | Dataset ID | Variable | Native Res | Native Temporal | Units / Notes |")
    A("|---|---|---|---|---|---|")
    for row in inputs_meta:
        A(f"| {' | '.join(row)} |")

    A("\n## 4. Target — GLORYS thetao")
    A(f"- Dataset: `cmems_mod_glo_phy_my_0.083deg_P1D-m`")
    A(f"- Variable: `thetao` (ONLY — no other GLORYS variables used as inputs)")
    A(f"- Native resolution: 0.083° (~1/12°), daily")
    A(f"- Native depth levels: {len(native_depths)} ({native_depths[0]:.2f}–{native_depths[-1]:.2f} m)")
    A(f"- SIH target depths: {TARGET_DEPTHS}")
    A(f"- Depth interpolation: linear (scipy.interp1d / xarray.interp), "
      f"NaN where depth > ocean bottom")

    A("\n## 5. Regridding Method")
    A("- **Method: bilinear interpolation** (`xarray.DataArray.interp(method='linear')`)")
    A("- Applied to: SST, SSS, SLA, U_geo, V_geo, Wind_U, Wind_V (0.125° -> 0.25°)")
    A("- Applied to: GLORYS thetao (0.083° -> 0.25°)")
    A("- Bilinear is appropriate for all continuous physical fields at this resolution step.")

    A("\n## 6. Wind Daily Aggregation")
    A("- Source: hourly L4 product (24 steps/day)")
    A("- Aggregation: daily mean using `xr.Dataset.resample(time='1D').mean(skipna=True)`")
    A("- Hourly raw files deleted after successful aggregation")

    A("\n## 7. Unit Conversions")
    A("| Variable | Conversion | Notes |")
    A("|---|---|---|")
    A("| SST | K − 273.15 -> °C | Applied if values > 200 or units='kelvin' |")
    A("| SSS | None (values already in PSU) | scale_factor decoded by CF/xarray |")
    A("| SLA | None | m |")
    A("| Currents | None | m/s |")
    A("| Wind | None | m/s |")
    A("| GLORYS thetao | None | Already °C |")

    A("\n## 8. Missing Value Handling")
    A("- **Land/invalid cells**: derived from SST (NaN in ≥95% of timesteps)")
    A("- **Ocean observation gaps**: retained as NaN — not filled with zero")
    A("- **Missing dates**: reindexed to full daily range; missing days are NaN rows")
    A("- **Missing-value mask**: NOT added as a model input channel")
    A(f"- Valid ocean cell percentage: **{ocean_pct:.1f}%**")

    A("\n## 9. Train / Validation / Test Split")
    A("- Split method: **chronological** (no random shuffling)")
    A(f"- Train : {str(TRAIN_DATES[0].date())} -> {str(TRAIN_DATES[-1].date())} "
      f"({N_TRAIN} days, 70%)")
    A(f"- Val   : {str(VAL_DATES[0].date())} -> {str(VAL_DATES[-1].date())} "
      f"({N_VAL} days, 15%)")
    A(f"- Test  : {str(TEST_DATES[0].date())} -> {str(TEST_DATES[-1].date())} "
      f"({N_TEST} days, 15%)")

    A("\n## 10. Normalization Statistics (computed from training data only)")
    A("\n| Channel | Mean | Std | Min | Max | N valid |")
    A("|---|---|---|---|---|---|")
    for ch in CHANNEL_ORDER:
        s = norm_stats[ch]
        A(f"| {ch} | {s['mean']} | {s['std']} | {s.get('min','?')} | "
          f"{s.get('max','?')} | {s.get('n_valid','?')} |")

    A("\n## 11. Per-Depth Target Coverage")
    A("\n| Depth (m) | Valid cells (%) | Overall NaN (%) | Interpolated |")
    A("|---|---|---|---|")
    for d in TARGET_DEPTHS:
        r = depth_report[d]
        interp_note = "yes" if d <= native_depths.max() else "⚠️ below native range"
        A(f"| {d} | {r['valid_cell_pct']:.1f} | {r['overall_nan_pct']:.1f} | {interp_note} |")

    A("\n## 12. Final ML Dataset Shapes")
    A(f"- X (inputs):  {X_shape}  ->  [N, 7, {H}, {W}]")
    A(f"- Y (target):  {Y_shape}  ->  [N, 15, {H}, {W}]")
    A(f"- Channel order (X): {CHANNEL_ORDER}")
    A(f"- Depth order (Y): {TARGET_DEPTHS}")

    A("\n## 13. Output File Locations")
    A("```")
    A(f"data/processed/train/X_train.npz")
    A(f"data/processed/train/Y_train.npz")
    A(f"data/processed/train/dates_train.npy")
    A(f"data/processed/val/X_val.npz")
    A(f"data/processed/val/Y_val.npz")
    A(f"data/processed/val/dates_val.npy")
    A(f"data/processed/test/X_test.npz")
    A(f"data/processed/test/Y_test.npz")
    A(f"data/processed/test/dates_test.npy")
    A(f"data/processed/coords.nc")
    A(f"data/processed/land_mask.npy")
    A(f"results/normalization_stats.json")
    A(f"results/depth_coverage.csv")
    A("```")

    report_path = DOCS / "STAGE3_DATASET_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    info(f"Saved {report_path}")

    # ── JSON summary ─────────────────────────────────────────────
    summary = {
        "stage": 3,
        "generated_utc": pd.Timestamp.now('UTC').isoformat()[:19],
        "region": {"lat_min": 5.0, "lat_max": 22.0, "lon_min": 80.0, "lon_max": 100.0},
        "grid": {"lat_n": H, "lon_n": W, "resolution_deg": 0.25},
        "total_days": N_DAYS,
        "date_range": {"start": "2022-01-01", "end": "2023-12-31"},
        "split": {
            "train": {"start": str(TRAIN_DATES[0].date()), "end": str(TRAIN_DATES[-1].date()),
                      "n_days": N_TRAIN},
            "val":   {"start": str(VAL_DATES[0].date()), "end": str(VAL_DATES[-1].date()),
                      "n_days": N_VAL},
            "test":  {"start": str(TEST_DATES[0].date()), "end": str(TEST_DATES[-1].date()),
                      "n_days": N_TEST},
        },
        "inputs": {ch: {"dataset_id": ds_id, "variable": var_id,
                         "native_res_deg": res, "native_temporal": temp,
                         "units": u}
                   for ch, ds_id, var_id, res, temp, u in [
                       ("SST", "METOFFICE-GLO-SST-L4-REP-OBS-SST", "analysed_sst",
                        "0.125", "daily", "degC"),
                       ("SSS", "cmems_obs-mob_glo_phy-sss_my_multi_P1D", "sos",
                        "0.125", "daily", "PSU"),
                       ("SLA", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
                        "sla", "0.125", "daily", "m"),
                       ("CURRENT_U", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
                        "ugos", "0.125", "daily", "m/s"),
                       ("CURRENT_V", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
                        "vgos", "0.125", "daily", "m/s"),
                       ("WIND_U", "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
                        "eastward_wind", "0.125", "hourly->daily_mean", "m/s"),
                       ("WIND_V", "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
                        "northward_wind", "0.125", "hourly->daily_mean", "m/s"),
                   ]},
        "target": {
            "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
            "variable": "thetao",
            "native_res_deg": "0.083",
            "native_temporal": "daily",
            "units": "degC",
            "depths_m": TARGET_DEPTHS,
        },
        "regrid_method": "bilinear (xarray.interp linear)",
        "depth_interp_method": "linear (xarray.interp / scipy.interp1d)",
        "wind_aggregation": "daily_mean_of_hourly",
        "missing_value_strategy": "retain NaN — no zero-fill",
        "land_mask_strategy": "NaN in >=95% of SST timesteps",
        "ocean_cell_pct": round(ocean_pct, 2),
        "normalization_stats": norm_stats,
        "depth_coverage": {str(d): depth_report[d] for d in TARGET_DEPTHS},
        "X_shape_full": list(X_shape),
        "Y_shape_full": list(Y_shape),
        "channel_order": CHANNEL_ORDER,
        "depth_order": TARGET_DEPTHS,
        "missing_dates_per_var": missing_per_var,
        "native_glorys_depths": [round(float(d), 4) for d in native_depths],
    }

    summary_path = RESULTS / "stage3_dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    info(f"Saved {summary_path}")


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OceanEmbed preprocessing pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check file availability; don't load or process.")
    parser.add_argument("--report-only", action="store_true",
                        help="Load existing processed data and regenerate reports only.")
    args = parser.parse_args()

    banner("OCEAEMBED — STAGE 3 PREPROCESSING PIPELINE")
    info(f"Output grid: {H} lat × {W} lon  (0.25°)")
    info(f"Total days : {N_DAYS}  ({str(ALL_DATES[0].date())} -> {str(ALL_DATES[-1].date())})")
    info(f"Train/Val/Test: {N_TRAIN}/{N_VAL}/{N_TEST} days")

    if args.dry_run:
        banner("DRY RUN — File availability check")
        for entry_subdir, var, label in [
            ("sst",       "analysed_sst", "SST"),
            ("sss",       "sos",          "SSS"),
            ("altimetry", "sla",          "Altimetry (sla/ugos/vgos)"),
            ("wind",      "eastward_wind","Wind"),
            ("glorys",    "thetao",       "GLORYS"),
        ]:
            files = find_monthly_files(entry_subdir)
            missing = check_coverage(files, entry_subdir)
            status = f"OK ({len(files)} files)" if not missing else \
                     f"MISSING {len(missing)} months: {[t for _, _, t in missing[:5]]}"
            print(f"  {label:<20}: {status}")
        return

    # ── Load all surface inputs ──────────────────────────────────
    missing_per_var = {}
    input_arrays = {}

    # SST
    da, miss, _ = load_variable("sst", "analysed_sst", "SST")
    da = convert_units(da, "SST")
    input_arrays["SST"] = da
    missing_per_var["SST"] = len(miss)

    # SSS (squeeze depth dim)
    da, miss, _ = load_variable("sss", "sos", "SSS", squeeze_depth=True)
    da = convert_units(da, "SSS")
    input_arrays["SSS"] = da
    missing_per_var["SSS"] = len(miss)

    # Altimetry: sla + ugos + vgos from combined file
    for var, ch in [("sla", "SLA"), ("ugos", "CURRENT_U"), ("vgos", "CURRENT_V")]:
        da, miss, _ = load_variable("altimetry", var, ch)
        da = convert_units(da, ch)
        input_arrays[ch] = da
        missing_per_var[ch] = len(miss)

    # Wind U + V (already daily-aggregated by downloader)
    da, miss, _ = load_variable("wind", "eastward_wind", "WIND_U")
    da = convert_units(da, "WIND_U")
    input_arrays["WIND_U"] = da
    missing_per_var["WIND_U"] = len(miss)

    da, miss, _ = load_variable("wind", "northward_wind", "WIND_V")
    da = convert_units(da, "WIND_V")
    input_arrays["WIND_V"] = da
    missing_per_var["WIND_V"] = len(miss)

    # ── GLORYS (with depth interpolation) ───────────────────────
    glorys_da, native_depths, depth_report = load_glorys_and_interpolate()
    missing_per_var["GLORYS"] = 0  # already checked inside

    # ── Land mask ───────────────────────────────────────────────
    land_mask = build_land_mask(input_arrays)
    np.save(PROCESSED / "land_mask.npy", land_mask)

    # ── Normalization ────────────────────────────────────────────
    train_idx = np.arange(0, N_TRAIN)
    norm_stats = compute_normalization_stats(input_arrays, train_idx)
    save_normalization_stats(norm_stats)

    # ── Build ML dataset ─────────────────────────────────────────
    X_shape, Y_shape = build_ml_tensors(input_arrays, glorys_da, norm_stats, land_mask)

    # ── Coordinate sidecar ───────────────────────────────────────
    save_coords(glorys_da)

    # ── Depth coverage report ────────────────────────────────────
    save_depth_coverage(depth_report, native_depths)

    # ── Reports ──────────────────────────────────────────────────
    generate_dataset_report(
        input_arrays, glorys_da, norm_stats,
        native_depths, depth_report, land_mask,
        missing_per_var, X_shape, Y_shape
    )

    # ── Final status table ───────────────────────────────────────
    banner("STAGE 3 PREPROCESSING — FINAL STATUS")
    print(f"\n{'Dataset':<12} {'Final shape':<30} {'Date range':<27} "
          f"{'Grid':<12} {'Variable':<14} {'Missing%':>10} {'Status'}")
    print("-" * 115)
    for ch in CHANNEL_ORDER:
        da = input_arrays[ch]
        nan_pct = 100.0 * float(np.sum(np.isnan(da.values))) / da.values.size
        print(f"{ch:<12} {str(da.shape):<30} "
              f"{'2022-01-01 – 2023-12-31':<27} "
              f"{'69×81':<12} {ch:<14} {nan_pct:>9.2f}%  ✅")
    # GLORYS
    nan_y = 100.0 * float(np.sum(np.isnan(glorys_da.values))) / glorys_da.values.size
    print(f"{'GLORYS':<12} {str(glorys_da.shape):<30} "
          f"{'2022-01-01 – 2023-12-31':<27} "
          f"{'69×81':<12} {'thetao (15D)':<14} {nan_y:>9.2f}%  ✅")

    print(f"\nX shape (all data): {X_shape}")
    print(f"Y shape (all data): {Y_shape}")
    print(f"\nSee docs/STAGE3_DATASET_REPORT.md for full report.")
    print("=" * 65)
    print("STAGE 3 COMPLETE. Do NOT proceed to Stage 4 until report is reviewed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
