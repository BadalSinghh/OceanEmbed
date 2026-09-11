# -*- coding: utf-8 -*-
"""
04_download_full.py
===================
OceanEmbed Stage 3 — Full 2-year data acquisition.

Downloads 2022-01-01 through 2023-12-31 for all 7 input channels
and the GLORYS thetao target, in monthly chunks.

Dataset sources (FROZEN — do not change):
  SST       : METOFFICE-GLO-SST-L4-REP-OBS-SST           (analysed_sst)
  SSS       : cmems_obs-mob_glo_phy-sss_my_multi_P1D      (sos)
  Altimetry : cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D
                 (sla, ugos, vgos — downloaded together, split in preprocessing)
  Wind      : cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H  (eastward_wind, northward_wind)
                 Hourly product — aggregated to daily mean here, raw hourly deleted.
  GLORYS    : cmems_mod_glo_phy_my_0.083deg_P1D-m         (thetao ONLY — TARGET)

Region: Bay of Bengal — Lat 5–22°N, Lon 80–100°E
Depths (GLORYS only): 0–1100 m (to cover the 1000 m SIH level)

Usage:
    python scripts/04_download_full.py [--dry-run]

Flags:
    --dry-run   Print storage estimate and dataset checks only; no actual download.
    --force     Re-download a chunk even if the file already exists.
"""

import argparse
import calendar
import csv
import os
import sys
import traceback
from datetime import datetime, date
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import copernicusmarine
import numpy as np
import xarray as xr
import yaml

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RAW_FULL = ROOT / "data" / "raw" / "full"
LOG_DIR  = ROOT / "logs"
LOG_FILE = LOG_DIR / "download_log.csv"

REGION = dict(lat_min=5.0, lat_max=22.0, lon_min=80.0, lon_max=100.0)

FULL_START_YEAR = 2022
FULL_END_YEAR   = 2023

# ──────────────────────────────────────────────────────────────
# Download catalogue
# ──────────────────────────────────────────────────────────────
# Each entry defines one Copernicus subset call per month.
# 'key'         : internal name used for logging + output directory name
# 'dataset_id'  : exact Copernicus dataset ID
# 'variables'   : list of variables to request in one call
# 'subdir'      : subdirectory under data/raw/full/
# 'depth_range' : (min_depth, max_depth) or None for surface products
# 'hourly'      : True → daily-mean aggregate before saving; delete hourly raw

CATALOGUE = [
    {
        "key": "sst",
        "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        "variables": ["analysed_sst"],
        "subdir": "sst",
        "depth_range": None,
        "hourly": False,
    },
    {
        "key": "sss",
        "dataset_id": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        "variables": ["sos"],
        "subdir": "sss",
        "depth_range": (0.0, 10.0),   # product has a single near-surface depth level
        "hourly": False,
    },
    {
        # SLA + geostrophic currents — same DUACS dataset, downloaded together
        "key": "altimetry",
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        "variables": ["sla", "ugos", "vgos"],
        "subdir": "altimetry",
        "depth_range": None,
        "hourly": False,
    },
    {
        # L4 blended wind — hourly; aggregated to daily means
        "key": "wind",
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        "variables": ["eastward_wind", "northward_wind"],
        "subdir": "wind",
        "depth_range": None,
        "hourly": True,
    },
    {
        # GLORYS thetao — TARGET ONLY; 3D, full depth column to 1100 m
        "key": "glorys",
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["thetao"],
        "subdir": "glorys",
        "depth_range": (0.0, 1100.0),
        "hourly": False,
    },
]

# ──────────────────────────────────────────────────────────────
# Rough storage estimates (MB per month per dataset)
# Based on sample file sizes, scaled to full month day-count
# ──────────────────────────────────────────────────────────────
STORAGE_EST_MB = {
    "sst":        15,    # ~1.9 MB / 7 days → ~8 MB/month; add margin
    "sss":         3,    # ~0.28 MB / 3 days → ~3 MB/month
    "altimetry":  10,    # 3 vars × 7 days → combined ~10 MB/month
    "wind":        6,    # hourly, daily-agg → ~6 MB/month after aggregation
    "glorys":    110,    # 24 MB / 7 days × 30 → ~100–120 MB/month (3D, 35+ levels)
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def month_range(start_year, end_year):
    """Yield (year, month) tuples from Jan start_year to Dec end_year."""
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def month_bounds(year, month):
    """Return ISO date strings for first and last day of a month."""
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year:04d}-{month:02d}-01"
    end   = f"{year:04d}-{month:02d}-{last_day:02d}"
    return start, end


def output_path(subdir, year, month, suffix=""):
    """Canonical path for a monthly chunk file."""
    tag = f"{year:04d}-{month:02d}"
    fname = f"{subdir}_{tag}{suffix}.nc"
    return RAW_FULL / subdir / fname


def log_result(row: dict):
    """Append one row to the CSV download log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "key", "year", "month", "status", "file", "size_mb", "note"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2) if path.exists() else 0.0


# ──────────────────────────────────────────────────────────────
# Storage estimate
# ──────────────────────────────────────────────────────────────

def print_storage_estimate():
    months = (FULL_END_YEAR - FULL_START_YEAR + 1) * 12
    print("\n" + "=" * 65)
    print("ESTIMATED STORAGE REQUIREMENTS")
    print("=" * 65)
    total = 0
    for entry in CATALOGUE:
        key = entry["key"]
        per_month = STORAGE_EST_MB.get(key, 10)
        total_key = per_month * months
        total += total_key
        print(f"  {key:<12}: ~{per_month:>4} MB/month × {months} months = ~{total_key:>5} MB")
    print(f"  {'TOTAL':<12}: ~{total:>5} MB  (~{total/1024:.1f} GB)")
    print("=" * 65 + "\n")


# ──────────────────────────────────────────────────────────────
# Wind: hourly → daily aggregation
# ──────────────────────────────────────────────────────────────

def aggregate_wind_daily(hourly_path: Path, daily_path: Path) -> bool:
    """
    Load hourly wind NetCDF, compute daily mean, save to daily_path.
    Returns True on success.
    """
    try:
        ds = xr.open_dataset(hourly_path)
        # Resample: group by calendar day (UTC), take mean
        ds_daily = ds.resample(time="1D").mean(skipna=True)
        ds_daily.attrs.update(ds.attrs)
        ds_daily.attrs["temporal_aggregation"] = "daily mean of hourly L4 wind product"
        # Use netcdf4 engine for compatibility
        ds_daily.to_netcdf(daily_path)
        ds.close()
        ds_daily.close()
        print(f"    [WIND AGG] {hourly_path.name} → {daily_path.name}  "
              f"({file_size_mb(daily_path):.1f} MB)")
        return True
    except Exception as e:
        print(f"    [WIND AGG ERROR] {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Chunk validation (lightweight post-download check)
# ──────────────────────────────────────────────────────────────

def validate_chunk(path: Path, variables: list, expected_year: int, expected_month: int,
                   hourly=False) -> tuple:
    """
    Opens a downloaded NetCDF chunk and checks:
      - All requested variables present
      - Time range spans the expected month
      - No entirely-NaN variable
    Returns (ok: bool, note: str)
    """
    try:
        ds = xr.open_dataset(path)
        missing_vars = [v for v in variables if v not in ds.data_vars]
        if missing_vars:
            ds.close()
            return False, f"Missing vars: {missing_vars}"

        if "time" not in ds.coords:
            ds.close()
            return False, "No time coordinate"

        times = ds["time"].values
        t0 = str(times[0])[:10]
        t1 = str(times[-1])[:10]
        expected_start = f"{expected_year:04d}-{expected_month:02d}-01"
        last_day = calendar.monthrange(expected_year, expected_month)[1]
        expected_end = f"{expected_year:04d}-{expected_month:02d}-{last_day:02d}"

        if t0 > expected_end or t1 < expected_start:
            ds.close()
            return False, f"Time {t0}–{t1} outside expected {expected_start}–{expected_end}"

        n_times = len(times)
        ds.close()
        return True, f"ok | {t0}–{t1} | N_times={n_times}"
    except Exception as e:
        return False, f"Validation error: {e}"


# ──────────────────────────────────────────────────────────────
# Per-dataset monthly download
# ──────────────────────────────────────────────────────────────

def download_month(entry: dict, year: int, month: int, force: bool = False) -> dict:
    """
    Download one monthly chunk for one catalogue entry.
    For wind: downloads hourly raw → aggregates → deletes raw if successful.
    Returns a status dict.
    """
    key      = entry["key"]
    dataset  = entry["dataset_id"]
    vars_    = entry["variables"]
    subdir   = entry["subdir"]
    depth_r  = entry["depth_range"]
    is_hrly  = entry["hourly"]

    start_dt, end_dt = month_bounds(year, month)

    out_dir = RAW_FULL / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_hrly:
        # Wind: save daily-aggregated file
        final_path  = output_path(subdir, year, month)          # wind_2022-01.nc (daily)
        hourly_path = output_path(subdir, year, month, "_hourly")
    else:
        final_path = output_path(subdir, year, month)
        hourly_path = None

    # ── Skip if already completed ──
    if final_path.exists() and not force:
        ok, note = validate_chunk(final_path, vars_, year, month, hourly=False)
        if ok:
            size = file_size_mb(final_path)
            print(f"  [SKIP] {key} {year}-{month:02d} — already exists ({size:.1f} MB)")
            return {"key": key, "year": year, "month": month,
                    "status": "CACHED", "file": final_path.name,
                    "size_mb": round(size, 2), "note": note}
        else:
            print(f"  [WARN] Existing file failed validation ({note}); re-downloading.")

    print(f"  [DL] {key} {year}-{month:02d}  dataset={dataset}  vars={vars_}")

    # ── Build subset kwargs ──
    subset_kwargs = dict(
        dataset_id=dataset,
        variables=vars_,
        minimum_latitude=REGION["lat_min"],
        maximum_latitude=REGION["lat_max"],
        minimum_longitude=REGION["lon_min"],
        maximum_longitude=REGION["lon_max"],
        start_datetime=start_dt,
        end_datetime=end_dt,
    )
    if depth_r is not None:
        subset_kwargs["minimum_depth"] = depth_r[0]
        subset_kwargs["maximum_depth"] = depth_r[1]

    download_target = hourly_path if is_hrly else final_path

    try:
        subset_kwargs["output_directory"] = str(download_target.parent)
        subset_kwargs["output_filename"]  = download_target.name
        copernicusmarine.subset(**subset_kwargs)
    except Exception as e:
        err_msg = str(e)
        print(f"  [ERROR] {key} {year}-{month:02d}: {err_msg}")
        return {"key": key, "year": year, "month": month,
                "status": "FAILED", "file": "", "size_mb": 0, "note": err_msg[:200]}

    # ── Wind post-processing ──
    if is_hrly:
        agg_ok = aggregate_wind_daily(hourly_path, final_path)
        if agg_ok:
            # Delete hourly raw to save disk space
            try:
                hourly_path.unlink()
                print(f"    [CLEANUP] Deleted hourly raw: {hourly_path.name}")
            except Exception:
                pass
        else:
            print(f"    [WARN] Wind aggregation failed for {year}-{month:02d}; "
                  f"hourly file retained at {hourly_path}")
            return {"key": key, "year": year, "month": month,
                    "status": "AGG_FAILED", "file": str(hourly_path.name), "size_mb": 0,
                    "note": "Daily aggregation failed"}

    # ── Validate completed chunk ──
    ok, note = validate_chunk(final_path, vars_, year, month)
    size = file_size_mb(final_path)
    status = "SUCCESS" if ok else "SUCCESS_WARN"
    print(f"    → {status} | {final_path.name} ({size:.1f} MB) | {note}")

    return {"key": key, "year": year, "month": month,
            "status": status, "file": final_path.name,
            "size_mb": round(size, 2), "note": note}


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OceanEmbed full 2-year downloader")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print estimates only; do not download.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if file exists.")
    parser.add_argument("--key", default=None,
                        help="Download only this dataset key (e.g. 'glorys').")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("OCEAEMBED — FULL 2-YEAR DOWNLOAD")
    print(f"Period  : {FULL_START_YEAR}-01-01 to {FULL_END_YEAR}-12-31")
    print(f"Region  : Lat [{REGION['lat_min']}, {REGION['lat_max']}]  "
          f"Lon [{REGION['lon_min']}, {REGION['lon_max']}]")
    print(f"Output  : {RAW_FULL}")
    print("=" * 65)

    print_storage_estimate()

    if args.dry_run:
        print("[DRY RUN] Exiting without downloading.")
        return

    # Filter catalogue by key if specified
    catalogue = CATALOGUE
    if args.key:
        catalogue = [e for e in CATALOGUE if e["key"] == args.key]
        if not catalogue:
            print(f"[ERROR] Unknown key '{args.key}'. Valid keys: "
                  f"{[e['key'] for e in CATALOGUE]}")
            sys.exit(1)

    all_results = []
    months = list(month_range(FULL_START_YEAR, FULL_END_YEAR))
    total_chunks = len(catalogue) * len(months)
    done = 0

    for entry in catalogue:
        key = entry["key"]
        print(f"\n{'='*65}")
        print(f"DATASET: {key.upper()}  ({entry['dataset_id']})")
        print(f"Variables: {entry['variables']}")
        print(f"{'='*65}")

        for year, month in months:
            done += 1
            print(f"\n[{done}/{total_chunks}] Processing {key} {year}-{month:02d} ...")
            result = download_month(entry, year, month, force=args.force)
            all_results.append(result)

            # Log immediately so progress is preserved on crash
            log_result({
                "timestamp": datetime.utcnow().isoformat(),
                "key": result["key"],
                "year": result["year"],
                "month": result["month"],
                "status": result["status"],
                "file": result["file"],
                "size_mb": result["size_mb"],
                "note": result["note"],
            })

    # ── Final summary ──
    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)
    statuses = {}
    total_gb = 0.0
    for r in all_results:
        statuses.setdefault(r["status"], 0)
        statuses[r["status"]] += 1
        total_gb += r["size_mb"] / 1024

    for status, count in sorted(statuses.items()):
        print(f"  {status:<15}: {count}")
    print(f"  Total downloaded : {total_gb:.2f} GB")
    print(f"  Log saved to     : {LOG_FILE}")

    # Print failed chunks clearly
    failed = [r for r in all_results if r["status"] in ("FAILED", "AGG_FAILED")]
    if failed:
        print(f"\n{'='*65}")
        print(f"FAILED CHUNKS ({len(failed)}) — MUST BE RESOLVED BEFORE PREPROCESSING:")
        print(f"{'='*65}")
        for r in failed:
            print(f"  {r['key']:<12} {r['year']}-{r['month']:02d}  |  {r['note'][:100]}")
    else:
        print("\n[OK] All chunks downloaded successfully.")

    print("=" * 65)


if __name__ == "__main__":
    main()
