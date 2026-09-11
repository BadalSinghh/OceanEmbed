# -*- coding: utf-8 -*-
"""
scripts/audit_cora_files.py
============================
Audits the downloaded CORA NC files for the test period (2023-09-14 to 2023-12-31).
Reports: profile counts, Bay of Bengal profiles, QC-passed profiles, missing dates.
Does NOT run validation. Does NOT use synthetic data.
"""
import sys
from pathlib import Path
from datetime import date, timedelta
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import netCDF4 as nc

ROOT = Path(r"C:\Users\arush\OneDrive\Desktop\oceanembedPrototype")
CORA_DIR = ROOT / "data" / "raw" / "cora_full_download"

START = date(2023, 9, 14)
END   = date(2023, 12, 31)
BOB_LAT = (5.0, 22.0)
BOB_LON = (80.0, 100.0)

print("=" * 70)
print("CORA FILE AUDIT — Test Period 2023-09-14 to 2023-12-31")
print("=" * 70)

# ── 1. Enumerate all .nc files in test period ──────────────────────────────
all_nc = list(CORA_DIR.glob("**/*.nc"))
print(f"\nTotal .nc files in cora_full_download/: {len(all_nc)}")

test_files = []
non_test   = []
for f in all_nc:
    fname = f.name
    if "PR_PF" not in fname:
        non_test.append(fname)
        continue
    try:
        parts     = fname.split("_")
        date_str  = parts[2]
        file_date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        if START <= file_date <= END:
            test_files.append((file_date, f))
        else:
            non_test.append(fname)
    except Exception:
        non_test.append(fname)

test_files.sort(key=lambda x: x[0])
print(f"Test-period PR_PF files (2023-09-14 to 2023-12-31): {len(test_files)}")
print(f"Non-test / non-PR_PF files (ignored): {len(non_test)}")

# ── 2. Check coverage — missing dates ─────────────────────────────────────
expected_dates = set()
d = START
while d <= END:
    expected_dates.add(d)
    d += timedelta(days=1)
total_expected = len(expected_dates)

found_dates = set(fd for fd, _ in test_files)
missing_dates = sorted(expected_dates - found_dates)

print(f"\nExpected dates in test period: {total_expected}")
print(f"Dates with a downloaded file : {len(found_dates)}")
print(f"Missing dates                : {len(missing_dates)}")
if missing_dates:
    for md in missing_dates:
        print(f"  MISSING: {md}")
else:
    print("  (none — full coverage)")

# ── 3. Parse each test-period file ────────────────────────────────────────
print(f"\n{'='*70}")
print("Parsing {len(test_files)} test-period files...")
print(f"{'='*70}")

total_global_profiles  = 0
total_bob_profiles     = 0
total_qc_passed        = 0
unique_profile_ids     = []
bob_profile_dates      = {}   # date -> count
depth_level_counts     = []   # valid levels per QC-passed profile
parse_errors           = 0

for file_date, fpath in test_files:
    try:
        with nc.Dataset(fpath, "r") as ds:
            vkeys = list(ds.variables.keys())

            lat_name  = next((v for v in ["LATITUDE","lat","latitude"]  if v in vkeys), None)
            lon_name  = next((v for v in ["LONGITUDE","lon","longitude"] if v in vkeys), None)
            time_name = next((v for v in ["TIME","time"]                 if v in vkeys), None)
            temp_var  = next((v for v in vkeys if "TEMP" in v.upper() and "QC" not in v.upper()), None)
            pres_var  = next((v for v in vkeys if "PRES" in v.upper() and "QC" not in v.upper()), None)
            tqc_var   = next((v for v in vkeys if v.upper() == "TEMP_QC"), None)

            if not all([lat_name, lon_name, temp_var, pres_var]):
                parse_errors += 1
                continue

            lats = np.array(ds.variables[lat_name][:]).flatten()
            lons = np.array(ds.variables[lon_name][:]).flatten()
            n    = len(lats)
            total_global_profiles += n

            temp_data = ds.variables[temp_var][:]
            pres_data = ds.variables[pres_var][:]
            tqc_data  = ds.variables[tqc_var][:] if tqc_var else None

            for i in range(n):
                lat_v = float(lats[i])
                lon_v = float(lons[i])

                # Spatial filter
                if not (BOB_LAT[0] <= lat_v <= BOB_LAT[1] and BOB_LON[0] <= lon_v <= BOB_LON[1]):
                    continue
                total_bob_profiles += 1

                # Extract profile arrays
                if temp_data.ndim == 2:
                    p_temp = np.array(temp_data[i, :], dtype=float)
                    p_pres = np.array(pres_data[i, :], dtype=float)
                    p_qc   = np.array(tqc_data[i, :]) if tqc_data is not None else None
                else:
                    p_temp = np.array(temp_data, dtype=float)
                    p_pres = np.array(pres_data, dtype=float)
                    p_qc   = np.array(tqc_data)  if tqc_data is not None else None

                # Valid finite values
                valid = np.isfinite(p_temp) & np.isfinite(p_pres) & (p_pres >= 0)

                # QC filter
                if p_qc is not None:
                    qc_arr = np.array(p_qc)
                    if qc_arr.dtype.kind in ['S', 'U', 'O']:
                        qc_ok = np.isin(qc_arr, [b'1', b'2', '1', '2'])
                    else:
                        qc_ok = np.isin(qc_arr.astype(int) if qc_arr.dtype.kind not in ['S','U','O'] else qc_arr, [1, 2, 49, 50])
                    valid = valid & qc_ok

                n_valid = int(valid.sum())
                if n_valid < 3:
                    continue

                total_qc_passed += 1
                pid = f"{fpath.stem}_{i:04d}"
                unique_profile_ids.append(pid)
                depth_level_counts.append(n_valid)

                ds_str = str(file_date)
                bob_profile_dates[ds_str] = bob_profile_dates.get(ds_str, 0) + 1

    except Exception as e:
        parse_errors += 1

# ── 4. Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("AUDIT RESULTS")
print(f"{'='*70}")
print(f"Files parsed                         : {len(test_files)}")
print(f"Parse errors                         : {parse_errors}")
print(f"Total global profiles across files   : {total_global_profiles:,}")
print(f"Profiles inside Bay of Bengal        : {total_bob_profiles:,}")
print(f"BoB profiles passing QC (>=3 levels) : {total_qc_passed:,}")
print(f"Unique QC-passed profile IDs         : {len(unique_profile_ids)}")
if depth_level_counts:
    print(f"Valid depth levels per profile       : min={min(depth_level_counts)}, max={max(depth_level_counts)}, mean={sum(depth_level_counts)/len(depth_level_counts):.1f}")
print(f"\nDates with >=1 QC-passed BoB profile : {len(bob_profile_dates)}")
print(f"Dates with 0 QC-passed BoB profiles  : {total_expected - len(bob_profile_dates)}")

print(f"\n{'='*70}")
print("Profiles per date (dates with BoB observations):")
print(f"{'='*70}")
for d_str in sorted(bob_profile_dates.keys()):
    count = bob_profile_dates[d_str]
    bar   = "#" * min(count, 20)
    print(f"  {d_str}: {count:3d} profiles  {bar}")

print(f"\n{'='*70}")
print("AUDIT COMPLETE — awaiting user approval before running validation.")
print(f"{'='*70}")
