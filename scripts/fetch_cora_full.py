# -*- coding: utf-8 -*-
import sys, shutil
from pathlib import Path
from datetime import date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import copernicusmarine

ROOT = Path(__file__).resolve().parent.parent
ARGO_RAW = ROOT / "data" / "raw" / "argo"
ARGO_RAW.mkdir(parents=True, exist_ok=True)

DATASET_ID = "cmems_obs-ins_glo_phy-temp-sal_my_cora_irr"
START_DATE = date(2023, 9, 14)
END_DATE   = date(2023, 12, 31)
TEMP_DIR   = ROOT / "data" / "raw" / "cora_full_download"

print("=" * 70, flush=True)
print("FULL CORA DOWNLOAD -- Independent Argo Validation", flush=True)
print(f"Period: {START_DATE} to {END_DATE}", flush=True)
print("=" * 70, flush=True)

# Copy 5-day test batch already downloaded
test_batch = ROOT / "data" / "raw" / "cora_test_batch"
tb_copied = 0
for nc_file in test_batch.glob("**/*.nc"):
    dest = ARGO_RAW / nc_file.name
    if not dest.exists():
        shutil.copy2(nc_file, dest)
        tb_copied += 1
print(f"Copied {tb_copied} test_batch files to argo/", flush=True)

print("\nDownloading full-year 2023 CORA profiling float files...", flush=True)

try:
    result = copernicusmarine.get(
        dataset_id=DATASET_ID,
        filter="*2023*PR_PF*.nc",
        output_directory=str(TEMP_DIR),
        no_directories=True,
        overwrite=False,
        skip_existing=True,
    )
    print(f"Download complete. Files in temp dir: {TEMP_DIR}", flush=True)
    downloaded = True
except Exception as e:
    print(f"ERROR downloading: {e}", flush=True)
    downloaded = False

if downloaded:
    all_nc = list(TEMP_DIR.glob("*.nc")) + list(TEMP_DIR.glob("**/*.nc"))
    print(f"Found {len(all_nc)} NC files in temp dir.", flush=True)
    copied = 0
    for nc_file in all_nc:
        fname = nc_file.name
        try:
            parts = fname.split("_")
            date_part = parts[2]  # CO_DMQCGL01_YYYYMMDD_PR_PF.nc
            file_date = date(int(date_part[:4]), int(date_part[4:6]), int(date_part[6:8]))
            if START_DATE <= file_date <= END_DATE:
                dest = ARGO_RAW / fname
                if not dest.exists():
                    shutil.copy2(nc_file, dest)
                    copied += 1
        except Exception:
            dest = ARGO_RAW / fname
            if not dest.exists():
                shutil.copy2(nc_file, dest)
                copied += 1
    print(f"Copied {copied} test-period files to data/raw/argo/", flush=True)

final_count = len(list(ARGO_RAW.glob("*.nc")))
print(f"\nTotal .nc files in data/raw/argo/: {final_count}", flush=True)
print("DONE. Ready to run 13_process_argo_and_validate.py", flush=True)
