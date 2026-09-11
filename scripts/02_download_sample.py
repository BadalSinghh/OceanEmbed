"""
02_download_sample.py

Downloads ~7 daily timesteps (2022-01-01 to 2022-01-07) for the 7 observation inputs
and GLORYS target thetao into data/raw/test/
"""

import sys
import yaml
from pathlib import Path
import copernicusmarine

def load_config(config_path="config/dataset_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_samples():
    config = load_config()
    raw_test_dir = Path("data/raw/test")
    raw_test_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = config["sample_time"]["start_date"]
    end_date = config["sample_time"]["end_date"]
    lat_min, lat_max = config["region"]["lat_min"], config["region"]["lat_max"]
    lon_min, lon_max = config["region"]["lon_min"], config["region"]["lon_max"]
    depth_min, depth_max = 0.0, 1050.0

    print("=" * 70)
    print(f"STARTING SAMPLE DOWNLOAD ({start_date} to {end_date})")
    print(f"Domain: Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}]")
    print(f"Destination: {raw_test_dir.resolve()}")
    print("=" * 70)

    # Track download statuses
    results = {}

    # 1. Download Surface Observation Inputs
    inputs_cfg = config["inputs"]
    
    for key, info in inputs_cfg.items():
        dataset_id = info["dataset_id"]
        var_name = info["variable"]
        out_filename = f"sample_{key}.nc"
        out_path = raw_test_dir / out_filename
        
        print(f"\n[INFO] Subsetting {key.upper()} ({dataset_id}) variable '{var_name}'...")
        if out_path.exists():
            print(f"[SKIP] {out_filename} already exists locally. Skipping download.")
            results[key] = {"status": "SUCCESS (cached)", "file": out_filename, "var": var_name, "dataset_id": dataset_id}
            continue
        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=[var_name],
                minimum_latitude=lat_min, maximum_latitude=lat_max,
                minimum_longitude=lon_min, maximum_longitude=lon_max,
                start_datetime=start_date, end_datetime=end_date,
                output_directory=str(raw_test_dir), output_filename=out_filename
            )
            print(f"[SUCCESS] Saved sample to {out_path}")
            results[key] = {"status": "SUCCESS", "file": out_filename, "var": var_name, "dataset_id": dataset_id}
        except Exception as e:
            print(f"[ERROR] Failed downloading {key}: {e}")
            results[key] = {"status": "FAILED", "error": str(e), "var": var_name, "dataset_id": dataset_id}

    # 2. Download Target GLORYS thetao ONLY
    target_cfg = config["target"]["glorys_3d_temp"]
    target_dataset_id = target_cfg["dataset_id"]
    target_var_name = target_cfg["variable"]
    target_out_filename = "sample_glorys_thetao.nc"
    target_out_path = raw_test_dir / target_out_filename

    print(f"\n[INFO] Subsetting GLORYS TARGET ({target_dataset_id}) variable '{target_var_name}'...")
    if target_out_path.exists():
        print(f"[SKIP] {target_out_filename} already exists locally. Skipping download.")
        results["glorys_target"] = {"status": "SUCCESS (cached)", "file": target_out_filename, "var": target_var_name, "dataset_id": target_dataset_id}
    else:
        try:
            copernicusmarine.subset(
                dataset_id=target_dataset_id,
                variables=[target_var_name],
                minimum_latitude=lat_min, maximum_latitude=lat_max,
                minimum_longitude=lon_min, maximum_longitude=lon_max,
                minimum_depth=depth_min, maximum_depth=depth_max,
                start_datetime=start_date, end_datetime=end_date,
                output_directory=str(raw_test_dir), output_filename=target_out_filename
            )
            print(f"[SUCCESS] Saved GLORYS sample target to {target_out_path}")
            results["glorys_target"] = {"status": "SUCCESS", "file": target_out_filename, "var": target_var_name, "dataset_id": target_dataset_id}
        except Exception as e:
            print(f"[ERROR] Failed downloading GLORYS target: {e}")
            results["glorys_target"] = {"status": "FAILED", "error": str(e), "var": target_var_name, "dataset_id": target_dataset_id}

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY RESULTS:")
    for k, res in results.items():
        print(f" {k:<15}: {res['status']} | Dataset ID: {res['dataset_id']}")

if __name__ == "__main__":
    download_samples()
