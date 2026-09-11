"""
probe_sss_wind.py

Open downloaded SSS NC file directly to inspect what variables exist.
Probe wind dataset time range via small metadata query.
"""

import xarray as xr
import numpy as np
import copernicusmarine

# 1. Try opening SSS dataset directly to check what variables exist
print("=== Probing SSS dataset cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c ===")
print("Attempting xarray.open_dataset via copernicusmarine.open_dataset...")
try:
    ds_sss = copernicusmarine.open_dataset(
        dataset_id="cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c",
        minimum_latitude=5.0,
        maximum_latitude=22.0,
        minimum_longitude=80.0,
        maximum_longitude=100.0,
        start_datetime="2022-01-01",
        end_datetime="2022-01-07"
    )
    print(f"Variables: {list(ds_sss.data_vars)}")
    print(f"Coords: {list(ds_sss.coords)}")
    for v in ds_sss.data_vars:
        print(f"  {v}: {ds_sss[v].dims} units={ds_sss[v].attrs.get('units','N/A')} long_name={ds_sss[v].attrs.get('long_name','N/A')}")
    ds_sss.close()
except Exception as e:
    print(f"  Error: {e}")

# 2. Probe wind dataset available time range via copernicusmarine metadata
print("\n=== Probing Wind Dataset Time Range ===")
# The 'my' (multi-year) product only covers 1994-2009
# Search for an NRT or extended wind product
wind_products = ["WIND_GLO_PHY_L4_MY_012_006"]
for pid in wind_products:
    try:
        cat = copernicusmarine.describe(product_id=pid)
        for prod in getattr(cat, "products", []):
            for ds in getattr(prod, "datasets", []):
                print(f"\nDataset: {ds.dataset_id}")
                print(f"  Title: {ds.title if hasattr(ds,'title') else 'N/A'}")
                ti = getattr(ds, "start_datetime", "N/A")
                te = getattr(ds, "end_datetime", "N/A")
                print(f"  Time: {ti} -> {te}")
    except Exception as e:
        print(f"  Error for {pid}: {e}")

# 3. Check if there is a combined / NRT wind product covering 2022
print("\n=== Searching for alternative wind products covering 2022 ===")
try:
    cat_all = copernicusmarine.describe(product_id="WIND_GLO_PHY_L4_REP_012_005")
    for prod in getattr(cat_all, "products", []):
        print(f"Product: {prod.product_id} - {prod.title}")
        for ds in getattr(prod, "datasets", []):
            ti = getattr(ds, "start_datetime", "N/A")
            te = getattr(ds, "end_datetime", "N/A")
            print(f"  Dataset: {ds.dataset_id} | time {ti} -> {te}")
except Exception as e:
    print(f"WIND_GLO_PHY_L4_REP_012_005: {e}")

try:
    cat2 = copernicusmarine.describe(product_id="WIND_GLO_PHY_CLIMATE_L4_012_003")
    for prod in getattr(cat2, "products", []):
        print(f"Product: {prod.product_id} - {prod.title}")
        for ds in getattr(prod, "datasets", []):
            ti = getattr(ds, "start_datetime", "N/A")
            te = getattr(ds, "end_datetime", "N/A")
            print(f"  Dataset: {ds.dataset_id} | time {ti} -> {te}")
except Exception as e:
    print(f"WIND_GLO_PHY_CLIMATE_L4_012_003: {e}")
