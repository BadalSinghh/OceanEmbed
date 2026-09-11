"""
inspect_sss_wind.py

Inspect the actual variable names available in SSS and Wind datasets,
and find the correct wind dataset that covers 2022.
"""

import copernicusmarine

# --- SSS ---
print("=== SSS PRODUCT ===")
cat = copernicusmarine.describe(product_id="MULTIOBS_GLO_PHY_SSS_L4_MY_015_015")
for prod in getattr(cat, "products", []):
    for ds in getattr(prod, "datasets", []):
        print(f"Dataset ID: {ds.dataset_id}")
        for v in getattr(ds, "variables", []):
            print(f"  var: {v.name}")

# --- WIND - look for a product covering 2022 ---
print("\n=== WIND PRODUCT ===")
cat2 = copernicusmarine.describe(product_id="WIND_GLO_PHY_L4_MY_012_006")
for prod in getattr(cat2, "products", []):
    print(f"Product: {prod.title}")
    for ds in getattr(prod, "datasets", []):
        print(f"  Dataset ID: {ds.dataset_id}")
        for v in getattr(ds, "variables", []):
            print(f"    var: {v.name}")
        # Check temporal coverage
        try:
            ta = getattr(ds, "start_datetime", "N/A")
            tb = getattr(ds, "end_datetime", "N/A")
            print(f"    time: {ta} to {tb}")
        except Exception:
            pass
