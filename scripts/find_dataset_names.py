"""
find_dataset_names.py

Directly inspect dataset IDs and variable names for key Copernicus products.
"""

import copernicusmarine

product_list = [
    "SST_GLO_SST_L4_REP_OBSERVATIONS_010_011",
    "SEALEVEL_GLO_PHY_L4_MY_008_047",
    "MULTIOBS_GLO_PHY_SSS_L4_MY_015_015",
    "MULTIOBS_GLO_PHY_REP_015_002",
    "WIND_GLO_PHY_L4_MY_012_006",
    "GLOBAL_MULTIYEAR_PHY_001_030",
    "GLOBAL_REANALYSIS_PHY_001_030"
]

print("PRODUCT TO DATASET RESOLUTION:")
for pid in product_list:
    print("-" * 60)
    print(f"Querying Product: {pid}")
    try:
        cat = copernicusmarine.describe(product_id=pid)
        for prod in getattr(cat, "products", []):
            print(f"Product Title: {prod.title}")
            for ds in getattr(prod, "datasets", []):
                vars_list = [v.name for v in getattr(ds, "variables", [])]
                print(f"  [DATASET ID]: {ds.dataset_id}")
                print(f"    Variables: {vars_list}")
    except Exception as e:
        print(f"  Failed: {e}")
