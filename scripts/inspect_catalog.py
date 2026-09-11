"""
inspect_catalog.py

Queries copernicusmarine catalogue for product and dataset structures
"""

import copernicusmarine

print("Listing datasets for products...")

pids = [
    "SST_GLO_SST_L4_REP_OBSERVATIONS_010_011",
    "SEALEVEL_GLO_PHY_L4_MY_008_047",
    "MULTIOBS_GLO_PHY_SITU_MULTISAT_REP_015_013",
    "WIND_GLO_PHY_L4_MY_012_006",
    "GLOBAL_REANALYSIS_PHY_001_030"
]

for pid in pids:
    print("\n" + "=" * 60)
    print(f"PRODUCT: {pid}")
    try:
        cat = copernicusmarine.describe(product_id=pid)
        for ds in getattr(cat, "products", []):
            print(f"Product title: {ds.title} | ID: {ds.product_id}")
            for dataset in getattr(ds, "datasets", []):
                print(f"   ---> Dataset ID: {dataset.dataset_id}")
                for v in getattr(dataset, "variables", []):
                    print(f"         * Var name: {v.name}")
    except Exception as e:
        print(f"  Error describing product {pid}: {e}")

print("\nSearching dataset IDs directly...")
try:
    cat_all = copernicusmarine.describe()
    for prod in getattr(cat_all, "products", []):
        for ds in getattr(prod, "datasets", []):
            did = ds.dataset_id
            if any(k in did.lower() for k in ["sst", "sss", "sea_surface_height", "wind", "reanalysis-3d-so"]):
                print(f"Match Dataset ID: {did}")
except Exception as e:
    print(f"Error searching catalogue: {e}")
