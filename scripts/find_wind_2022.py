"""
find_wind_2022.py

Search for the correct scatterometer wind product covering 2022-2023.
The MY product (cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H) only covers 1994-2009.
"""

import copernicusmarine

# Try open_dataset on the MY product to confirm time range
print("=== Probing MY wind dataset actual time coverage ===")
try:
    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H",
        minimum_latitude=5.0,
        maximum_latitude=6.0,
        minimum_longitude=80.0,
        maximum_longitude=81.0,
    )
    times = ds["time"].values
    print(f"MY wind time range: {times[0]} to {times[-1]}")
    print(f"Variables: {list(ds.data_vars)}")
    ds.close()
except Exception as e:
    print(f"MY wind error: {e}")

# Now probe NRT/extended wind product IDs known from Copernicus
nrt_candidates = [
    ("WIND_GLO_PHY_NRT_L4_012_004", None),
    ("WIND_GLO_PHY_L4_NRT_012_004", None),
    ("WIND_GLO_PHY_NRT_012_006", None),
    ("WIND_GLO_PHY_L4_012_006", None),
    ("WIND_GLO_WIND_L4_MY_012_006", None),
    (None, "cmems_obs-wind_glo_phy_nrt_l4_0.25deg_PT1H"),
    (None, "cmems_obs-wind_glo_phy_l4_0.25deg_PT1H"),
]

print("\n=== Probing alternative wind dataset IDs ===")
for pid, did in nrt_candidates:
    try:
        if did:
            ds = copernicusmarine.open_dataset(
                dataset_id=did,
                minimum_latitude=5.0,
                maximum_latitude=6.0,
                minimum_longitude=80.0,
                maximum_longitude=81.0,
            )
            times = ds["time"].values
            print(f"  [OK] dataset_id={did}: {times[0]} to {times[-1]}")
            print(f"       Variables: {list(ds.data_vars)}")
            ds.close()
        elif pid:
            cat = copernicusmarine.describe(product_id=pid)
            for prod in getattr(cat, "products", []):
                print(f"  Product {pid}: {prod.title}")
                for ds in getattr(prod, "datasets", []):
                    print(f"    -> Dataset ID: {ds.dataset_id}")
    except Exception as e:
        label = did or pid
        print(f"  [FAIL] {label}: {str(e)[:120]}")
