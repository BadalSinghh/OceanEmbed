"""
find_exact_ids.py

Searches Copernicus Marine catalogue for exact dataset IDs matching our required 7 inputs and GLORYS target.
"""

import copernicusmarine

cat = copernicusmarine.describe()
print(f"Total products in catalogue: {len(getattr(cat, 'products', []))}")

query_keywords = {
    "glorys_target": ["glorys", "reanalysis", "001_030", "001_024", "my_001"],
    "sst": ["sst", "ostia", "010_011", "010_001"],
    "sss": ["sss", "salinity", "015_015", "015_013"],
    "sea_level": ["duacs", "allsat", "sea_surface_height", "008_047"],
    "wind": ["wind", "scatterometer", "012_006", "012_004"]
}

for category, keywords in query_keywords.items():
    print("\n" + "=" * 70)
    print(f"CATEGORY: {category.upper()} (Keywords: {keywords})")
    print("=" * 70)
    
    for prod in getattr(cat, "products", []):
        text_to_search = f"{prod.product_id} {prod.title}".lower()
        if any(kw.lower() in text_to_search for kw in keywords):
            print(f"\nProduct ID: {prod.product_id}")
            print(f"Title: {prod.title}")
            for ds in getattr(prod, "datasets", []):
                vars_str = ", ".join([v.name for v in getattr(ds, "variables", [])])
                print(f"   ---> Dataset ID: {ds.dataset_id}")
                print(f"        Variables: {vars_str[:120]}")
