"""
copernicus_downloader.py

Automated fetch utility for Copernicus Marine observation products and GLORYS target.
Strictly separates observation inputs from GLORYS ground-truth targets.
"""

from pathlib import Path
import yaml

class CopernicusDownloader:
    """
    Downloader interface wrapping `copernicusmarine.subset` for OceanEmbed datasets.
    """
    def __init__(self, config_path="config/dataset_config.yaml", output_dir="data/raw"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.region = self.config["region"]
        self.time = self.config["time"]

    def fetch_dataset_metadata(self, dataset_id: str):
        import copernicusmarine
        return copernicusmarine.describe(dataset_id=dataset_id)

    def download_observation_variable(self, var_key: str, output_filename: str):
        import copernicusmarine
        var_config = self.config["inputs"][var_key]
        dataset_id = var_config["dataset_id"]
        variable = var_config["variable"]

        output_path = self.output_dir / output_filename
        print(f"[INFO] Requesting Copernicus subset for {var_key} ({dataset_id})...")

        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=[variable],
            minimum_latitude=self.region["lat_min"],
            maximum_latitude=self.region["lat_max"],
            minimum_longitude=self.region["lon_min"],
            maximum_longitude=self.region["lon_max"],
            start_datetime=self.time["start_date"],
            end_datetime=self.time["end_date"],
            output_filename=str(output_path),
            force_download=True
        )
        return output_path

    def download_glorys_target(self, output_filename="glorys_thetao_2022_2023.nc"):
        import copernicusmarine
        target_config = self.config["target"]["glorys_3d_temp"]
        dataset_id = target_config["dataset_id"]
        variable = target_config["variable"]
        depths = self.config["target_depths"]

        output_path = self.output_dir / output_filename
        print(f"[INFO] Requesting GLORYS 3D target subset ({dataset_id})...")

        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=[variable],
            minimum_latitude=self.region["lat_min"],
            maximum_latitude=self.region["lat_max"],
            minimum_longitude=self.region["lon_min"],
            maximum_longitude=self.region["lon_max"],
            minimum_depth=min(depths),
            maximum_depth=max(depths),
            start_datetime=self.time["start_date"],
            end_datetime=self.time["end_date"],
            output_filename=str(output_path),
            force_download=True
        )
        return output_path
