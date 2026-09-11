# OceanEmbed Data Sources & Catalog Verification

## Overview

This document provides a comprehensive catalogue of the 7 surface satellite & multi-observation derived input datasets, 1 target ocean reanalysis dataset, and 1 independent in-situ validation dataset selected for the **OceanEmbed** prototype for Smart India Hackathon 2026 (Problem SIH26066).

---

## Strict Scientific Boundary Rules

> [!IMPORTANT]
> 1. **GLORYS target separation**: GLORYS (`GLOBAL_MULTIYEAR_PHY_001_030`) is **STRICTLY** used as the ground truth training target (`thetao` at 15 standard depths). GLORYS surface variables (SST, SSS, SSH, currents) are **NOT** used as inputs to prevent data leakage and artificial skill inflation.
> 2. **No ERA5 replacement**: Surface wind vectors are obtained from scatterometer observations (`WIND_GLO_PHY_L4_MY_012_006`), **NOT** ERA5 reanalysis.
> 3. **Accurate Terminology**: Observation-based products are accurately described as satellite & multi-observation derived products.

---

## Spatial & Temporal Domain Specification

| Parameter | Value |
|---|---|
| **Target Region** | Bay of Bengal |
| **Latitude Range** | 5.0° N to 22.0° N |
| **Longitude Range** | 80.0° E to 100.0° E |
| **Time Period** | 2022-01-01 to 2023-12-31 (730 daily timesteps) |
| **Target Grid Resolution** | 0.25° × 0.25° (~28 km) |
| **Grid Matrix Dimensions** | Lat: 69 points, Lon: 81 points |
| **Target Depths (15 SIH levels)** | `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters |

---

## Product Catalogue Summary

### 1. Training Target (Ground Truth)

* **Dataset Name**: CMEMS GLORYS12V1 Global Ocean Physics Reanalysis
* **CMEMS Product ID**: `GLOBAL_MULTIYEAR_PHY_001_030`
* **CMEMS Dataset ID**: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
* **Variable**: `thetao` (Sea Water Potential Temperature, °C)
* **Native Resolution**: 1/12° (~0.083°), 50 vertical levels
* **Access Method**: `copernicusmarine` Python API (`copernicusmarine.subset`)

---

### 2. Surface Inputs (7 Channels - Satellite & Multi-Observation Derived)

#### Channel 1: Sea Surface Temperature (SST)
* **Dataset Name**: OSTIA Global Ocean Foundation Sea Surface Temperature Reprocessed
* **CMEMS Product ID**: `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`
* **Dataset ID**: `METOFFICE-GLO-SST-L4-REP-OBS-SST`
* **Variable**: `analysed_sst` (Converted from Kelvin to Celsius)
* **Native Resolution**: 0.05° daily

#### Channel 2: Sea Surface Salinity (SSS)
* **Dataset Name**: Multi Observation Global Ocean Sea Surface Salinity and Sea Surface Density
* **CMEMS Product ID**: `MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013`
* **Dataset ID**: `cmems_obs-mob_glo_phy-sss_my_multi_P1D`
* **Variable**: `sos` (Sea Surface Salinity, PSU / pss-78)
* **Exact Description**: "daily observation-based satellite/in-situ SSS analysis" (combines satellite SSS observations with in-situ observations and satellite SST through optimal interpolation).
* **Native Resolution**: 0.125° daily

#### Channel 3: Sea Level Anomaly (SLA / SSH)
* **Dataset Name**: Global Ocean Gridded L4 Sea Surface Heights and Derived Variables Reprocessed
* **CMEMS Product ID**: `SEALEVEL_GLO_PHY_L4_MY_008_047`
* **Dataset ID**: `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`
* **Variable**: `sla` (Sea Level Anomaly, m)
* **Native Resolution**: 0.125° daily

#### Channels 4 & 5: Surface Geostrophic Velocity (U & V)
* **Dataset Name**: Global Ocean Gridded L4 Sea Surface Heights and Derived Variables Reprocessed
* **CMEMS Product ID**: `SEALEVEL_GLO_PHY_L4_MY_008_047`
* **Dataset ID**: `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`
* **Variables**: `ugos` (Eastward surface geostrophic current, m/s), `vgos` (Northward surface geostrophic current, m/s)
* **Exact Description**: "satellite-altimetry-derived geostrophic surface current components"
* **Native Resolution**: 0.125° daily

#### Channels 6 & 7: Surface Wind Vector (U & V)
* **Dataset Name**: Global Ocean Hourly Reprocessed Sea Surface Wind and Stress from Scatterometer and Model
* **CMEMS Product ID**: `WIND_GLO_PHY_L4_MY_012_006`
* **Dataset ID**: `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H`
* **Variables**: `eastward_wind` (Surface zonal wind velocity, m/s), `northward_wind` (Surface meridional wind velocity, m/s)
* **Exact Description**: "satellite-observation-informed L4 wind product, bias-corrected using scatterometer observations" (Note: Incorporates ERA5 model fields to construct hourly gap-free fields).
* **Native Resolution**: 0.125°, hourly cadence (PT1H, aggregated to daily mean, regridded to 0.25°)

---

### 3. Independent Validation Benchmark

#### In-Situ Argo Profiling Floats
* **Dataset Name**: Global Ocean In-Situ Near Real Time / Reprocessed Physical Observations
* **CMEMS Product ID**: `INSITU_GLO_PHY_SITU_OBS_MY_013_001`
* **Dataset ID**: `cmems_obs-ins_glo_phy-cur_my_argo_irr`
* **Variables**: `TEMP` (Temperature, °C), `PRES` (Pressure / Depth dbar/m), `LATITUDE`, `LONGITUDE`, `TIME`
* **Processing**: Filtered for 5°N–22°N, 80°E–100°E, 2022–2023. Linear / Akima cubic spline interpolation to 15 standard depths.
