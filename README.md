# CycloneDSNet

Code repository for the paper:

**CycloneDSNet: Probabilistic Downscaling of Tropical Cyclone Forecasts from the TianXing Large Weather Model for Intensity Calibration**

## Overview

CycloneDSNet is a probabilistic tropical-cyclone downscaling framework developed for improving high-resolution intensity representation from coarse-resolution forecasts produced by the TianXing large weather model.

The framework contains two stages:

- **CycloneDSNet-Deterministic**, which provides a stable high-resolution structural estimate;
- **CycloneDSNet-Diffusion**, which further refines the deterministic output to recover missing inner-core details and represent forecast uncertainty.

This repository contains the main code used for data preparation, model training, inference, evaluation, and figure generation in the paper.

## Repository structure

- `model/`: core model architectures
- `project/`: dataset construction, training scripts, and related utilities
- `preprocess/`: preprocessing scripts for forecast fields, WRF target extraction, and statistics generation
- `evaluate/`: inference, evaluation, and plotting scripts
- `data_file/`: example metadata files and normalization statistics

## Data availability

This repository does **not** redistribute the full datasets.

Publicly available data used in this work include:

- **ERA5 reanalysis** from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets)
- **CMA tropical cyclone best-track data** from [China Typhoon Online](https://tcdata.typhoon.org.cn/zjljsjj.html)

## Model weights

Pretrained model weights are not stored directly in this GitHub repository.

They are released separately through:

- **Hugging Face** for convenient download
- **Zenodo** for archived release versions

Please refer to the corresponding release page for download links and version information.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt