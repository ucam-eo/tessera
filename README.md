<p align="center">
  <a href="https://github.com/FrankFeng-23/tessera-v2-animation">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FrankFeng-23/tessera-v2-animation/master/out/tessera-v2-hero-lite.gif">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FrankFeng-23/tessera-v2-animation/master/out/tessera-v2-hero-lite.gif">
      <img src="https://raw.githubusercontent.com/FrankFeng-23/tessera-v2-animation/master/out/tessera-v2-hero-lite.gif"
           alt="TESSERA v2 — pixel-wise Earth foundation model" width="100%">
    </picture>
  </a>
</p>

> [!TIP]
> **Want TESSERA v2 embeddings without running inference yourself?**
> [**Submit a v2 Embedding Pre-Request**](https://github.com/ucam-eo/geotessera/issues/new?template=v2-embedding-prerequest.yml&labels=v2-embedding-prerequest)
> to reserve your region and join the **early testers** — we'll prioritize your area.
> v2 coverage is still rolling out, so if you need embeddings right now, request
> [v1.1 embeddings](https://github.com/ucam-eo/geotessera#request-missing-embeddings) instead.

# Temporal Embeddings of Surface Spectra for Earth Representation and Analysis (TESSERA) [CVPR2026]
<div align="center">
    <a href="#readme-top">
        <img src="images/banner.png" alt="Banner">
    </a>
    <br />
    <p align="center">
        <a href="https://geotessera.org/">Project Website 🌐</a> &nbsp;&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/FrankFeng-23/btfm_project/issues">Report Bugs 🛠️</a> &nbsp;&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/FrankFeng-23/btfm_project/issues">Request Features 💡</a>
    </p>
</div>

<!--  ![Version](https://img.shields.io/badge/version-alpha-red) -->
![PyPI version](https://img.shields.io/pypi/v/geotessera?label=PyPI%20version&color=blue)
![License](https://img.shields.io/badge/License-MIT-blue.svg)


# Table of Contents

  - Learning about TESSERA
      - [Introduction](#introduction)
      - [Papers](#Papers)
      - [Podcast](https://www.satellite-image-deep-learning.com/p/tessera-a-temporal-foundation-model)
      - [Presentations](#presentations)
      - [License](#License)
  - Using TESSERA
      - [Acceptable Use Policy](#AUP)
      - [Accessing Precomputed Embeddings](#global-embeddings-access)
      - [Creating Your Own Embeddings](#creating-your-own-embeddings)
          - [End-to-end runbook](#end-to-end-runbook)
      - [Inference](#inference)
          - [TESSERA v2 (recommended)](#tessera-v2-recommended)
          - [TESSERA v1.1](#tessera-v11-qat-int8)
          - [TESSERA v1.0 (QAT)](#tessera-v10-qat-int8)
          - [TESSERA v1.0 (early)](#tessera-v10-early-float32)
      - [Downstream Tasks](#downstream-tasks)
      - [TESSERA Users Group](#tessera-users-group)
  - Additional information
      - [Team](#team)
      - [Contact](#contact)
      - [Citation](#citation)
      - [Acknowledgments](#acknowledgments)
      - [Star History](#star-history)

# Learning about TESSERA
## Introduction

Satellite remote sensing enables a wide range of downstream applications, including habitat mapping, carbon accounting, and strategies for conservation and sustainable land use. However, satellite time series are voluminous and often cloud-corrupted, making them challenging to use: the scientific community's ability to extract actionable insights is often constrained by the scarcity of labelled training datasets and the computational burden of processing temporal data. The key insight behind our work, due to [Dr. Clement Atzberger](https://www.linkedin.com/in/clement-atzberger-8abb8065/) is that forcing auto-encoder embeddings derived from two cloud-free random samples of satellite time series to align using [Barlow Twins](https://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf) results in an embedding that represents the entire time series, including the missing observations.

This idea is the key behind TESSERA, an open foundation model that preserves per-pixel spectral-temporal signals in 128-dimensional latent representations at 10-meter resolution globally. It uses self-supervised learning to summarise petabytes of Earth observation data. We compare our work with state-of-the-art task-specific models and other foundation models in five diverse downstream tasks and find that TESSERA closely matches or outperforms these baselines. By preserving temporal phenological signals that are typically lost in conventional approaches, TESSERA enables new insights into ecosystem dynamics, agricultural food systems, and environmental change detection. Moreover, our open-source implementation supports reproducibility and extensibility, while the privacy-preserving design allows researchers to maintain data sovereignty.

To our knowledge, TESSERA is unprecedented in its ease of use, scale, and accuracy: no other foundation model provides analysis-ready outputs, is open, and provides global, annual coverage at 10m resolution using only spectral-temporal features at pixel level.

Here are some visualization results of the TESSERA representation map (using the first three channels as RGB):

![repr_demo](images/repr_demo.png)

## Papers
Here are publications and preprints related to TESSERA, listed chronologically:
* Lisaius, M. C., Blake, A., Keshav, S., & Atzberger, C. (2024). Using Barlow Twins to Create Representations From Cloud-Corrupted Remote Sensing Time Series. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 17, 13162–13168. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. https://doi.org/10.1109/JSTARS.2024.3426044

* Z. Feng, C. Atzberger, S. Jaffer, J. Knezevic, S. Sormunen, R. Young, M.C. Lisaius, M. Immitzer, T. Jackson, J. Ball, D.A. Coomes, A. Madhavapeddy, A. Blake, S. Keshav (2025), [TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis](https://arxiv.org/abs/2506.20380), To Appear, CVPR 2026. ArXiv reprint. https://arxiv.org/abs/2506.20380

* Lisaius, M. C., Blake, A., Atzberger, C., & Keshav, S. (2026). Towards improved crop type classification: A compact embedding approach suitable for small fields. Accepted in Proceedings of the ISPRS Conference 2026. International Society for Photogrammetry and Remote Sensing.

* Z. Feng, C. Atzberger, S. Jaffer, J. Knezevic, S. Sormunen, R. Young, M.C. Lisaius, M. Immitzer, T. Jackson, J. Ball, D.A. Coomes, A. Madhavapeddy, A. Blake, S. Keshav, (2026) [Applications of the TESSERA Geospatial Foundation Model to Diverse Environmental Mapping Tasks](http://ssrn.com/abstract=6142416), SSRN preprint. http://ssrn.com/abstract=6142416
  
* Young, R., & Keshav, S. (2026). Interpolation of GEDI Biomass Estimates with Calibrated Uncertainty Quantification, arXiv preprint. https://doi.org/10.48550/ArXiv.2601.16834
  
* Lisaius, M. C., Keshav, S., Blake, A., & Atzberger, C. (2026). Embedding-based Crop Type Classification in the Groundnut Basin of Senegal (arXiv:2601.16900). ArXiv preprint. https://doi.org/10.48550/arXiv.2601.16900

* Ball, J.G.C, Wicklein J.A. , Feng, Z.,  Knezevic, J.,  Jaffer, S., Atzberger, C.,  Dalponte, M., and Coomes, D. Geospatial foundation models enable data-efficient tree species mapping in temperate montane forests, BioArxiv, https://doi.org/10.64898/2026.02.23.707022

* Z. Feng, S. Jaffer, I. Shokar, J. Knezevic, M. Elvers, C. Atzberger, R. Young, A. Naik, N. Robinson, A. Blake, D. Coomes, A. Madhavapeddy, S. Keshav (2026), [TESSERA v2: Scaling Pixel-wise Earth Foundation Models](https://arxiv.org/abs/2607.03949), arXiv preprint. https://arxiv.org/abs/2607.03949

## Presentations

* [TESSERA overview in AI for Good seminar](https://www.youtube.com/live/9yrpwFrwbGY), Frank Feng, Jan 22, 2026
* [TESSERA: Precomputed FAIR Global Pixel Embeddings for Earth Representation and Analysis](https://www.grss-ieee.org/event/tessera-precomputed-fair-global-pixel-embeddings-for-earth-representation-and-analysis/) IEEE GRSS Talk, Frank Feng, 12 December, 2025
* [2-slide summary (PPTX)](https://www.dropbox.com/scl/fi/zjo4trov0z2qnmdeitng0/CRI-2slide.pptx?rlkey=5kkojiknt6hdn2zplzlotqnbt&st=ezafh67n&dl=0) for CRI Flash Talks, S. Keshav, October 7, 2025
* Foundation model overview (PPTX) for Ecology Groups meeting, University of Cambridge, DAB, James Ball, October 6, 2025
* [TESSERA overview presentation with a focus on ecological applications](https://www.dropbox.com/scl/fi/8xvanw3kk586lp1ld31kd/maryland_talk_slides.pdf?rlkey=osyhtk1kc2pcj81iel0u32lub&st=6kedpwv6&dl=0) (PDF) University of Maryland, Frank Feng, October 1, 2025
* [TESSERA overview presentation](https://www.dropbox.com/scl/fi/0rsq4wkao3c7fgwljd8ec/JCU-tesserav2.pptx?rlkey=ccutcxgwi068c09n20t1yi549&st=13if23b3&dl=0) (PPTX) James Cook University, S. Keshav, September 29, 2025
* [TESSERA overview presentation](https://www.dropbox.com/scl/fi/1p7nabvlvie8fzyomkx7w/dab_talk_slides.pdf?rlkey=ym3d44o80mbrdkasyzct9kzi5&st=ozvwczs7&dl=0) University of Cambridge, DAB, Frank Feng, May 20, 2025
* [Self-supervised learning for earth observation](https://www.dropbox.com/scl/fi/zjo4trov0z2qnmdeitng0/CRI-2slide.pptx?rlkey=5kkojiknt6hdn2zplzlotqnbt&st=ezafh67n&dl=0) (PPTX) S. Keshav, Exeter, April 2025

## License

TESSERA software is released under the standard MIT license. Embeddings and model weights are released under the [CC0](https://creativecommons.org/publicdomain/zero/1.0/) license: essentially, 
they can be freely used for both commercial and non-commercial purposes. Although we do not legally require attribution,
we do request it.

# Using TESSERA

<a id="global-embeddings-access"></a>

## Accessing Embeddings using GeoTessera (recommended)

We have generated embeddings for the whole globe at 10m resolution for 2024.
Additionally, for regions such as the United States and Europe, embeddings are available from 2017 to 2025.
These can be downloaded and used for downstream applications, saving significant computational time and resources, using 
the [GeoTessera](https://github.com/ucam-eo/geotessera) library. 
We will progressively extending coverage backwards year by year until 2017.
If you find that your region does not have embeddings for a specific year, you can go to [here](https://github.com/ucam-eo/geotessera#request-missing-embeddings) to submit an embedding request.
The current coverage map is below:

<img src="https://github.com/ucam-eo/tessera-coverage-map/blob/main/map.png"> 

## TESSERA Users Group

Interested users are invited to join our [Zulip](https://eeg.zulipchat.com/login/) discussion groups.


# Creating Your Own Embeddings

If you would like to use our software to create your own embeddings, please follow the instructions below. Note that this is a comptuationally challenging task and you will need access to significant computational and storage resources. 

## Hardware Requirements

### 1. Storage Requirements

Running this pipeline requires substantial storage space. Although the pipeline cleans up some intermediate files after processing, the downloaded raw Sentinel-2 and Sentinel-1 files will still occupy considerable disk space. For example, processing a 100km×100km area from 2022 to output a TESSERA Representation map (10m resolution) requires at least 1TB of storage.

### 2. Memory Requirements

We use preprocessed data, initially from Microsoft Planetary Computer. However, the next generation of embeddings will use OPERA from ASF DAAC. In either case, most of the geo-preprocessing has been done. Still, we recommend having at least 128GB of RAM.

### 3. CPU and GPU

The pipeline has no strict requirements for CPU and GPU, but more CPU cores and more powerful GPUs can significantly speed up inference. When processing a 110km×110km area from 2022, our tests using a 128-core CPU and a single NVIDIA A30 GPU for inference (CPU and GPU each handling 50% of the inference) took approximately 10 hours to complete.

### 4. Operating System

For the data preprocessing pipeline, we support almost all Linux systems. For Windows, we recommend using WSL. We do not support MacOS at this point.

For the model inference part, we have only tested it on Linux and Windows WSL, and they are working.

## End-to-end runbook

The fastest path from a region of interest to a TESSERA v2 representation map. Every step runs from the **repository root** — there is no need to `cd` anywhere or edit any script: all paths and tuning knobs are passed as environment variables or CLI flags. Each step writes into a numbered folder under your data directory so the outputs stay ordered (`0.roi/`, `1.data_raw/`, …, `5.result/`).

> The [Data Preprocessing](#data-preprocessing) and [Inference](#inference) sections below remain the detailed reference for each stage and the other model versions. This runbook is the complete, streamlined chain.

### Setup environment

The pipeline needs a Python environment with the base repo dependencies (geoprocessing), PyTorch, and the v2 package + weights. Run this once — it `cd`s into `tessera_infer_v2` and `cd ..` back to the repo root afterwards, so the steps below still run from the root:

```bash
#  create + activate a virtualenv — a conda env works too
python3 -m venv venv
source venv/bin/activate

# base repo dependencies (geoprocessing: rasterio, fiona, …)
pip install -r requirements.txt

# PyTorch — match your CUDA version (see https://pytorch.org/)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# v2 inference deps + the default Medium checkpoint (~84 MB)
cd tessera_infer_v2
pip install -r requirements.txt
python download_weights.py --model medium
cd ..
```

> [!NOTE]
> If the base `pip install -r requirements.txt` fails while building **fiona** with a `gdal-config not found` (or similar GDAL) error, install the system GDAL libraries and re-run that line:
> ```bash
> sudo apt install gdal-bin libgdal-dev
> ```

This fetches the recommended **Medium** student into `tessera_infer_v2/student/checkpoints/`. For the other students (Nano/Small/Large), the 2B teacher, or direct Hugging Face access, see [TESSERA v2 (recommended)](#tessera-v2-recommended).

**Run this once in your terminal** — the variables are reused by every step. Keep the same shell session so they persist (which also makes retrying a step a single re-paste):

```bash
DATA_DIR=/absolute/path/to/your/data_dir              # all outputs are written under here
ROI_SHP=/absolute/path/to/your/roi.shp
PYTHON_ENV=/absolute/path/to/python_env/bin/python    # run `which python` when you have activated venv
BASENAME=myregion                                     # final result file name (.npy and .tif)
YEAR=2025                                             # data year, range [2017-2025]
```

### Step 0 — Prepare ROI GeoTIFF  → `0.roi/`

```bash
ROI_TIFF="${DATA_DIR}/0.roi/roi.tiff"     # ROI extent: downloaded over + used as geo-reference
mkdir -p "${DATA_DIR}/0.roi"
python tessera_preprocessing/convert_shp_to_tiff.py \
    --shp_path   "${ROI_SHP}" \
    --tiff_path  "${ROI_TIFF}" \
    --pixel_size 10
```

Commands above writes `roi.tiff` and `roi_convex_hull.tiff` to `0.roi/`. `ROI_TIFF` above points at the Geotiff file — that is the extent the next step downloads over.

- `--shp_path` — your input shapefile.
- `--pixel_size` — metres per pixel (optional, default `10`).
- `--tiff_path` — custom output name (optional).
- `--force_crs EPSG:32650` — force a target CRS instead of the auto-detected best UTM zone (optional).

### Step 1 — Download Sentinel-1 & Sentinel-2 → `1.data_sar_raw/`, `1.data_raw/`

The network-heavy step and **the one most likely to fail partway**. It is safe to re-run: with `S1_OVERWRITE=false` / `S2_OVERWRITE=false` (set below), each observation day whose per-date output already exists *and* validates is skipped, so re-pasting the block fetches only the missing days — completed work is never re-downloaded.

```bash
INPUT_TIFF="${ROI_TIFF}" \
OUT_DIR="${DATA_DIR}" \
TEMP_DIR="${DATA_DIR}/tmp" \
PYTHON_ENV="${PYTHON_ENV}" \
YEAR="${YEAR}" \
DATA_SOURCE=mpc \
S1_RAW_SUBDIR=1.data_sar_raw \
S2_RAW_SUBDIR=1.data_raw \
S1_OVERWRITE=false \
S2_OVERWRITE=false \
    bash tessera_preprocessing/s1_s2_downloader.sh
```

Outputs: S1 → `${DATA_DIR}/1.data_sar_raw`, S2 → `${DATA_DIR}/1.data_raw`.

- `INPUT_TIFF` / `OUT_DIR` / `TEMP_DIR` / `PYTHON_ENV` / `YEAR` — the ROI to download over, data root, scratch dir, interpreter, and year.
- `DATA_SOURCE` — `mpc` (Microsoft Planetary Computer) or `aws`. For `aws`, Sentinel-1 needs an Earthdata bearer token at `~/.edl_bearer_token` (see the AWS Credentials section below).
- `S1_RAW_SUBDIR` / `S2_RAW_SUBDIR` — output folder names (optional, default `data_sar_raw` / `data_raw`).
- `S1_OVERWRITE` / `S2_OVERWRITE` — kept `false` here so a re-run *resumes*: each day's output is skipped if it already exists **and** passes a shape/CRS/transform check (the guard lives in `process_day_orbit` / `process_day`), so only missing or invalid days are re-fetched. Set `true` to force a full clean re-download (the script default) — rarely needed, since invalid tiles are already re-fetched automatically.
- Advanced knobs inside `s1_s2_downloader.sh` (edit only if needed): `S1_PARTITIONS` / `S2_PARTITIONS` (time-slice parallelism), `S1_TOTAL_WORKERS` / `S2_TOTAL_WORKERS` (Dask workers), `START_TIME_OVERRIDE` / `END_TIME_OVERRIDE` (download a sub-range for a quick test).

### Step 2 — Stack into yearly composites → `2.data_processed/`

```bash
BASE_DIR="${DATA_DIR}" \
DOWNSAMPLE_RATE=1 \
S1_RAW_SUBDIR=1.data_sar_raw \
S2_RAW_SUBDIR=1.data_raw \
PROCESSED_SUBDIR=2.data_processed \
    bash tessera_preprocessing/s1_s2_stacker.sh
```

Stacks the Step 1 outputs along the time dimension into `.npy` composites under `${DATA_DIR}/2.data_processed`. Fast (Rust binaries).

- `BASE_DIR` — data root (same as Step 1's `OUT_DIR`).
- `DOWNSAMPLE_RATE` — `1` keeps full 10 m resolution (optional, default `1`).
- `S1_RAW_SUBDIR` / `S2_RAW_SUBDIR` / `PROCESSED_SUBDIR` — the three folder names; must match Step 1 (optional, default `data_sar_raw` / `data_raw` / `data_processed`).

### Step 3 — Patchify into tiles → `3.retiled_d_pixel/`

```bash
python tessera_preprocessing/dpixel_retiler.py \
    --tiff_path   "${ROI_TIFF}" \
    --d_pixel_dir "${DATA_DIR}/2.data_processed" \
    --out_dir     "${DATA_DIR}/3.retiled_d_pixel" \
    --patch_size  500 \
    --block_size  2000 \
    --num_workers 16 \
    --overwrite
```

- `--tiff_path` — reference TIFF defining the grid; `--d_pixel_dir` — Step 2 output; `--out_dir` — where tiles are written.
- `--patch_size` / `--block_size` — tile and super-block size in px (optional, default `500` / `2000`; good starting point for a ~5000×5000 px ROI at 10 m).
- `--num_workers` — parallel workers (optional, default `16`).
- `--overwrite` — wipe `out_dir` first; drop it to resume without re-doing finished patches (or use `--skip_existing`).

### Step 4 — TESSERA v2 inference → `4.embeddings_v2/`

```bash
mkdir -p "${DATA_DIR}/4.embeddings_v2"
python tessera_infer_v2/infer_v2.py \
    --model     medium \
    --data-root "${DATA_DIR}/3.retiled_d_pixel" \
    --out-dir   "${DATA_DIR}/4.embeddings_v2"
```

Writes one 128-d fp32 `.npy` per tile. Weights are fetched automatically from Hugging Face on first use (or run `python tessera_infer_v2/download_weights.py --model medium` beforehand).

- `--model` — `nano` / `small` / `medium` (default) / `large` / `teacher`; `--data-root` — Step 3 output; `--out-dir` — embeddings directory.
- `--device` — `cuda` (default if available) or `cpu`.
- `--batch-pixels` — pixels per forward pass; lower it if you run out of memory (optional, default `4096`).
- `--dim 16 --int8` — store a Matryoshka prefix as int8 + a scale map (much smaller). Other versions (v1.1 / v1.0) are documented under [Inference](#inference).

### Step 5 — Stitch & convert → `5.result/<BASENAME>.npy`, `5.result/<BASENAME>.tif`

```bash
mkdir -p "${DATA_DIR}/5.result"

# 5a. stitch per-tile embeddings into one map
python tessera_infer/stitch_tiled_representation.py \
    --d_pixel_retiled_path        "${DATA_DIR}/3.retiled_d_pixel" \
    --representation_retiled_path "${DATA_DIR}/4.embeddings_v2" \
    --downstream_tiff             "${ROI_TIFF}" \
    --out_dir                     "${DATA_DIR}/5.result" \
    --out_name                    "${BASENAME}"

# 5b. convert the .npy to a viewable GeoTIFF
python tessera_infer/convert_npy2tiff.py \
    --npy_path      "${DATA_DIR}/5.result/${BASENAME}.npy" \
    --ref_tiff_path "${ROI_TIFF}" \
    --out_dir       "${DATA_DIR}/5.result" \
    --downsample_rate 1
```

The `.tif` takes its name from the `.npy`, so both land in `5.result/` as `${BASENAME}.npy` and `${BASENAME}.tif`. Open the `.tif` in QGIS to inspect.

- Stitch — `--d_pixel_retiled_path` (Step 3) + `--representation_retiled_path` (Step 4) + `--downstream_tiff` (output extent) + `--out_dir`; `--out_name` sets the basename (optional, default `stitched_representation`).
- Convert — `--npy_path` (Step 5a output), `--ref_tiff_path` (supplies CRS + geotransform), `--out_dir`; `--downsample_rate` coarsens the output — `2` → 20 m (optional, default `1`).

### One-click blocks

Too much copy & paste? The pipeline can also be ran with fewer copy-paste blocks.(*you can run all blocks at once if the area of ROI is relatively small, e.g. $<400km^2$*)  
**Block Config  —** Copy this and edit following configs, paste edited configs and run in shell

```bash
DATA_DIR=/absolute/path/to/your/data_dir              # all outputs are written under here
ROI_SHP=/absolute/path/to/your/roi.shp
PYTHON_ENV=/absolute/path/to/python_env/bin/python    # absolute path to your interpreter; only the Step 1 downloader needs it
BASENAME=myregion                                     # final result file name (.npy and .tif)
YEAR=2025                                             # data year, range [2017-2025]
```

**Block A — ROI + download (Step 0 + Step 1).** This is the failure-prone part; because `overwrite=false`, re-run `bash tessera_preprocessing/s1_s2_downloader.sh` resumes — only the days that failed or timed out are re-fetched.

```bash
ROI_TIFF="${DATA_DIR}/0.roi/roi.tiff"     # ROI extent: downloaded over + used as geo-reference
mkdir -p "${DATA_DIR}/0.roi" "${DATA_DIR}/tmp"
python tessera_preprocessing/convert_shp_to_tiff.py \
    --shp_path "${ROI_SHP}" --tiff_path  "${ROI_TIFF}" --pixel_size 10

INPUT_TIFF="${ROI_TIFF}" OUT_DIR="${DATA_DIR}" TEMP_DIR="${DATA_DIR}/tmp" \
PYTHON_ENV="${PYTHON_ENV}" YEAR="${YEAR}" DATA_SOURCE=mpc \
S1_RAW_SUBDIR=1.data_sar_raw S2_RAW_SUBDIR=1.data_raw S1_OVERWRITE=false S2_OVERWRITE=false \
    bash tessera_preprocessing/s1_s2_downloader.sh
```

**Block B — stack → result (Step 2 → Step 5).** Paste and run this once Step 1 has downloaded cleanly.

```bash
BASE_DIR="${DATA_DIR}" DOWNSAMPLE_RATE=1 \
S1_RAW_SUBDIR=1.data_sar_raw S2_RAW_SUBDIR=1.data_raw PROCESSED_SUBDIR=2.data_processed \
    bash tessera_preprocessing/s1_s2_stacker.sh
python tessera_preprocessing/dpixel_retiler.py \
    --tiff_path "${ROI_TIFF}" --d_pixel_dir "${DATA_DIR}/2.data_processed" \
    --out_dir "${DATA_DIR}/3.retiled_d_pixel" \
    --patch_size 500 --block_size 2000 --num_workers 16 --overwrite
mkdir -p "${DATA_DIR}/4.embeddings_v2"
python tessera_infer_v2/infer_v2.py \
    --model medium --data-root "${DATA_DIR}/3.retiled_d_pixel" --out-dir "${DATA_DIR}/4.embeddings_v2"
mkdir -p "${DATA_DIR}/5.result"
python tessera_infer/stitch_tiled_representation.py \
    --d_pixel_retiled_path "${DATA_DIR}/3.retiled_d_pixel" \
    --representation_retiled_path "${DATA_DIR}/4.embeddings_v2" \
    --downstream_tiff "${ROI_TIFF}" --out_dir "${DATA_DIR}/5.result" --out_name "${BASENAME}"
python tessera_infer/convert_npy2tiff.py \
    --npy_path "${DATA_DIR}/5.result/${BASENAME}.npy" \
    --ref_tiff_path "${ROI_TIFF}" --out_dir "${DATA_DIR}/5.result" --downsample_rate 1
```

## Data Preprocessing

### Overview
_**We strongly recommend that you quickly review the entire tutorial before running the pipeline.**_

In this step, we stack a full year of Sentinel-1 and Sentinel-2 data along the time dimension to generate a composite. For Sentinel-2, the composite shape is (T,H,W,B), where T is the number of valid observations in that year, and B is the number of bands (we selected 10 bands). For Sentinel-1, we extracted both ascending and descending orbit data. Taking the ascending orbit as an example, the composite shape is (T',H,W,B'), where T' is the number of valid ascending observations in that year, and B' is 2 because we only obtain VV and VH bands.

> **Sentinel-2 channel order in `bands.npy`.** The 10 bands are stored in the order shown below. This is **not** the conventional Sentinel-2 wavelength-ascending order; it is the convention used during Tessera pretraining, and the released model checkpoints together with the `S2_BAND_MEAN` / `S2_BAND_STD` constants in `tessera_infer*/src/datasets/ssl_dataset.py` are all bound to this exact ordering. Do not reorder unless you are also retraining or reordering the input-projection weights of the checkpoint.
>
> | idx | name | S2 code |
> |---|---|---|
> | 0 | red | B04 |
> | 1 | blue | B02 |
> | 2 | green | B03 |
> | 3 | nir | B08 |
> | 4 | nir08 | B8A |
> | 5 | rededge1 | B05 |
> | 6 | rededge2 | B06 |
> | 7 | rededge3 | B07 |
> | 8 | swir16 | B11 |
> | 9 | swir22 | B12 |
>
> Sentinel-1 (`sar_ascending.npy`, `sar_descending.npy`) channels are `[VV, VH]`.

We initially sourced Sentinel-1 and Sentinel-2 data from Microsoft's Planetary Computer:
- Sentinel-1 data source: https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc
- Sentinel-2 data source: https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a

The new generation of embeddings will use OPERA from ASF DAAC: 
- Sentinel-1 data source:  https://registry.opendata.aws/nasa-operal2rtc-s1v1/ 
- Sentinel-2 data source: https://registry.opendata.aws/sentinel-2-l2a-cogs/

Currently, our pipeline only accepts TIFF format input. The resolution of the input ROI TIFF can vary (e.g., 30m), but the pipeline will **always generate Sentinel-1 and Sentinel-2 outputs at the configured `RESOLUTION`** (default 10m) while keeping the **ROI extent/bounds identical**. For valid ROI areas within the TIFF, the value is 1; otherwise, it's 0. If you only have a shapefile, that's fine too - we provide a `convert_shp_to_tiff.py` script.

### Download Source Code

First, create an empty working directory:

```bash
mkdir tessera_project
cd tessera_project
git clone https://github.com/ucam-eo/tessera.git
```

For easier pipeline operation, we recommend placing the data output directory at the same level as `tessera_infer` and `tessera_preprocessing`:

```
tessera_project
 ┣ tessera_infer
 ┣ tessera_preprocessing
 ┣ my_data
   ┣ roi.shp (your shapefile)
   ┗ roi.tiff (we recommend generating this using convert_shp_to_tiff.py)
```

The `roi.tiff` can be generated using `convert_shp_to_tiff.py` located in `tessera_preprocessing/convert_shp_to_tiff.py`:

```bash
cd tessera_preprocessing
python convert_shp_to_tiff.py \
    --shp_path   /absolute/path/to/your/data_dir/roi.shp \
    --pixel_size 10
```

By default it writes a TIFF with the same name as the shapefile in the same directory (e.g. `roi.shp` → `roi.tiff`) and a `<name>_convex_hull.tiff` alongside it; override the destination with `--tiff_path`. Set `--pixel_size` in meters (default `10`), and optionally force a target CRS with `--force_crs EPSG:32650` (default: auto-detect the best UTM zone from the shapefile centroid).

⚠️Notice: _If your ROI is relatively large, for example 100 km × 100 km, we strongly recommend pre-splitting the TIFF into smaller sections no larger than 20 km × 20 km. Then process each small TIFF file sequentially in the pipeline. An excessively large ROI may cause issues with backend tile providers_

### Python Environment

We need some geographic processing packages (fortunately, we won't be using GDAL, as configuring the environment is a nightmare) and some machine learning packages (PyTorch, but you'll need to install this yourself since the hardware on each computer is different). We've put some common packages in `requirements.txt`, which you can install as follows:

```bash
pip install -r requirements.txt
```
Note: If you are in a managed environment, you may need to install a venv first, using 
```bash
python3 -m venv venv
source venv/bin/activate
```

### Script Configuration

First, navigate to the `tessera_preprocessing` folder:

```bash
cd tessera_preprocessing
```

Then edit the file s1_s2_downloader.sh to point to the ROI TIFF file, the output and temporary directories, and the data source:

```bash
# === Basic Configuration ===
INPUT_TIFF="/absolute/path/to/your/data_dir/roi.tiff"
OUT_DIR="/absolute/path/to/your/data_dir"

export TEMP_DIR="/absolute/path/to/your/temp_dir"     # Temporary file directory

mkdir -p "$OUT_DIR"

# Python environment path
PYTHON_ENV="/absolute/path/to/your/python_env/bin/python"

# === Sentinel-1 & Sentinel-2 Processing Configuration ===
YEAR=2022 # Range [2017-2025]
RESOLUTION=10.0  # Output resolution (meters). ROI TIFF can be any resolution; extent is preserved.

# === Data Source Configuration ===
# mpc: Microsoft Planetary Computer (sentinel-1-rtc, sentinel-2-l2a)
# aws: AWS Open Data backends (S1=OPERA RTC-S1 via ASF/CMR + ASF Earthdata Cloud COGs, S2=Earth-search Sentinel-2 L2A COGs)
DATA_SOURCE="mpc"   # choices: mpc/aws
```

Note: `RESOLUTION` controls output pixel size. The pipeline keeps the ROI bounds fixed and resamples the ROI mask into the output grid.

### AWS Credentials (only needed when `DATA_SOURCE="aws"`)
Sentinel-2 on Earth-search is public and **does not require credentials**.

Sentinel-1 OPERA RTC-S1 is accessed via ASF Earthdata Cloud (COG over HTTPS). You need an Earthdata Login token:
- **Create an Earthdata account**: via [NASA Earthdata Login](https://urs.earthdata.nasa.gov/home).
- **Approve Application**: After registering your account, you can go to the Applications tab and add Alaska Satellite Facility Data Access to the list of approved applications.
- **Obtain an EDL Bearer token / JWT** by clicking **Generate Token** and store it locally (do not commit it).

Recommended (simple + explicit):

```bash
nano ~/.edl_bearer_token
# paste token, save+exit (Ctrl-O Enter, then Ctrl-X)
chmod 600 ~/.edl_bearer_token
```

The AWS S1 downloader will use this token to read COGs from ASF Earthdata Cloud.

If you want to retrieve temporary S3 credentials (advanced; usually not required for this pipeline), see ASF guidance:
- `https://cumulus.asf.alaska.edu/s3credentialsREADME`

Below the above configuration, there are some additional configurations that you can modify according to your computer's performance.

First, give permission to `s1_s2_downloader.sh`:

```bash
chmod +x s1_s2_downloader.sh
```

Then, we can run:

```bash
bash s1_s2_downloader.sh
```

Due to network conditions, processing some tiles may time out. Our script includes sophisticated timeout management to avoid these issues. However, sometimes some tiles may still fail. Running the above command again usually resolves this.

If all Sentinel-1 and Sentinel-2 data are generated correctly, they can be stacked along the time dimension. For this step, we use two Rust-generated executables, making it very fast.

#### (Optional) Build `s1_stack` and `s2_stack` from source

The repository ships pre-built `s1_stack` and `s2_stack` binaries directly under `tessera_preprocessing/` so most users do not need to compile anything. The full Rust source for both stackers lives next to them:

```
tessera_preprocessing
 ┣ rust_for_s1_stacking
 ┃  ┣ Cargo.toml
 ┃  ┣ Dockerfile
 ┃  ┗ src/main.rs
 ┣ rust_for_s2_stacking
 ┃  ┣ Cargo.toml
 ┃  ┣ Cargo.lock
 ┃  ┣ Dockerfile
 ┃  ┗ src/main.rs
 ┣ s1_stack          # pre-built binary
 ┗ s2_stack          # pre-built binary
```

Rebuild from source if you want to modify the stacker (e.g. change channel order, sampling logic) or need a binary for a different architecture.

**Option A — Local Cargo build (fastest path; produces a dynamically linked binary):**

```bash
# Requires rustc / cargo >= 1.70
cd tessera_preprocessing/rust_for_s1_stacking
cargo build --release
cp target/release/s1_stack ../s1_stack          # overwrite the shipped binary
chmod +x ../s1_stack

cd ../rust_for_s2_stacking
cargo build --release
# the s2 crate name is `process_tile_downstream_wo_json`
cp target/release/process_tile_downstream_wo_json ../s2_stack
chmod +x ../s2_stack
```

**Option B — Docker build (matches the way the shipped binaries are produced; statically linked against musl, runs on any Linux x86_64):**

```bash
# s1
cd tessera_preprocessing/rust_for_s1_stacking
docker build -t s1_stack .
docker create --name tmp_s1 s1_stack
docker cp tmp_s1:/s1_stack ../s1_stack
docker rm tmp_s1
chmod +x ../s1_stack

# s2
cd ../rust_for_s2_stacking
docker build -t s2_stack .
docker create --name tmp_s2 s2_stack
docker cp tmp_s2:/s2_stack ../s2_stack
docker rm tmp_s2
chmod +x ../s2_stack
```

Sanity-check the resulting binaries:

```bash
./s1_stack --help
./s2_stack --help
```

> **Important — do not casually change the S2 channel order.** The `BANDS` constant in `rust_for_s2_stacking/src/main.rs` defines the output channel order in `bands.npy`. It is bound to the released model checkpoints and the hard-coded `S2_BAND_MEAN` / `S2_BAND_STD` constants in `tessera_infer*/src/datasets/ssl_dataset.py`. Reordering it without also retraining (or reordering the input-projection weights of the checkpoint) will silently corrupt inference.

#### Run the stacker

You can open `s1_s2_stacker.sh` and edit the following:

```bash
# === Basic Configuration ===
BASE_DIR="/absolute/path/to/your/data_dir"
OUT_DIR="${BASE_DIR}/data_processed"
DOWNSAMPLE_RATE=1
```

Normally, we don't modify `DOWNSAMPLE_RATE`, which keeps it from performing any downsampling during stacking. The `BASE_DIR` in the above snippet is the same as the `OUT_DIR` you modified in `s1_s2_downloader.sh`.

Similarly, give permission to `s1_s2_stacker.sh`:

```bash
chmod +x s1_s2_stacker.sh
```

Then you can execute the stacking:

```bash
bash s1_s2_stacker.sh
```

After success, you will get some `.npy` files in `/absolute/path/to/your/data_dir/data_processed`. Usually, these `.npy` files are quite large, so we will patchify them into smaller, more manageable units.

Execute:

```bash
python dpixel_retiler.py \
    --tiff_path /absolute/path/to/your/data_dir/roi.tif \
    --d_pixel_dir /absolute/path/to/your/data_dir/data_processed \
    --patch_size 500 \
    --out_dir /absolute/path/to/your/data_dir/retiled_d_pixel \
    --num_workers 16 \
    --overwrite \
    --block_size 2000
```

You can change the above `patch_size` and `block_size` yourself. The above configuration is a recommended configuration for a TIFF with a shape of (5000,5000) and a 10m resolution.

If the above code runs smoothly, you can get some subfolders in `my_data/retiled_d_pixel`.

## Inference

Once preprocessing has produced your tiles, you run a TESSERA model over them to
generate embeddings. TESSERA comes in **several versions** — start with
[**Before you start**](#before-you-start) (shared by all versions), pick one in
[**Which version should I use?**](#which-version-should-i-use), follow that
version's own subsection, then [**stitch the tiles**](#stitch-the-tiles-into-a-representation-map)
into a single map.

### Before you start

**1. Check your preprocessed tiles.** Every version reads the same tile
directories produced by `tessera_preprocessing`. Confirm `my_data/retiled_d_pixel`
contains per-tile subfolders:

```
retiled_d_pixel
 ┣ 0_3500_500_4000
 ┣ 0_4000_500_4500
 ┣ 0_4500_500_5000
 ┗ ...
```

and that each subfolder has these files:

```
0_3500_500_4000
 ┣ bands.npy
 ┣ doys.npy
 ┣ masks.npy
 ┣ roi.tiff
 ┣ sar_ascending.npy
 ┣ sar_ascending_doy.npy
 ┣ sar_descending.npy
 ┗ sar_descending_doy.npy
```

If these files are missing, revisit the [Data Preprocessing](#data-preprocessing)
step.

**2. Install PyTorch.** Inference only needs PyTorch (no GDAL/SNAP), so it is far
simpler to set up than preprocessing. If you don't already have it, check your
CUDA version with `nvidia-smi`, then install the matching build from
https://pytorch.org/, for example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Which version should I use?

| Version | Code | Weights | Output | Use it when |
| ------- | ---- | ------- | ------ | ----------- |
| **[TESSERA v2](#tessera-v2-recommended)** (recommended) | `tessera_infer_v2/` | [Hugging Face](https://huggingface.co/geotessera) | 128-d Matryoshka (16/32/64/128) fp32, optional int8 | New projects — best quality, smallest models, flexible embedding size |
| **[TESSERA v1.1](#tessera-v11-qat-int8)** | `tessera_infer_QAT/` | [Google Drive](https://drive.google.com/drive/folders/18RPptbUkCIgUfw1aMdMeOrFML_ZVMszn?usp=sharing) | 128-d int8 + scales | You need the model behind the current GeoTessera int8 embeddings, MPC or AWS |
| **[TESSERA v1.0 (QAT)](#tessera-v10-qat-int8)** | `tessera_infer_QAT/` | [Google Drive](https://drive.google.com/file/d/1HJ92aS5ERXMLfSFYJ4m3OKycJJdC1QvO/view?usp=sharing) | 128-d int8 + scales | Reproducing the original int8 embedding product |
| **[TESSERA v1.0 (early)](#tessera-v10-early-float32)** | `tessera_infer/` | [Google Drive](https://drive.google.com/drive/folders/18RPptbUkCIgUfw1aMdMeOrFML_ZVMszn?usp=sharing) | 128-d fp32 | Reproducing early fp32 results only |

The four code directories are independent and their checkpoints are **not**
interchangeable — a checkpoint only loads with the pipeline it belongs to. If in
doubt, use **v2**.

---

### TESSERA v2 (recommended)

**TESSERA v2** is the current generation of the model — see the preprint,
[*TESSERA v2: Scaling Pixel-wise Earth Foundation Models*](https://arxiv.org/abs/2607.03949).
Its inference code lives in **[`tessera_infer_v2/`](tessera_infer_v2/)**.

> **Don't want to run inference yourself?** You can ask us to generate v2 embeddings
> for your region instead. Submit a
> [**v2 Embedding Pre-Request**](https://github.com/ucam-eo/geotessera/issues/new?template=v2-embedding-prerequest.yml&labels=v2-embedding-prerequest)
> and we'll **prioritize your area** and add you to the **early testers**. Note that v2
> coverage is still being rolled out, so if you need embeddings right away, request
> [v1.1 embeddings](https://github.com/ucam-eo/geotessera#request-missing-embeddings)
> instead — they're available today.

v2 ships **four compact pixel students** plus the **2B teacher** they were distilled from:

| Model | Parameters | Output | Hugging Face repository |
| ----- | ---------- | ------ | ----------------------- |
| Nano | 1.07 M | 128-d Matryoshka | [`geotessera/TESSERA-V-2.0-2B-N`](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-N) |
| Small | 7.11 M | 128-d Matryoshka | [`geotessera/TESSERA-V-2.0-2B-S`](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-S) |
| **Medium** (recommended) | 21.03 M | 128-d Matryoshka | [`geotessera/TESSERA-V-2.0-2B-M`](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-M) |
| Large | 43.83 M | 128-d Matryoshka | [`geotessera/TESSERA-V-2.0-2B-L`](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-L) |
| 2B teacher | 2,064,266,242 | 1024-d | [`geotessera/TESSERA-V-2.0-2B-Teacher`](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-Teacher) |

The students emit **Matryoshka** embeddings: the first K dimensions are independently
usable for K ∈ {16, 32, 64, 128}, so 16, 32 or 64 dimensions can be stored instead of
128 with no retraining and no second checkpoint. The `2B` in each name records the
teacher the model was distilled from.

> The 2B teacher is **not a deployment model** — it evaluates 2.06 billion parameters
> per pixel, which makes tile-scale, let alone global-scale, embedding generation
> impractical on ordinary hardware. It is published so the distillation is reproducible.
> For real work, use one of the students.

#### Download the weights

v2 checkpoints are **not** stored in this repository; they are hosted on the Hugging
Face Hub under [`geotessera`](https://huggingface.co/geotessera) and fetched on demand:

```bash
cd tessera_infer_v2
pip install -r requirements.txt

python download_weights.py --model medium        # the recommended default (84 MB)
python download_weights.py --model all-students  # nano + small + medium + large
python download_weights.py --model teacher       # 8.26 GB
```

They land in `tessera_infer_v2/student/checkpoints/` and
`tessera_infer_v2/teacher/checkpoints/`, which is where the inference script looks by
default. A single file can also be pulled directly:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("geotessera/TESSERA-V-2.0-2B-M", "ckpt/student_medium.pt")
```

#### Run v2 inference

`tessera_infer_v2/infer_v2.py` consumes the preprocessed tile directories
described in [Before you start](#before-you-start):

```bash
cd tessera_infer_v2

# default student, fp32 128-d output, one .npy per tile
python infer_v2.py --model medium \
    --data-root my_data/retiled_d_pixel \
    --out-dir   my_data/embeddings_v2

# 16-d Matryoshka prefix stored as int8 + a float32 scale map
python infer_v2.py --model medium --dim 16 --int8 \
    --data-root my_data/retiled_d_pixel \
    --out-dir   my_data/embeddings_v2_d16

# the 2B teacher on a single tile (GPU strongly recommended)
python infer_v2.py --model teacher --bf16 --batch-pixels 512 \
    --tile    my_data/retiled_d_pixel/0_3500_500_4000 \
    --out-dir my_data/embeddings_v2_teacher
```

Or call the encoders directly:

```python
import sys, torch
sys.path.insert(0, "tessera_infer_v2/student")

from model import load_model
from infer import encode_tile

model = load_model("tessera_infer_v2/student/checkpoints/student_medium.pt",
                   torch.device("cuda"))
emb = encode_tile(model, s2_bands, s2_doys, s2_masks=s2_masks,
                  s1_asc_bands=s1_asc, s1_asc_doys=s1_asc_doys,
                  s1_desc_bands=s1_desc, s1_desc_doys=s1_desc_doys,
                  device=torch.device("cuda"))    # -> (H, W, 128)
emb16 = emb[..., :16]                             # Matryoshka truncation
```

> **Two conventions that fail silently.** The Sentinel-2 channel order is
> `B04 B02 B03 B08 B8A B05 B06 B07 B11 B12`, *not* ascending wavelength (data from
> `tessera_preprocessing` is already correct). And the students normalise Sentinel-1
> ascending/descending with their own per-source statistics before merging, whereas the
> teacher merges them raw and applies a single pooled set — do not carry one
> normalisation across to the other model. Full details in
> [`tessera_infer_v2/README.md`](tessera_infer_v2/README.md).

When inference finishes, jump to
[Stitch the tiles into a representation map](#stitch-the-tiles-into-a-representation-map).

---

### TESSERA v1.1 (QAT, int8)

TESSERA **v1.1** is a pretrained QAT model that improves on the v1.0 QAT
checkpoint. Its code lives in **[`tessera_infer_QAT/`](tessera_infer_QAT/)**.
Compared to v1.0, v1.1 brings:

- **Wider encoder** (`latent_dim=192`, transformer `d_model=768`) and an **MLP
  `dim_reducer`** (Linear → LayerNorm → ReLU → Dropout → Linear) that outputs a
  192-D representation; the first 128 dims are saved for downstream use.
- **All-observation inference.** Unlike v1.0, which randomly samples a fixed 40
  timesteps per pixel, v1.1 uses **every valid observation** per pixel. Observation
  counts are bucketised to a configurable list `num_obs_checkpoints` (default: every
  multiple of 8 from 8 to 256, i.e. `[8, 16, 24, 32, ..., 248, 256]`) so pixels
  sharing the same bucket can be batched together.
- **Per-modality S1 normalisation.** S1 ascending and descending are normalised
  with their OWN per-band mean/std, then concatenated time-wise into a single
  merged S1 stream that feeds one S1 backbone (same two-backbone topology as
  v1.0).

#### Two data sources × two checkpoint flavours

v1.1 ships **two checkpoints per preprocessing source × two flavours each**:

- **encoder-only** (~250 MB, default for inference) — contains exactly the
  weights consumed by the inference graph (`s2_backbone`, `s1_backbone`,
  `dim_reducer`). This is what you almost always want.
- **full** (~10 GB) — encoder + projector + optimiser/scaler state. Only
  download this if you intend to **fine-tune** v1.1 on your own data.

Each checkpoint was trained with its own normalisation statistics, so **you
must pair the right checkpoint with the right `data_source` setting in the
config** (otherwise the input distribution is silently mis-shifted and embedding
quality collapses):

| Data source | Encoder-only (recommended, ~221 MB)            | Full (fine-tune, ~10 GB)                         | `data_source` value |
| ----------- | ----------------------------------------------- | ------------------------------------------------- | ------------------- |
| Microsoft Planetary Computer (S2 L2A + S1 RTC)   | [`tessera_v1_1_mpc_encoder.pt`](https://drive.google.com/file/d/1t-gfTxi3Hg_uJXpJ9etROCRgKt2myfJ2/view?usp=drive_link) | [`tessera_v1_1_mpc_full.pt`](https://drive.google.com/file/d/1pBXlBscBedlh0CkfD6vW277XkN8WevZA/view?usp=sharing) | `"mpc"` |
| AWS Open Data (Earth-search S2 L2A + ASF OPERA RTC-S1) | [`tessera_v1_1_aws_encoder.pt`](https://drive.google.com/file/d/1taLxwJOId-pfqUafEOCf5zDPXA7kzdyu/view?usp=sharing) | [`tessera_v1_1_aws_full.pt`](https://drive.google.com/file/d/1GqtqaAPaJhyZzQxxjtnZOqG3dq5JQzMK/view?usp=sharing) | `"aws"` |

> **AWS checkpoint update (2026-05-03).** The AWS v1.1 checkpoint was
> retrained on data collected with the corrected preprocessing pipeline
> (see `tessera_preprocessing/s2_fast_processor.py::harmonize_arr`, which
> previously double-applied the PB-04.00 BOA_ADD_OFFSET on AWS / Earth-search
> Sentinel-2 data). The AWS row in
> `tessera_infer_QAT/src/datasets/v1_1_norm_stats.py` has been refreshed
> against the new pretraining distribution.

Per-source normalisation statistics are kept in
`tessera_infer_QAT/src/datasets/v1_1_norm_stats.py`. The config field
`data_source` (default `"mpc"`) selects which set is used at inference time.

The inference loader uses `strict=False` and silently ignores keys that
aren't part of the inference graph, so a *full* checkpoint will also work in
the same command — it's just ~45× larger to download. (A HuggingFace mirror
will follow once the Drive links stabilise.)

#### Download the weights

**Download** one (or more) of the four checkpoints above and place into
`tessera_infer_QAT/checkpoints`:

```
tessera_infer_QAT
 ┣ checkpoints
 ┃   ┣ tessera_v1_1_mpc_encoder.pt          # MPC, encoder-only (default)
 ┃   ┣ tessera_v1_1_aws_encoder.pt          # AWS, encoder-only
 ┃   ┣ tessera_v1_1_mpc_full.pt             # MPC, full (fine-tune only)
 ┃   ┗ tessera_v1_1_aws_full.pt             # AWS, full (fine-tune only)
 ┣ configs
 ┃   ┗ v1_1_infer_config.py                 # set data_source = "mpc" or "aws"
 ┣ src
 ┃   ┣ infer_v1_1.py
 ┃   ┗ datasets
 ┃       └─ v1_1_norm_stats.py
 ┗ visualize_embedding_v1_1.py
```

#### Run v1.1 inference

**Run inference** on a single preprocessed tile:

```bash
cd tessera_infer_QAT

# MPC inference (default; encoder-only ckp is enough)
python src/infer_v1_1.py \
    --config          configs/v1_1_infer_config.py \
    --checkpoint_path checkpoints/tessera_v1_1_mpc_encoder.pt \
    --tile_path       /absolute/path/to/retiled_d_pixel/0_3500_500_4000 \
    --output_dir      /absolute/path/to/representation_retiled_v1_1
```

For AWS data, pass `--data_source aws` and use the AWS encoder ckp:
`--checkpoint_path checkpoints/tessera_v1_1_aws_encoder.pt`.

Adjust `num_obs_checkpoints` to trade off between embedding quality and compute
(fewer / smaller checkpoints = faster, more / larger = more temporal detail).
For a Slurm cluster, see `tessera_infer_QAT/infer_v1_1.slurm` for a
ready-to-edit template.

**Output** files (int8 + scales, with `_emb128_` naming):
- `<prefix>_emb128_int8.npy`   — shape `(H, W, 128)`, dtype `int8`
- `<prefix>_emb128_scales.npy` — shape `(H, W)`,      dtype `float32`

Reconstruct fp32 embeddings with
`fp32 = int8.astype(np.float32) * scales[..., None]`. A helper script
`visualize_embedding_v1_1.py` dequantises the output and saves a first-3-dim RGB
plus a PCA-3 RGB for quick visual inspection.

When inference finishes, jump to
[Stitch the tiles into a representation map](#stitch-the-tiles-into-a-representation-map).

---

### TESSERA v1.0 (QAT, int8)

The original quantization-aware model — this is the checkpoint used to generate
the int8 embeddings in the GeoTessera library. Its code lives in
**[`tessera_infer_QAT/`](tessera_infer_QAT/)**.

#### Download the weights

Download the QAT checkpoint from
[Google Drive](https://drive.google.com/file/d/1HJ92aS5ERXMLfSFYJ4m3OKycJJdC1QvO/view?usp=sharing)
and place it in `tessera_infer_QAT/checkpoints`:

```
tessera_infer_QAT
 ┣ checkpoints
 ┃   ┗ best_model_fsdp_20250608_220648_QAT.pt
 ┣ configs
 ┗ src
```

The QAT pipeline outputs quantized embeddings as **int8 + scales**:
- `tile_name.npy`: int8 embedding tensor, shape `(H, W, 128)`
- `tile_name_scales.npy`: float32 scale map, shape `(H, W)`

#### Run v1.0 QAT inference

The QAT pipeline runs through its own batch script, `tessera_infer_QAT/infer_all_tiles.sh`:

```bash
cd tessera_infer_QAT
chmod +x infer_all_tiles.sh
bash infer_all_tiles.sh
```

Before running, edit these parameters in `tessera_infer_QAT/infer_all_tiles.sh`:

```bash
BASE_DATA_DIR="/absolute_path_to_your_data_dir"
export PYTHON_ENV="/absolute_path_to_your_python/bin/python"
CPU_GPU_SPLIT="1:1"  # CPU:GPU ratio, e.g. 1:0 or 0:1
CHECKPOINT_PATH="checkpoints/best_model_fsdp_20250608_220648_QAT.pt"
```

Notes:
- QAT supports both CPU and GPU inference in one run (ratio-based split, same
  style as the v1.0 early runner below).
- On CPU, AMX is automatically detected and enabled when available; if AMX is
  not available, it automatically falls back to default CPU inference.

When inference finishes, jump to
[Stitch the tiles into a representation map](#stitch-the-tiles-into-a-representation-map).

---

### TESSERA v1.0 (early, float32)

The earliest public checkpoint. It natively generates **float32** embeddings, so
it is **not** the model used for the int8 embeddings in the GeoTessera library —
use [v1.0 QAT](#tessera-v10-qat-int8) or [v1.1](#tessera-v11-qat-int8) for those.
Its code lives in **[`tessera_infer/`](tessera_infer/)**.

#### Download the weights

Download the model weights from
[Google Drive](https://drive.google.com/drive/folders/18RPptbUkCIgUfw1aMdMeOrFML_ZVMszn?usp=sharing)
and place the `.pt` file in the `tessera_infer/checkpoints` directory:

```
tessera_infer
 ┗ checkpoints
     ┗ best_model_fsdp_20250427_084307.pt
 ┗ configs
 ┗ src
```

#### Configure the batch script

Inference runs through `tessera_infer/infer_all_tiles.sh`. You only need to edit
a few parameters:

a. Base data directory:
```bash
BASE_DATA_DIR="your_data_directory"
```
This is your data storage folder, the same as `BASE_DATA_DIR` used in preprocessing, e.g., `/maps/usr/tessera_project/my_data`

b. Python environment:
```bash
export PYTHON_ENV="your_python_path"
```
Write the absolute path to your Python environment here, e.g., `/home/user/anaconda3/envs/tessera_env/bin/python`

c. CPU/GPU split:
```bash
CPU_GPU_SPLIT="1:1"  # Format: CPU:GPU ratio
```
The script supports simultaneous inference using both CPU and GPU. This ratio specifies the proportion of `retiled_patches` each device will handle. Default is 1:1 (even split). For GPU-only inference, set to 0:1.

d. CPU Related Settings

```bash
MAX_CONCURRENT_PROCESSES_CPU=20
```
Maximum number of CPU processes for tile inference. For example, if set to 20, it will process 20 tiles simultaneously.

```bash
AVAILABLE_CORES=$((TOTAL_CPU_CORES / 2)) # Use 50% of the cores
```
Number of CPU cores to use. Please modify this value if necessary to avoid consuming too many CPU resources!

e. GPU Related Settings:
```bash
MAX_CONCURRENT_PROCESSES_GPU=1
```
Maximum number of GPU processes for inference. If the system has only 1 GPU, set this to 1.

```bash
GPU_BATCH_SIZE=1024  # Larger for GPU, if this takes too much memory, reduce it
```
Number of samples to process at once during PyTorch inference. If this value consumes too much GPU memory or causes an OOM error on the GPU, please reduce it accordingly.

f. Other Settings
There are other parameters available for configuration. Please adjust them as needed.

#### Run v1.0 early inference

Once everything is ready, navigate to the `tessera_infer` folder, make the script
executable, and run it:

```bash
cd tessera_infer
chmod +x infer_all_tiles.sh
bash infer_all_tiles.sh
```

If successful, you should see logs like:

```
(base) zf281@daintree:/scratch/zf281/tessera_project/tessera_infer$ bash infer_all_tiles.sh
[INFO] Total CPU cores: 256, Using: 192
[INFO] CPU:GPU split ratio = 1:1 (total: 2)

==== SETUP DIRECTORIES ====
[SUCCESS] Created necessary directories

==== SCANNING TILES ====
[INFO] Tile directory: /scratch/zf281/jovana/retiled_d_pixel
[INFO] Output directory: /scratch/zf281/jovana/representation_retiled
[SUCCESS] Found 226 tiles total
[INFO] Sample tiles:
  - 0_3500_500_4000
  - 0_4000_500_4500
  - 0_4500_500_5000
  - ...
```

At the same time, a `logs` folder will be generated in the `tessera_infer` folder with more detailed logging for each CPU and GPU process.

---

### Stitch the tiles into a representation map

This step is the same for every version. Inference usually takes a long time, depending on your ROI size and hardware performance. Once completed, you can find many `.npy` files in `my_data/representation_retiled`:

```
representation_retiled
 ┣ 0_3500_500_4000.npy
 ┣ 0_4000_500_4500.npy
 ┣ 0_4500_500_5000.npy
 ┣ 0_5000_500_5500.npy
 ┣ 0_5500_500_6000.npy
 ┣ 0_6000_500_6500.npy
 ┣ 0_6500_500_7000.npy
 ┣ 0_7000_500_7500.npy
 ┣ 1000_0_1500_500.npy
 ┣ 1000_1000_1500_1500.npy
 ┣ 1000_1500_1500_2000.npy
 ┣ 1000_2000_1500_2500.npy
```

The final step is to stitch them together using `tessera_infer/stitch_tiled_representation.py`:

```bash
python stitch_tiled_representation.py \
--d_pixel_retiled_path /path/to/d_pixel_retiled \
--representation_retiled_path /path/to/representation_retiled \
--downstream_tiff /path/to/downstream.tiff \
--out_dir /path/to/output_directory
```

For example:

```bash
python stitch_tiled_representation.py \
--d_pixel_retiled_path /maps/usr/tessera_project/my_data/d_pixel_retiled \
--representation_retiled_path /maps/usr/tessera_project/my_data/representation_retiled \
--downstream_tiff /maps/usr/tessera_project/my_data/downstream.tiff \
--out_dir /maps/usr/tessera_project/my_data
```

Finally, you'll get a stitched representation map in the `my_data` directory with the shape (H,W,C), where H and W match your initial `roi.tiff` and C is the embedding dimension of the version you ran. The representation map is a NumPy array. If you want to convert it to TIFF for viewing in software like QGIS, you can use the `tessera_infer/convert_npy2tiff.py` script:

```bash
python convert_npy2tiff.py \
    --npy_path      /path/to/stitched_representation.npy \
    --ref_tiff_path /path/to/roi.tiff \
    --out_dir       /path/to/output_directory \
    --downsample_rate 1
```

For example:

```bash
python convert_npy2tiff.py \
    --npy_path      /maps/usr/tessera_project/my_data/stitched_representation.npy \
    --ref_tiff_path /maps/usr/tessera_project/my_data/roi.tiff \
    --out_dir       /maps/usr/tessera_project/my_data \
    --downsample_rate 1
```

The output GeoTIFF borrows its CRS and geotransform from `--ref_tiff_path`. Set `--downsample_rate` (integer, area-average, default `1`) to coarsen the resolution — e.g. `2` produces a 20m output.

## Downstream tasks

If you want to reproduce the downstream tasks in the paper, you can visit https://github.com/ucam-eo/tessera-downstream-task. There are many examples provided there.

# Additional information

## Team

### Cambridge Faculty
* [S. Keshav](https://svr-sk818-web.cl.cam.ac.uk/keshav/wiki/index.php/Main_Page)
* [Anil Madhavapeddy](https://anil.recoil.org)
* [Sadiq Jaffer](https://toao.com)
* [David Coomes](https://www.plantsci.cam.ac.uk/directory/david-coomes)

### Postdoc
* James Ball
  
### PhD
* Madeleine Lisaius
* Zhengpeng (Frank) Feng
* Robin Young
* Jovana Knezevic

### Undergrad
* Zejia Yang (Part II student, working with Frank Feng on MAE pretraining of spatial feature extractors)

### Interns
* Kenzy Soror (U. Waterloo, working with Robin Young)
* Artyom Gabtraupov (U. Waterloo, working with Robin Young)
* Gabriel Mahler (U. Cambridge, working with Anil Madhavapeddy and Silviu Petrovan on [hedgehog habitats and tracking](https://anil.recoil.org/ideas/hedgehog-mapping))
* Leyu Pan (Imperial College, working with Frank Feng on text embeddings generated from OSM)

### Collaborators
* [Clement Atzberger](https://www.linkedin.com/in/clement-atzberger-8abb8065/?originalSubdomain=at), dClimate Labs
* [Andrew Blake](https://en.wikipedia.org/wiki/Andrew_Blake_(computer_scientist)), Mantle Labs

### Visitors
* Silja Sormunnen, Aalto University, Finland
* Isabel Mansley (U. Edinburgh, working with David Coomes and Anil Madhavapeddy on [habitat mapping in Scotland](https://anil.recoil.org/ideas/cairngorms-connect-habitats)

## Contact

Please direct your technical questions to Frank Feng (zf281@cam.ac.uk) or ask it on our [Zulip forum](https://eeg.zulipchat.com/login/). Non-technical questions can be sent to Prof. S. Keshav (sk818@cam.ac.uk).

## Citation

If you use TESSERA in your research, please cite the relevant paper(s).

**TESSERA v2** ([arXiv:2607.03949](https://arxiv.org/abs/2607.03949)):

```bibtex
@article{feng2026tesserav2,
  title={TESSERA v2: Scaling Pixel-wise Earth Foundation Models},
  author={Feng, Zhengpeng and Jaffer, Sadiq and Shokar, Ira and Knezevic, Jovana and Elvers, Mark and Atzberger, Clement and Young, Robin and Naik, Aneesh and Robinson, Niall and Blake, Andrew and others},
  journal={arXiv preprint arXiv:2607.03949},
  year={2026}
}
```

**TESSERA (v1, CVPR 2026)** ([arXiv:2506.20380](https://arxiv.org/abs/2506.20380)):

```bibtex
@inproceedings{feng2026tessera,
  title={Tessera: Temporal embeddings of surface spectra for earth representation and analysis},
  author={Feng, Zhengpeng and Atzberger, Clement and Jaffer, Sadiq and Knezevic, Jovana and Sormunen, Silja and Young, Robin and Lisaius, Madeline C and Immitzer, Markus and Jackson, Toby and Ball, James and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={34818--34831},
  year={2026}
}
```

## Acknowledgments
We would like to express our gratitude to UKRI, the [Isambard AI](https://www.bristol.ac.uk/research/centres/bristol-supercomputing/#isambard-ai) supercomputer team at Bristol, and the [DAWN](https://www.hpc.cam.ac.uk/d-w-n) supercomputer team at Cambridge, for their generous support in this project. We also acknowledge support from [NVIDIA](https://www.nvidia.com/), [AMD](https://www.amd.com/en.html),  [Vultr](https://www.vultr.com/), the [Dirac High Performance Computing Facility](https://dirac.ac.uk), [Microsoft AI For Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/), Dr. Robert Sansom, [dClimate](https://www.dclimate.net/), and [Amazon Web Services (AWS)](https://aws.amazon.com/) under their AWS Open Data program (https://opendata.aws/). This work would not have been possible without their support, computational resources and technical assistance.  

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=ucam-eo/tessera&type=Date)](https://www.star-history.com/#ucam-eo/tessera&Date)

## AUP

### TESSERA Terms of Use and Ethical Guidelines

### License

TESSERA data and embeddings are made available under the **Creative Commons 0 International License [CC-0](https://creativecommons.org/public-domain/cc0/)**. 
This means you are free to:

* **Share** — copy and redistribute the material in any medium or format
* **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

### Purpose and Intended Uses

TESSERA was developed to advance scientific research and support environmental monitoring, conservation, sustainable agriculture, and understanding of Earth systems. We designed this tool to enable:

* Scientific research and education
* Environmental monitoring and conservation
* Agricultural and food security analysis
* Climate change research and adaptation planning
* Sustainable land use and resource management
* Public interest applications that benefit society and the environment

### Ethical Guidelines

While the CC0 license permits broad use, we strongly encourage users to consider the ethical implications of their work. These ethical guidelines are advisory and do not impose legally enforceable restrictions. We request that users:

**Act Responsibly:**
* Consider privacy implications when analyzing specific locations
* Respect the rights and dignity of affected communities
* Be mindful of potential dual-use concerns

**Be Transparent:**
* Accurately represent the data's characteristics (annual resolution, 10m spatial resolution)
* Acknowledge limitations in your applications
* Do not misrepresent TESSERA's capabilities

**Support Positive Impact:**
* Consider how your work contributes to societal benefit
* Engage with affected communities when appropriate
* Share findings that advance public knowledge

### Data Characteristics

Users should understand that TESSERA provides:
* **Annual temporal resolution** — data represents yearly summaries, not real-time or high-frequency monitoring
* **10-meter spatial resolution** — suitable for landscape-scale analysis
* **Spectral-temporal embeddings** — compressed representations, not raw imagery

Please accurately represent these characteristics in your work.

### Community Standards

We encourage responsible use and welcome community feedback. If you have concerns about potential applications or suggestions for improving these guidelines, please contact us.

We reserve the right to update these guidelines based on community input and emerging considerations, though such updates do not retroactively affect the CC-0 license under which data is released.

### Contact

For questions or feedback: Email sk818@cam.ac.uk

---

*Last updated: February 25, 2026*

