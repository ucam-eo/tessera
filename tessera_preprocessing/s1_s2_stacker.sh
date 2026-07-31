#!/usr/bin/env bash
#
# tessera_processor.sh — Sentinel-1 & Sentinel-2 Parallel Processing Pipeline
# Dependencies: bash ≥4, GNU coreutils, Python ≥3.7
# Usage: bash s1_s2_stacker.sh

# set -euo pipefail
set -u

# Resolve this script's own directory so the sibling s1_stack / s2_stack
# binaries can be located regardless of where the script is invoked from
# (no `cd` required).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#######################################
# USER CONFIGURABLE PARAMETERS
#######################################

# === Basic Configuration ===
 : "${BASE_DIR:=/absolute/path/to/your/data_dir}"
#BASE_DIR="/scratch/zf281/tessera/data/cambridge/output/2024"
 : "${S1_RAW_SUBDIR:=data_sar_raw}"      # S1 raw input subdir under BASE_DIR (matches the downloader output)
 : "${S2_RAW_SUBDIR:=data_raw}"          # S2 raw input subdir under BASE_DIR (matches the downloader output)
 : "${PROCESSED_SUBDIR:=data_processed}" # stacked output subdir under BASE_DIR
OUT_DIR="${BASE_DIR}/${PROCESSED_SUBDIR}"
 : "${DOWNSAMPLE_RATE:=1}"

mkdir -p "$OUT_DIR"

# S1 stacking
: '
s1_stack 0.1.0
Process Sentinel-1 data for a single tile

USAGE:
    s1_stack [OPTIONS] --input-dir <input-dir> --output-dir <output-dir>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -i, --input-dir <input-dir>      Input directory (where TIFF files are)
    -o, --output-dir <output-dir>    Output directory (where processed NPY files will go)
    -p, --parallel <parallel>        Number of parallel processes to use [default: 8]
    -r, --rate <rate>                Downsampling rate (e.g., 10 means take every 10th pixel) [default: 10]
'

"$SCRIPT_DIR/s1_stack" \
  --input-dir "${BASE_DIR}/${S1_RAW_SUBDIR}" \
  --output-dir $OUT_DIR \
  --parallel 16 \
  --rate $DOWNSAMPLE_RATE

# S2 stacking
: '
s2_stack 0.1.0
Process Sentinel-2 data for a single tile

USAGE:
    s2_stack [OPTIONS] --input <input-dir> --output <output-dir>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -b, --batch-size <batch-size>      Number of time slices to process in parallel [default: 5]
    -c, --cache-level <cache-level>    Cache strategy (0=minimal, 1=moderate, 2=aggressive) [default: 1]
    -i, --input <input-dir>            Input directory (where raw tiff files are organized in band folders)
    -n, --num-threads <num-threads>    Number of threads (default=10) to use for parallel tasks [default: 10]
    -o, --output <output-dir>          Output directory (where processed NPY files will go)
    -r, --sample-rate <sample-rate>    Downsample rate (default=10) [default: 10]
'

"$SCRIPT_DIR/s2_stack" \
  --input "${BASE_DIR}/${S2_RAW_SUBDIR}" \
  --output $OUT_DIR \
  --batch-size 16 \
  --cache-level 1 \
  --num-threads 16 \
  --sample-rate $DOWNSAMPLE_RATE

echo "Processing complete. Processed data is available in: $OUT_DIR"
