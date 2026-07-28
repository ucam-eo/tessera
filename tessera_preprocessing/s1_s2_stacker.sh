#!/usr/bin/env bash
#
# tessera_processor.sh — Sentinel-1 & Sentinel-2 Parallel Processing Pipeline
# Dependencies: bash ≥4, GNU coreutils, Python ≥3.7
# Usage: bash s1_s2_stacker.sh

# set -euo pipefail
set -u

#######################################
# USER CONFIGURABLE PARAMETERS
#######################################

# === Basic Configuration ===
 : "${BASE_DIR:=/absolute/path/to/your/data_dir}"
OUT_DIR="${BASE_DIR}/data_processed"
DOWNSAMPLE_RATE=1

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

./s1_stack \
  --input-dir "${BASE_DIR}/data_sar_raw" \
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

./s2_stack \
  --input "${BASE_DIR}/data_raw" \
  --output $OUT_DIR \
  --batch-size 16 \
  --cache-level 1 \
  --num-threads 16 \
  --sample-rate $DOWNSAMPLE_RATE

echo "Processing complete. Processed data is available in: $OUT_DIR"
