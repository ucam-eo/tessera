#!/usr/bin/env bash
#
# run_coastal_pipeline.sh -- Orchestrate TESSERA inference for coastal gap-fill
#
# Processes groups from a manifest through five stages:
#   1. Download Sentinel-1 + Sentinel-2 data
#   2. Stack along time dimension
#   3. Patchify (retile into inference tiles)
#   4. QAT inference
#   5. Stitch tiled representations into a single embedding
#
# Usage:
#   bash run_coastal_pipeline.sh --project-dir /path/to/project \
#       --tessera-root /path/to/tessera --python-env /path/to/python [OPTIONS]
#
# See --help for all options.

set -u

# ── Defaults (overridden by CLI flags or env vars) ──────────────────────────

PROJECT_DIR=""
TESSERA_ROOT=""
PYTHON_ENV=""
MANIFEST=""

YEAR="${YEAR:-2024}"
RESOLUTION="${RESOLUTION:-10.0}"
DATA_SOURCE="${DATA_SOURCE:-mpc}"

S1_PARTITIONS="${S1_PARTITIONS:-4}"
S1_WORKERS="${S1_WORKERS:-4}"
S1_WORKER_MEMORY="${S1_WORKER_MEMORY:-2}"
S2_PARTITIONS="${S2_PARTITIONS:-6}"
S2_WORKERS="${S2_WORKERS:-6}"
S2_WORKER_MEMORY="${S2_WORKER_MEMORY:-2}"

GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-512}"
PATCH_SIZE="${PATCH_SIZE:-500}"

MIN_VALID_FRACTION="${MIN_VALID_FRACTION:-0.0}"

SINGLE_GROUP=""
RETRY_FAILED=false
LOW_MEM=false

# ── Colours ─────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC}   $(date '+%H:%M:%S') $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') $1"; }
log_err()  { echo -e "${RED}[ERR]${NC}  $(date '+%H:%M:%S') $1"; }

# ── Argument parsing ────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Required:
  --project-dir DIR      Project root (contains the manifest's data directories)
  --tessera-root DIR     Path to the TESSERA repository
  --python-env PATH      Path to the Python interpreter in the TESSERA conda env

Optional:
  --manifest PATH        Path to manifest.json (default: auto-detected from project-dir)
  --group NAME           Process only this single group
  --retry-failed         Re-run groups that have no stitched_representation.npy,
                         using conservative memory settings for S2 stacking
  --low-mem              Use conservative memory settings for all groups
  --year INT             Sentinel data year (default: $YEAR)
  --gpu-batch-size INT   Batch size for GPU inference (default: $GPU_BATCH_SIZE)
  --patch-size INT       Patch size for retiling (default: $PATCH_SIZE)
  -h, --help             Show this help message

Environment variables (for advanced tuning):
  S1_PARTITIONS, S1_WORKERS, S1_WORKER_MEMORY   Sentinel-1 download parallelism
  S2_PARTITIONS, S2_WORKERS, S2_WORKER_MEMORY   Sentinel-2 download parallelism
  MIN_VALID_FRACTION                             Skip groups below this valid-pixel fraction (default: 0.0)
  YEAR, RESOLUTION, DATA_SOURCE                  Sentinel query parameters
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-dir)    PROJECT_DIR="$2"; shift 2 ;;
        --tessera-root)   TESSERA_ROOT="$2"; shift 2 ;;
        --python-env)     PYTHON_ENV="$2"; shift 2 ;;
        --manifest)       MANIFEST="$2"; shift 2 ;;
        --group)          SINGLE_GROUP="$2"; shift 2 ;;
        --retry-failed)   RETRY_FAILED=true; shift ;;
        --low-mem)        LOW_MEM=true; shift ;;
        --year)           YEAR="$2"; shift 2 ;;
        --gpu-batch-size) GPU_BATCH_SIZE="$2"; shift 2 ;;
        --patch-size)     PATCH_SIZE="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1"; usage 1 ;;
    esac
done

# Validate required args
for var in PROJECT_DIR TESSERA_ROOT PYTHON_ENV; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: --$(echo $var | tr '_' '-' | tr '[:upper:]' '[:lower:]') is required"
        usage 1
    fi
done

PREPROC_DIR="${TESSERA_ROOT}/tessera_preprocessing"
INFER_DIR="${TESSERA_ROOT}/tessera_infer_QAT"

# Auto-detect manifest if not given
if [[ -z "$MANIFEST" ]]; then
    MANIFEST=$(find "$PROJECT_DIR" -maxdepth 3 -name "manifest.json" -path "*/coastal*" | head -1)
    if [[ -z "$MANIFEST" ]]; then
        MANIFEST=$(find "$PROJECT_DIR" -maxdepth 3 -name "manifest.json" | head -1)
    fi
    if [[ -z "$MANIFEST" ]]; then
        echo "ERROR: Could not find manifest.json. Use --manifest to specify."
        exit 1
    fi
    log_info "Auto-detected manifest: $MANIFEST"
fi

GAPS_DIR="$(dirname "$MANIFEST")"

# ── S2 memory profile ──────────────────────────────────────────────────────

s2_stack_args() {
    if [[ "$LOW_MEM" == true ]] || [[ "$RETRY_FAILED" == true ]]; then
        echo "--batch-size 1 --cache-level 0 --num-threads 2 --sample-rate 1"
    else
        echo "--batch-size 4 --cache-level 1 --num-threads 4 --sample-rate 1"
    fi
}

s1_parallel() {
    if [[ "$LOW_MEM" == true ]] || [[ "$RETRY_FAILED" == true ]]; then
        echo "2"
    else
        echo "4"
    fi
}

# ── Group listing ───────────────────────────────────────────────────────────

get_groups() {
    $PYTHON_ENV -c "
import json, sys
with open('${MANIFEST}') as f:
    manifest = json.load(f)
for entry in manifest:
    print(f'{entry[\"group_name\"]} {entry[\"valid_fraction\"]}')
"
}

# ── Process a single group ──────────────────────────────────────────────────

process_group() {
    local GROUP_NAME="$1"
    local DATA_DIR="${GAPS_DIR}/${GROUP_NAME}"
    local ROI_TIFF="${DATA_DIR}/roi.tiff"

    if [[ ! -f "$ROI_TIFF" ]]; then
        log_err "ROI TIFF not found: $ROI_TIFF"
        return 1
    fi

    local FINAL_REPR="${DATA_DIR}/stitched_representation.npy"
    if [[ -f "$FINAL_REPR" ]] && [[ "$RETRY_FAILED" != true ]]; then
        log_info "Skipping $GROUP_NAME -- already has stitched result"
        return 0
    fi

    local TEMP_DIR="${DATA_DIR}/tmp"
    mkdir -p "$TEMP_DIR" "${DATA_DIR}/logs"

    log_info "===== Processing $GROUP_NAME ====="

    # Stage 1: Download
    log_info "Stage 1/5: Downloading Sentinel-1 & Sentinel-2 data..."
    cd "$PREPROC_DIR"

    INPUT_TIFF="$ROI_TIFF" \
    OUT_DIR="$DATA_DIR" \
    TEMP_DIR="$TEMP_DIR" \
    PYTHON_ENV="$PYTHON_ENV" \
    YEAR="$YEAR" \
    RESOLUTION="$RESOLUTION" \
    DATA_SOURCE="$DATA_SOURCE" \
    S1_PARTITIONS="$S1_PARTITIONS" \
    S1_TOTAL_WORKERS="$S1_WORKERS" \
    S1_WORKER_MEMORY="$S1_WORKER_MEMORY" \
    S2_PARTITIONS="$S2_PARTITIONS" \
    S2_TOTAL_WORKERS="$S2_WORKERS" \
    S2_WORKER_MEMORY="$S2_WORKER_MEMORY" \
    bash s1_s2_downloader.sh > "${DATA_DIR}/logs/download.log" 2>&1
    local dl_exit=$?

    if [[ $dl_exit -ne 0 ]]; then
        log_warn "Download had issues (exit $dl_exit) for $GROUP_NAME -- check ${DATA_DIR}/logs/download.log"
        log_info "Continuing anyway (some partitions may have succeeded)..."
    else
        log_ok "Download complete for $GROUP_NAME"
    fi

    # Stage 2: Stack
    log_info "Stage 2/5: Stacking data along time dimension..."
    local PROCESSED_DIR="${DATA_DIR}/data_processed"
    mkdir -p "$PROCESSED_DIR"

    if [[ -d "${DATA_DIR}/data_sar_raw" ]]; then
        ./s1_stack \
            --input-dir "${DATA_DIR}/data_sar_raw" \
            --output-dir "$PROCESSED_DIR" \
            --parallel "$(s1_parallel)" \
            --rate 1 > "${DATA_DIR}/logs/s1_stack.log" 2>&1 || true
        log_ok "S1 stacking done"
    else
        log_warn "No S1 data found, skipping S1 stacking"
    fi

    if [[ -d "${DATA_DIR}/data_raw" ]]; then
        ./s2_stack \
            --input "${DATA_DIR}/data_raw" \
            --output "$PROCESSED_DIR" \
            $(s2_stack_args) > "${DATA_DIR}/logs/s2_stack.log" 2>&1
        local s2_exit=$?
        if [[ $s2_exit -ne 0 ]]; then
            log_err "S2 stacking failed for $GROUP_NAME"
            return 1
        fi
        log_ok "S2 stacking done"
    else
        log_warn "No S2 data found, skipping S2 stacking"
    fi

    # Stage 3: Patchify
    log_info "Stage 3/5: Patchifying into tiles..."
    local RETILED_DIR="${DATA_DIR}/retiled_d_pixel"
    mkdir -p "$RETILED_DIR"

    $PYTHON_ENV dpixel_retiler.py \
        --tiff_path "$ROI_TIFF" \
        --d_pixel_dir "$PROCESSED_DIR" \
        --patch_size "$PATCH_SIZE" \
        --out_dir "$RETILED_DIR" \
        --num_workers 4 \
        --overwrite \
        --block_size 2000 > "${DATA_DIR}/logs/retile.log" 2>&1

    if [[ $? -ne 0 ]]; then
        log_err "Patchification failed for $GROUP_NAME"
        return 1
    fi
    log_ok "Patchification complete"

    # Stage 4: QAT Inference
    log_info "Stage 4/5: Running QAT inference..."
    cd "$INFER_DIR"

    local REPR_DIR="${DATA_DIR}/representation_retiled_qat"
    mkdir -p "$REPR_DIR"

    export BASE_DATA_DIR="$DATA_DIR"
    export PYTHON_ENV="$PYTHON_ENV"
    export CPU_GPU_SPLIT="0:1"
    export MAX_CONCURRENT_PROCESSES_GPU=1
    export GPU_BATCH_SIZE="$GPU_BATCH_SIZE"

    bash infer_all_tiles.sh \
        --tiles-dir "$RETILED_DIR" \
        --output-dir "$REPR_DIR" > "${DATA_DIR}/logs/inference.log" 2>&1

    local infer_exit=$?
    if [[ $infer_exit -ne 0 ]]; then
        log_warn "Inference had issues (exit $infer_exit) for $GROUP_NAME"
    else
        log_ok "Inference complete"
    fi

    # Stage 5: Stitch
    log_info "Stage 5/5: Stitching representation map..."
    local STITCH_DIR="${TESSERA_ROOT}/tessera_infer"
    cd "$STITCH_DIR"

    $PYTHON_ENV stitch_tiled_representation.py \
        --d_pixel_retiled_path "$RETILED_DIR" \
        --representation_retiled_path "$REPR_DIR" \
        --downstream_tiff "$ROI_TIFF" \
        --out_dir "$DATA_DIR" > "${DATA_DIR}/logs/stitch.log" 2>&1

    if [[ $? -ne 0 ]]; then
        log_err "Stitching failed for $GROUP_NAME -- keeping intermediates for debugging"
        return 1
    fi
    log_ok "Stitching complete: ${DATA_DIR}/stitched_representation.npy"

    # Clean up intermediate files
    log_info "Cleaning up intermediate data..."
    rm -rf "$TEMP_DIR"
    rm -rf "${DATA_DIR}/data_sar_raw"
    rm -rf "${DATA_DIR}/data_raw"
    rm -rf "${DATA_DIR}/data_processed"
    rm -rf "${DATA_DIR}/retiled_d_pixel"
    rm -rf "${DATA_DIR}/representation_retiled_qat"
    rm -rf "${DATA_DIR}/logs"
    log_ok "Cleanup complete for $GROUP_NAME"

    cd "$PROJECT_DIR"
}

# ── Main ────────────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"

if [[ -n "$SINGLE_GROUP" ]]; then
    process_group "$SINGLE_GROUP"
    exit $?
fi

log_info "Reading manifest: $MANIFEST"
TOTAL_PROCESSED=0
TOTAL_SKIPPED=0
TOTAL_FAILED=0

while IFS=' ' read -r group_name valid_fraction; do
    # In retry mode, only process groups missing their stitched output
    if [[ "$RETRY_FAILED" == true ]]; then
        if [[ -f "${GAPS_DIR}/${group_name}/stitched_representation.npy" ]]; then
            continue
        fi
        log_info "Retrying $group_name (missing stitched output)..."
    fi

    vf_check=$($PYTHON_ENV -c "print('skip' if $valid_fraction < $MIN_VALID_FRACTION else 'process')")
    if [[ "$vf_check" == "skip" ]]; then
        log_warn "Skipping $group_name (valid fraction ${valid_fraction} below threshold)"
        TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
        continue
    fi

    if process_group "$group_name"; then
        TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))
    else
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
done < <(get_groups)

echo ""
log_info "========================================"
log_info "Pipeline complete!"
log_info "  Processed: $TOTAL_PROCESSED"
log_info "  Skipped:   $TOTAL_SKIPPED"
log_info "  Failed:    $TOTAL_FAILED"
log_info "========================================"
