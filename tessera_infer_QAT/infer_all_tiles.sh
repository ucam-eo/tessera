#!/bin/bash
#
# infer_all_tiles.sh - Hybrid CPU/GPU inference script for QAT multi-tile processing
#
# Usage:
#   bash infer_all_tiles.sh
#

############### This needs to be modified to your environment ###############
# Main directory where preprocessed tiles and outputs are located
 : "${BASE_DATA_DIR:=/absolute_path_to_data_dir}"

# Python environment with required dependencies
 : "${PYTHON_ENV:=/absolute/path/to/your/python_env/bin/python}"

# Base directory for logfiles
 : "${BASE_LOG_DIR:=.}"

# CPU:GPU split ratio (Format: CPU:GPU)
# Examples: "1:1" (balanced), "1:0" (CPU only), "0:1" (GPU only)
 : "${CPU_GPU_SPLIT:=1:0}"

# Max concurrent tile processes for CPU/GPU
 : "${MAX_CONCURRENT_PROCESSES_CPU:=20}"
 : "${MAX_CONCURRENT_PROCESSES_GPU:=1}"

# CPU cores to use
TOTAL_CPU_CORES=$(nproc)
AVAILABLE_CORES=$((TOTAL_CPU_CORES / 2))

# Batch and worker settings
CPU_BATCH_SIZE=1024
CPU_NUM_WORKERS=0
GPU_BATCH_SIZE=1024
GPU_NUM_WORKERS=4

# Precision and AMX behavior
CPU_ENABLE_BF16=false
GPU_ENABLE_BF16=false
CPU_DISABLE_AMX=false

# Logging behavior
VERBOSE_GPU=true
LOG_INTERVAL=5
###########################################################################

TILES_DIR="${BASE_DATA_DIR}/retiled_d_pixel"
OUTPUT_DIR="${BASE_DATA_DIR}/representation_retiled_qat"
CONFIG_FILE="configs/multi_tile_infer_config.py"
PYTHON_SCRIPT="src/multi_tile_infer.py"
CHECKPOINT_PATH="checkpoints/best_model_fsdp_20250608_220648_QAT.pt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --cpu-gpu-split)
            CPU_GPU_SPLIT="$2"
            shift 2
            ;;
        --max-cpu)
            MAX_CONCURRENT_PROCESSES_CPU="$2"
            shift 2
            ;;
        --max-gpu)
            MAX_CONCURRENT_PROCESSES_GPU="$2"
            shift 2
            ;;
        --tiles-dir)
            TILES_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --verbose-gpu)
            VERBOSE_GPU="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_START=$(date +%s)

# Terminal colors
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
CYAN="\033[36m"
RESET="\033[0m"

log_header() { echo -e "\n${BOLD}${BLUE}==== $1 ====${RESET}" >&2; }
log_info() { echo -e "${CYAN}[INFO]${RESET} $1" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS]${RESET} $1" >&2; }
log_warning() { echo -e "${YELLOW}[WARNING]${RESET} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${RESET} $1" >&2; }

format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    if [[ $hours -gt 0 ]]; then
        printf "%02dh:%02dm:%02ds" $hours $minutes $secs
    elif [[ $minutes -gt 0 ]]; then
        printf "%02dm:%02ds" $minutes $secs
    else
        printf "%02ds" $secs
    fi
}

calculate_time() {
    local start_time=$1
    local end_time=$2
    local duration=$((end_time - start_time))
    format_time $duration
}

# Parse CPU:GPU split ratio
IFS=':' read -r CPU_RATIO GPU_RATIO <<< "$CPU_GPU_SPLIT"
CPU_RATIO=${CPU_RATIO:-1}
GPU_RATIO=${GPU_RATIO:-1}
TOTAL_RATIO=$((CPU_RATIO + GPU_RATIO))

log_info "Total CPU cores: $TOTAL_CPU_CORES, Using: $AVAILABLE_CORES"
log_info "CPU:GPU split ratio = $CPU_RATIO:$GPU_RATIO (total: $TOTAL_RATIO)"

declare -a CPU_PIDS
declare -a GPU_PIDS

cleanup() {
    log_warning "Received interrupt signal. Cleaning up processes..."
    for pid in "${CPU_PIDS[@]}"; do
        ps -p "$pid" > /dev/null 2>&1 && kill -9 "$pid" 2>/dev/null || true
    done
    for pid in "${GPU_PIDS[@]}"; do
        ps -p "$pid" > /dev/null 2>&1 && kill -9 "$pid" 2>/dev/null || true
    done
    log_warning "All processes terminated. Exiting."
    exit 1
}
trap cleanup SIGINT SIGTERM

log_header "SETUP DIRECTORIES"
mkdir -p "$OUTPUT_DIR"
mkdir -p "src/tile_lists"
mkdir -p "${BASE_LOG_DIR}/logs"
log_success "Created necessary directories"

log_header "SCANNING TILES"
log_info "Tile directory: $TILES_DIR"
log_info "Output directory: $OUTPUT_DIR"

TILES=($(find "$TILES_DIR" -maxdepth 1 -mindepth 1 -type d | sort))
NUM_TILES=${#TILES[@]}

if [[ $NUM_TILES -eq 0 ]]; then
    log_error "No tiles found in $TILES_DIR"
    exit 1
fi

log_success "Found $NUM_TILES tiles total"

log_header "FILTERING TILES"
TILES_TO_PROCESS=()
ALREADY_PROCESSED=0

for tile_path in "${TILES[@]}"; do
    tile_name=$(basename "$tile_path")
    output_repr="$OUTPUT_DIR/${tile_name}.npy"
    output_scales="$OUTPUT_DIR/${tile_name}_scales.npy"
    if [[ -f "$output_repr" && -f "$output_scales" ]]; then
        ALREADY_PROCESSED=$((ALREADY_PROCESSED + 1))
    else
        TILES_TO_PROCESS+=("$tile_path")
    fi
done

NUM_TILES_TO_PROCESS=${#TILES_TO_PROCESS[@]}
log_success "Found $NUM_TILES_TO_PROCESS tiles to process (skipping $ALREADY_PROCESSED already done)"

if [[ $NUM_TILES_TO_PROCESS -eq 0 ]]; then
    log_success "No tiles need processing. All done!"
    exit 0
fi

log_header "RESOURCE ALLOCATION"
if [[ "$CPU_RATIO" -eq 0 ]]; then
    CPU_TILES_COUNT=0
    GPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
elif [[ "$GPU_RATIO" -eq 0 ]]; then
    CPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
    GPU_TILES_COUNT=0
else
    CPU_TILES_COUNT=$(( NUM_TILES_TO_PROCESS * CPU_RATIO / TOTAL_RATIO ))
    GPU_TILES_COUNT=$(( NUM_TILES_TO_PROCESS - CPU_TILES_COUNT ))
fi

CPU_TILES=("${TILES_TO_PROCESS[@]:0:$CPU_TILES_COUNT}")
GPU_TILES=("${TILES_TO_PROCESS[@]:$CPU_TILES_COUNT}")

if [[ $GPU_TILES_COUNT -gt 0 ]]; then
    if ! command -v nvidia-smi > /dev/null 2>&1; then
        log_warning "nvidia-smi not found, moving all GPU tiles to CPU"
        CPU_TILES=("${TILES_TO_PROCESS[@]}")
        CPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
        GPU_TILES=()
        GPU_TILES_COUNT=0
    else
        AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
        if [[ $AVAILABLE_GPUS -eq 0 ]]; then
            log_warning "No GPUs found, moving all GPU tiles to CPU"
            CPU_TILES=("${TILES_TO_PROCESS[@]}")
            CPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
            GPU_TILES=()
            GPU_TILES_COUNT=0
        else
            NUM_GPU_PROCESSES=$(( MAX_CONCURRENT_PROCESSES_GPU < AVAILABLE_GPUS ? MAX_CONCURRENT_PROCESSES_GPU : AVAILABLE_GPUS ))
            TILES_PER_GPU=$((GPU_TILES_COUNT / NUM_GPU_PROCESSES))
            REMAINDER=$((GPU_TILES_COUNT % NUM_GPU_PROCESSES))
            for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
                START=$((i * TILES_PER_GPU))
                END=$(((i + 1) * TILES_PER_GPU))
                if [[ $i -lt $REMAINDER ]]; then END=$((END + 1)); fi
                if [[ $i -ge $REMAINDER ]]; then START=$((START + REMAINDER)); else START=$((START + i)); fi

                TILE_LIST="src/tile_lists/tiles_gpu_${i}.json"
                echo -n "[" > "$TILE_LIST"
                for ((j=START; j<END && j<GPU_TILES_COUNT; j++)); do
                    [[ $j -gt $START ]] && echo -n "," >> "$TILE_LIST"
                    echo -n "\"${GPU_TILES[$j]}\"" >> "$TILE_LIST"
                done
                echo "]" >> "$TILE_LIST"
                log_info "Created tile list for GPU $i: $TILE_LIST"
            done
        fi
    fi
fi

if [[ $CPU_TILES_COUNT -gt 0 ]]; then
    NUM_CPU_PROCESSES=$(( CPU_TILES_COUNT < MAX_CONCURRENT_PROCESSES_CPU ? CPU_TILES_COUNT : MAX_CONCURRENT_PROCESSES_CPU ))
    THREADS_PER_CPU_PROCESS=$(( AVAILABLE_CORES / NUM_CPU_PROCESSES ))
    [[ $THREADS_PER_CPU_PROCESS -lt 1 ]] && THREADS_PER_CPU_PROCESS=1
else
    NUM_CPU_PROCESSES=0
fi

log_info "CPU tiles: $CPU_TILES_COUNT, GPU tiles: $GPU_TILES_COUNT"
log_info "CPU processes: $NUM_CPU_PROCESSES, CPU threads/process: $THREADS_PER_CPU_PROCESS"

CPU_BF16_FLAG=""
GPU_BF16_FLAG=""
CPU_AMX_FLAG=""
[[ "$CPU_ENABLE_BF16" == "true" ]] && CPU_BF16_FLAG="--enable_bf16"
[[ "$GPU_ENABLE_BF16" == "true" ]] && GPU_BF16_FLAG="--enable_bf16"
[[ "$CPU_DISABLE_AMX" == "true" ]] && CPU_AMX_FLAG="--disable_amx"

launch_gpu_processes() {
    if [[ $GPU_TILES_COUNT -le 0 ]]; then return; fi
    log_header "GPU PROCESSING"
    for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
        TILE_LIST="src/tile_lists/tiles_gpu_${i}.json"
        LOG_FILE="${BASE_LOG_DIR}/logs/infer_qat_gpu_${i}.log"
        VERBOSE_GPU_FLAG=""
        [[ "$VERBOSE_GPU" == "true" ]] && VERBOSE_GPU_FLAG="--verbose_gpu"
        CMD="$PYTHON_ENV $PYTHON_SCRIPT --config $CONFIG_FILE --mode gpu --gpu_id $i --tile_list $TILE_LIST --process_id $i --checkpoint_path $CHECKPOINT_PATH --output_dir $OUTPUT_DIR --batch_size $GPU_BATCH_SIZE --num_workers $GPU_NUM_WORKERS --log_interval $LOG_INTERVAL $GPU_BF16_FLAG $VERBOSE_GPU_FLAG --simplified_logging"
        log_info "Launching GPU process $i"
        nohup $CMD > "$LOG_FILE" 2>&1 &
        GPU_PIDS[$i]=$!
        log_info "GPU $i PID: ${GPU_PIDS[$i]}"
        sleep 2
    done
}

start_cpu_process() {
    local tile_path="$1"
    local process_id="$2"
    local log_file="${BASE_LOG_DIR}/logs/infer_qat_cpu_${process_id}.log"
    > "$log_file"
    $PYTHON_ENV "$PYTHON_SCRIPT" \
        --config "$CONFIG_FILE" \
        --mode cpu \
        --tile_path "$tile_path" \
        --process_id "$process_id" \
        --num_threads "$THREADS_PER_CPU_PROCESS" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$CPU_BATCH_SIZE" \
        --num_workers "$CPU_NUM_WORKERS" \
        --log_interval "$LOG_INTERVAL" \
        --log_level "INFO" \
        $CPU_BF16_FLAG \
        $CPU_AMX_FLAG \
        --simplified_logging > "$log_file" 2>&1 &
    echo $!
}

process_cpu_tiles() {
    if [[ $CPU_TILES_COUNT -le 0 ]]; then return; fi
    log_header "CPU PROCESSING"
    local started=0
    local completed=0
    local active_pids=()
    declare -A pid_to_tile

    local to_start=$(( CPU_TILES_COUNT < MAX_CONCURRENT_PROCESSES_CPU ? CPU_TILES_COUNT : MAX_CONCURRENT_PROCESSES_CPU ))
    for ((i=0; i<to_start; i++)); do
        pid=$(start_cpu_process "${CPU_TILES[$i]}" "$i")
        active_pids+=("$pid")
        CPU_PIDS+=("$pid")
        pid_to_tile[$pid]=$i
        started=$((started + 1))
    done

    while [[ $completed -lt $CPU_TILES_COUNT ]]; do
        local new_active=()
        for pid in "${active_pids[@]}"; do
            if ps -p "$pid" > /dev/null 2>&1; then
                new_active+=("$pid")
            else
                wait "$pid" 2>/dev/null || true
                completed=$((completed + 1))
                if [[ $started -lt $CPU_TILES_COUNT ]]; then
                    new_pid=$(start_cpu_process "${CPU_TILES[$started]}" "$started")
                    new_active+=("$new_pid")
                    CPU_PIDS+=("$new_pid")
                    pid_to_tile[$new_pid]=$started
                    started=$((started + 1))
                fi
            fi
        done
        active_pids=("${new_active[@]}")
        log_info "CPU progress: $completed/$CPU_TILES_COUNT complete, ${#active_pids[@]} running"
        sleep 5
    done
}

monitor_gpu_processes() {
    if [[ $GPU_TILES_COUNT -le 0 || ${#GPU_PIDS[@]} -eq 0 ]]; then return; fi
    log_header "GPU MONITORING"
    local finished=0
    while [[ $finished -lt $NUM_GPU_PROCESSES ]]; do
        finished=0
        for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
            if ! ps -p "${GPU_PIDS[$i]}" > /dev/null 2>&1; then
                finished=$((finished + 1))
            fi
        done
        log_info "GPU progress: $finished/$NUM_GPU_PROCESSES complete"
        sleep 10
    done
}

launch_gpu_processes
if [[ $CPU_TILES_COUNT -gt 0 ]]; then
    process_cpu_tiles &
    CPU_CONTROLLER_PID=$!
fi
if [[ $GPU_TILES_COUNT -gt 0 ]]; then
    monitor_gpu_processes &
    GPU_MONITOR_PID=$!
fi

[[ -n "$CPU_CONTROLLER_PID" ]] && wait "$CPU_CONTROLLER_PID" 2>/dev/null || true
[[ -n "$GPU_MONITOR_PID" ]] && wait "$GPU_MONITOR_PID" 2>/dev/null || true

log_header "RESULTS SUMMARY"
EXPECTED_FILES=$NUM_TILES_TO_PROCESS
ACTUAL_REPR=$(find "$OUTPUT_DIR" -name "*.npy" ! -name "*_scales.npy" | wc -l)
ACTUAL_SCALES=$(find "$OUTPUT_DIR" -name "*_scales.npy" | wc -l)

log_info "Expected outputs: $EXPECTED_FILES repr + $EXPECTED_FILES scales"
log_info "Found outputs: $ACTUAL_REPR repr + $ACTUAL_SCALES scales"

if [[ $ACTUAL_REPR -eq $NUM_TILES && $ACTUAL_SCALES -eq $NUM_TILES ]]; then
    log_success "All tiles have been processed successfully!"
else
    log_warning "Some tiles may not have been processed correctly"
fi

SCRIPT_END=$(date +%s)
TOTAL_DURATION=$(calculate_time $SCRIPT_START $SCRIPT_END)
log_header "COMPLETED IN $TOTAL_DURATION"
log_info "Detailed logs:"
log_info "  - CPU logs: ${BASE_LOG_DIR}/logs/infer_qat_cpu_*.log"
log_info "  - GPU logs: ${BASE_LOG_DIR}/logs/infer_qat_gpu_*.log"

