#!/usr/bin/env bash
#
# tessera_processor.sh — Sentinel-1 & Sentinel-2 Parallel Processing Pipeline
# Dependencies: bash ≥4, GNU coreutils, Python ≥3.7
# Usage: bash s1_s2_downloader.sh

# set -euo pipefail
set -u

# Resolve this script's own directory so the sibling *.py processors can be
# located regardless of where the script is invoked from (no `cd` required).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#######################################
# USER CONFIGURABLE PARAMETERS
#######################################

# === Basic Configuration ===
 : "${INPUT_TIFF:=/absolute/path/to/your/data_dir/roi.tiff}"
 : "${OUT_DIR:=/absolute/path/to/your/data_dir}"

 : "${TEMP_DIR:=/absolute/path/to/your/temp_dir}"     # Temporary file directory
export TEMP_DIR

mkdir -p "$OUT_DIR"

# Python environment path
 : "${PYTHON_ENV:=/absolute/path/to/your/python_env/bin/python}"

# === Sentinel-1 & Sentinel-2 Processing Configuration ===
 : "${YEAR:=2022}" # Range [2017-2025]
 : "${RESOLUTION:=10.0}"  # Resolution of the input TIFF, also the output resolution (meters)

# === Data Source Configuration ===
# mpc: Microsoft Planetary Computer (sentinel-1-rtc, sentinel-2-l2a)
# aws: AWS Open Data backends (S1=OPERA RTC-S1 via CMR links, S2=Earth-search sentinel-2-l2a COGs)
 : "${DATA_SOURCE:=mpc}"        # choices: mpc/aws

# Optional: override processing date range (useful for quick tests)
# Example:
# START_TIME_OVERRIDE="2025-01-01"
# END_TIME_OVERRIDE="2025-01-15"
 : "${START_TIME_OVERRIDE:=}"
 : "${END_TIME_OVERRIDE:=}"

# === Sentinel-1 Configuration ===
S1_ENABLED=true                    # Enable S1 processing
S1_PARTITIONS=12                   # Number of S1 parallel partitions
S1_TOTAL_WORKERS=12                # Total number of S1 Dask workers
S1_WORKER_MEMORY=4                 # Memory per S1 worker (GB)
S1_CHUNKSIZE=1024                  # S1 stackstac chunk size
S1_ORBIT_STATE="both"              # Orbit state: ascending/descending/both
S1_MIN_COVERAGE=0.01               # Minimum valid pixel coverage for S1 (%) set this to 0.01 to mitigate the tiling artefact!
S1_RESOLUTION=$RESOLUTION          # S1 output resolution (meters)
: "${S1_OVERWRITE:=true}"          # Overwrite existing S1 files


# === Sentinel-2 Configuration ===
S2_ENABLED=true                    # Enable S2 processing
S2_PARTITIONS=24                   # Number of S2 parallel partitions
S2_TOTAL_WORKERS=24                # Total number of S2 Dask workers
S2_WORKER_MEMORY=4                 # Memory per S2 worker (GB)
S2_CHUNKSIZE=1024                  # S2 stackstac chunk size
S2_MAX_CLOUD=100                   # Maximum cloud coverage for S2 (%) set this to 100 to mitigate the tiling artefact!
S2_RESOLUTION=$RESOLUTION          # S2 output resolution (meters)
S2_MIN_COVERAGE=0.01               # Minimum valid pixel coverage for S2 (%) set this to 0.01 to mitigate the tiling artefact!
: "${S2_OVERWRITE:=true}"          # Overwrite existing S2 files


# === System Configuration ===
DEBUG=false                        # Enable debug mode
LOG_INTERVAL=10                    # Progress update interval (seconds)

#######################################
# Internal Variables (Be Careful to Modify)
#######################################
START_TIME="${START_TIME_OVERRIDE:-${YEAR}-01-01}"
END_TIME="${END_TIME_OVERRIDE:-${YEAR}-12-31}"

SCRIPT_START_TIME=$(date +%s)
SCRIPT_NAME=$(basename "$0")
LOG_DIR="${OUT_DIR}/logs"
MAIN_LOG="${LOG_DIR}/tessera_processing_$(date +%Y%m%d_%H%M%S).log"
 : "${S1_RAW_SUBDIR:=data_sar_raw}"   # S1 raw output subdir under OUT_DIR (read by the stacker)
 : "${S2_RAW_SUBDIR:=data_raw}"       # S2 raw output subdir under OUT_DIR (read by the stacker)
S1_OUTPUT="${OUT_DIR}/${S1_RAW_SUBDIR}"
S2_OUTPUT="${OUT_DIR}/${S2_RAW_SUBDIR}"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

#######################################
# Logging Function
#######################################
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${timestamp} ${BLUE}[INFO]${NC} $message" | tee -a "$MAIN_LOG"
            ;;
        SUCCESS)
            echo -e "${timestamp} ${GREEN}[SUCCESS]${NC} $message" | tee -a "$MAIN_LOG"
            ;;
        WARNING)
            echo -e "${timestamp} ${YELLOW}[WARNING]${NC} $message" | tee -a "$MAIN_LOG"
            ;;
        ERROR)
            echo -e "${timestamp} ${RED}[ERROR]${NC} $message" | tee -a "$MAIN_LOG"
            ;;
        HEADER)
            echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}" | tee -a "$MAIN_LOG"
            echo -e "${timestamp} ${PURPLE}$message${NC}" | tee -a "$MAIN_LOG"
            echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}" | tee -a "$MAIN_LOG"
            ;;
    esac
}

#######################################
# Utility Functions
#######################################
# Convert timestamp to seconds
time_to_seconds() {
    date -d "$1" +%s
}

# Convert seconds to date format
seconds_to_date() {
    date -d "@$1" +"%Y-%m-%d"
}

# Format duration
format_duration() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60 ))
    local seconds=$((duration % 60))
    echo "${hours}h ${minutes}m ${seconds}s"
}

# Calculate time partitions
calculate_partitions() {
    local start_sec=$(time_to_seconds "$START_TIME")
    local end_sec=$(time_to_seconds "$END_TIME")
    local partitions=$1
    # Guard: if requested partitions exceed available days, clamp partitions to day count
    local total_days=$(( (end_sec - start_sec) / 86400 + 1 ))
    if [[ $total_days -lt 1 ]]; then
        total_days=1
    fi
    if [[ $partitions -gt $total_days ]]; then
        # IMPORTANT: do not write to stdout here (stdout is parsed by mapfile). Send warning to stderr + main log only.
        echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${YELLOW}[WARNING]${NC} Requested partitions ($partitions) exceed available days ($total_days) for range $START_TIME..$END_TIME; clamping partitions to $total_days" \
          | tee -a "$MAIN_LOG" >&2
        partitions=$total_days
    fi
    local total_seconds=$((end_sec - start_sec + 86400))
    local seconds_per_partition=$((total_seconds / partitions))
    
    for ((i=0; i<partitions; i++)); do
        local partition_start_sec
        local partition_end_sec
        
        if [[ $i -eq 0 ]]; then
            partition_start_sec=$start_sec
        else
            partition_start_sec=$((start_sec + i * seconds_per_partition))
        fi
        
        if [[ $i -eq $((partitions - 1)) ]]; then
            partition_end_sec=$end_sec
        else
            partition_end_sec=$((start_sec + (i + 1) * seconds_per_partition - 86400))
        fi
        
        local p_start=$(seconds_to_date $partition_start_sec)
        local p_end=$(seconds_to_date $partition_end_sec)
        
        echo "$p_start,$p_end"
    done
}

# Calculate workers per partition
calculate_workers_per_partition() {
    local total_workers=$1
    local partitions=$2
    local workers_per_partition=$((total_workers / partitions))
    local remaining_workers=$((total_workers % partitions))
    
    for ((i=0; i<partitions; i++)); do
        if [[ $i -lt $remaining_workers ]]; then
            echo $((workers_per_partition + 1))
        else
            echo $workers_per_partition
        fi
    done
}

# Generate partition ID
generate_partition_id() {
    local p_start=$1
    local p_end=$2
    local p_index=$3
    local prefix=$4
    
    local start_year=$(date -d "${p_start}" +%Y)
    local start_month=$(date -d "${p_start}" +%m)
    local start_day=$(date -d "${p_start}" +%d)
    local end_year=$(date -d "${p_end}" +%Y)
    local end_month=$(date -d "${p_end}" +%m)
    local end_day=$(date -d "${p_end}" +%d)
    
    if [[ "$start_year" == "$end_year" && "$start_month" == "$end_month" ]]; then
        if [[ "$start_day" == "$end_day" ]]; then
            echo "${prefix}_P${p_index}_${start_year}${start_month}${start_day}"
        else
            echo "${prefix}_P${p_index}_${start_year}${start_month}${start_day}-${end_day}"
        fi
    elif [[ "$start_year" == "$end_year" ]]; then
        echo "${prefix}_P${p_index}_${start_year}${start_month}${start_day}-${end_month}${end_day}"
    else
        echo "${prefix}_P${p_index}_${start_year}${start_month}${start_day}-${end_year}${end_month}${end_day}"
    fi
}

#######################################
# Monitoring Function
#######################################
monitor_processes() {
    local -n pids=$1
    local -n partition_ids=$2
    local -n start_times=$3
    local -n completed=$4
    local -n failed=$5
    local prefix=$6
    
    declare -A finished_pids
    
    while true; do
        local all_done=true
        local running_count=0
        
        for i in "${!pids[@]}"; do
            local pid=${pids[i]}
            if [[ -n "${finished_pids[$pid]:-}" ]]; then
                continue
            fi
            
            if kill -0 $pid 2>/dev/null; then
                all_done=false
                running_count=$((running_count + 1))
            else
                local end_time=$(date +%s)
                local duration=$((end_time - ${start_times[i]}))
                local partition_id=${partition_ids[i]}
                
                wait $pid
                local exit_code=$?
                
                finished_pids[$pid]=1
                
                if [ $exit_code -eq 0 ]; then
                    log SUCCESS "$prefix partition $partition_id completed ($(format_duration $duration))"
                    completed+=("$partition_id")
                else
                    log ERROR "$prefix partition $partition_id failed with exit code $exit_code ($(format_duration $duration))"
                    failed+=("$partition_id")
                fi
            fi
        done
        
        if ! $all_done; then
            echo -ne "\r${CYAN}[$(date '+%H:%M:%S')]${NC} $prefix: ${running_count}/${#pids[@]} partitions running...${NC}"
        fi
        
        if $all_done; then
            echo -ne "\r\033[K"  # Clear the line
            break
        fi
        
        sleep $LOG_INTERVAL
    done
}

#######################################
# Processing Functions
#######################################
process_sentinel1() {
    log HEADER "Starting Sentinel-1 Processing"
    
    mkdir -p "$S1_OUTPUT"
    
    # Generate partitions
    mapfile -t s1_partitions < <(calculate_partitions $S1_PARTITIONS)
    local s1_actual_partitions=${#s1_partitions[@]}
    mapfile -t s1_workers_per_partition < <(calculate_workers_per_partition $S1_TOTAL_WORKERS $s1_actual_partitions)
    
    log INFO "S1 Configuration:"
    log INFO "  - Partitions: $S1_PARTITIONS"
    log INFO "  - Total Workers: $S1_TOTAL_WORKERS"
    log INFO "  - Worker Memory: ${S1_WORKER_MEMORY}GB"
    log INFO "  - Orbit State: $S1_ORBIT_STATE"
    log INFO "  - Output: $S1_OUTPUT"
    
    # Start parallel processing
    local s1_pids=()
    local s1_partition_ids=()
    local s1_start_times=()
    local s1_completed=()
    local s1_failed=()
    
    for i in "${!s1_partitions[@]}"; do
        p_range=${s1_partitions[i]}
        p_start=${p_range%,*}
        p_end=${p_range#*,}
        workers=${s1_workers_per_partition[i]}
        partition_id=$(generate_partition_id "$p_start" "$p_end" "$i" "S1")
        
        s1_partition_ids+=("$partition_id")
        
        local overwrite_flag=""
        [[ "$S1_OVERWRITE" == "true" ]] && overwrite_flag="--overwrite"
        
        local debug_flag=""
        [[ "$DEBUG" == "true" ]] && debug_flag="--debug"
        
        $PYTHON_ENV "$SCRIPT_DIR/s1_fast_processor.py" \
            --input_tiff "$INPUT_TIFF" \
            --start_date "$p_start" \
            --end_date "$p_end" \
            --output "$S1_OUTPUT" \
            --orbit_state "$S1_ORBIT_STATE" \
            --data_source "$DATA_SOURCE" \
            --dask_workers "$workers" \
            --worker_memory "$S1_WORKER_MEMORY" \
            --resolution "$S1_RESOLUTION" \
            --chunksize "$S1_CHUNKSIZE" \
            --min_coverage "$S1_MIN_COVERAGE" \
            --partition_id "$partition_id" \
            $overwrite_flag $debug_flag \
            > "$LOG_DIR/${partition_id}.log" 2>&1 &
        
        s1_pids+=($!)
        s1_start_times+=("$(date +%s)")
        
        sleep 2
    done
    
    log INFO "S1: Launched ${#s1_pids[@]} partition processes"
    
    # Monitor processes
    monitor_processes s1_pids s1_partition_ids s1_start_times s1_completed s1_failed "S1"
    
    # Summarize results
    log INFO "S1 Processing Summary:"
    log INFO "  - Total: ${#s1_partitions[@]}"
    log INFO "  - Successful: ${#s1_completed[@]}"
    log INFO "  - Failed: ${#s1_failed[@]}"
    
    if [[ ${#s1_failed[@]} -gt 0 ]]; then
        log WARNING "S1 failed partitions: ${s1_failed[@]}"
        return 1
    fi
    
    return 0
}

process_sentinel2() {
    log HEADER "Starting Sentinel-2 Processing"
    
    mkdir -p "$S2_OUTPUT"
    
    # Generate partitions
    mapfile -t s2_partitions < <(calculate_partitions $S2_PARTITIONS)
    local s2_actual_partitions=${#s2_partitions[@]}
    mapfile -t s2_workers_per_partition < <(calculate_workers_per_partition $S2_TOTAL_WORKERS $s2_actual_partitions)
    
    log INFO "S2 Configuration:"
    log INFO "  - Partitions: $S2_PARTITIONS"
    log INFO "  - Total Workers: $S2_TOTAL_WORKERS"
    log INFO "  - Worker Memory: ${S2_WORKER_MEMORY}GB"
    log INFO "  - Max Cloud: ${S2_MAX_CLOUD}%"
    log INFO "  - Resolution: ${S2_RESOLUTION}m"
    log INFO "  - Output: $S2_OUTPUT"
    
    # Start parallel processing
    local s2_pids=()
    local s2_partition_ids=()
    local s2_start_times=()
    local s2_completed=()
    local s2_failed=()
    
    for i in "${!s2_partitions[@]}"; do
        p_range=${s2_partitions[i]}
        p_start=${p_range%,*}
        p_end=${p_range#*,}
        workers=${s2_workers_per_partition[i]}
        partition_id=$(generate_partition_id "$p_start" "$p_end" "$i" "S2")
        
        s2_partition_ids+=("$partition_id")
        
        local overwrite_flag=""
        [[ "$S2_OVERWRITE" == "true" ]] && overwrite_flag="--overwrite"
        
        local debug_flag=""
        [[ "$DEBUG" == "true" ]] && debug_flag="--debug"
        
        # Add time to S2
        local s2_start="${p_start}T00:00:00"
        local s2_end="${p_end}T23:59:59"
        
        $PYTHON_ENV "$SCRIPT_DIR/s2_fast_processor.py" \
            --input_tiff "$INPUT_TIFF" \
            --start_date "$s2_start" \
            --end_date "$s2_end" \
            --output "$S2_OUTPUT" \
            --max_cloud "$S2_MAX_CLOUD" \
            --data_source "$DATA_SOURCE" \
            --dask_workers "$workers" \
            --worker_memory "$S2_WORKER_MEMORY" \
            --chunksize "$S2_CHUNKSIZE" \
            --resolution "$S2_RESOLUTION" \
            --min_coverage "$S2_MIN_COVERAGE" \
            --partition_id "$partition_id" \
            $overwrite_flag $debug_flag \
            > "$LOG_DIR/${partition_id}.log" 2>&1 &
        
        s2_pids+=($!)
        s2_start_times+=("$(date +%s)")
        
        sleep 2
    done
    
    log INFO "S2: Launched ${#s2_pids[@]} partition processes"
    
    # Monitor processes
    monitor_processes s2_pids s2_partition_ids s2_start_times s2_completed s2_failed "S2"
    
    # Summarize results
    log INFO "S2 Processing Summary:"
    log INFO "  - Total: ${#s2_partitions[@]}"
    log INFO "  - Successful: ${#s2_completed[@]}"
    log INFO "  - Failed: ${#s2_failed[@]}"
    
    if [[ ${#s2_failed[@]} -gt 0 ]]; then
        log WARNING "S2 failed partitions: ${s2_failed[@]}"
        return 1
    fi
    
    return 0
}

#######################################
# Main Program
#######################################
main() {
    # Create necessary directories
    mkdir -p "$LOG_DIR" "$TEMP_DIR"
    
    # Initialize log
    echo "" > "$MAIN_LOG"
    
    log HEADER "TESSERA Processing Pipeline Started"
    log INFO "Script: $SCRIPT_NAME"
    log INFO "Start time: $(date)"
    log INFO "Configuration:"
    log INFO "  - Input TIFF: $INPUT_TIFF"
    log INFO "  - Output Directory: $OUT_DIR"
    log INFO "  - Time Range: $START_TIME to $END_TIME"
    log INFO "  - S1 Enabled: $S1_ENABLED"
    log INFO "  - S2 Enabled: $S2_ENABLED"
    
    # Check input file
    if [[ ! -f "$INPUT_TIFF" ]]; then
        log ERROR "Input TIFF file not found: $INPUT_TIFF"
        exit 1
    fi
    
    # Check Python environment
    if [[ ! -x "$PYTHON_ENV" ]]; then
        log ERROR "Python environment not found or not executable: $PYTHON_ENV"
        exit 1
    fi
    
    # Check Python scripts
    if [[ "$S1_ENABLED" == "true" && ! -f "$SCRIPT_DIR/s1_fast_processor.py" ]]; then
        log ERROR "S1 processor script not found: s1_fast_processor.py"
        exit 1
    fi
    
    if [[ "$S2_ENABLED" == "true" && ! -f "$SCRIPT_DIR/s2_fast_processor.py" ]]; then
        log ERROR "S2 processor script not found: s2_fast_processor.py"
        exit 1
    fi
    
    # Processing flags
    local s1_success=true
    local s2_success=true
    local s1_pid=""
    local s2_pid=""
    
    # Start S1 and S2 processing in parallel
    if [[ "$S1_ENABLED" == "true" && "$S2_ENABLED" == "true" ]]; then
        log INFO "Starting parallel processing of Sentinel-1 and Sentinel-2..."
        
        # Start S1 processing in the background
        ( process_sentinel1 ) &
        s1_pid=$!
        
        # Start S2 processing in the background
        ( process_sentinel2 ) &
        s2_pid=$!
        
        # Wait for both processes to complete
        log INFO "Waiting for both Sentinel-1 and Sentinel-2 processing to complete..."
        
        if ! wait $s1_pid; then
            s1_success=false
        fi
        
        if ! wait $s2_pid; then
            s2_success=false
        fi
        
    elif [[ "$S1_ENABLED" == "true" ]]; then
        log INFO "Processing only Sentinel-1..."
        if ! process_sentinel1; then
            s1_success=false
        fi
        
    elif [[ "$S2_ENABLED" == "true" ]]; then
        log INFO "Processing only Sentinel-2..."
        if ! process_sentinel2; then
            s2_success=false
        fi
        
    else
        log ERROR "Neither S1 nor S2 processing is enabled!"
        exit 1
    fi
    
    # Final summary
    log HEADER "Processing Complete"
    
    local total_duration=$(($(date +%s) - SCRIPT_START_TIME))
    log INFO "Total processing time: $(format_duration $total_duration)"
    
    # Generate final report
    {
        echo ""
        echo "════════════════════════════════════════════════════════════════════"
        echo "                    TESSERA PROCESSING REPORT                        "
        echo "════════════════════════════════════════════════════════════════════"
        echo ""
        echo "Processing Period: $START_TIME to $END_TIME"
        echo "Total Duration: $(format_duration $total_duration)"
        echo ""
        
        if [[ "$S1_ENABLED" == "true" ]]; then
            echo "Sentinel-1 Processing:"
            echo "  Status: $([ "$s1_success" == "true" ] && echo "SUCCESS ✓" || echo "FAILED ✗")"
            echo "  Output: $S1_OUTPUT"
            echo ""
        fi
        
        if [[ "$S2_ENABLED" == "true" ]]; then
            echo "Sentinel-2 Processing:"
            echo "  Status: $([ "$s2_success" == "true" ] && echo "SUCCESS ✓" || echo "FAILED ✗")"
            echo "  Output: $S2_OUTPUT"
            echo ""
        fi
        
        echo "Log Files:"
        echo "  Main Log: $MAIN_LOG"
        echo "  Partition Logs: $LOG_DIR/"
        echo ""
        echo "════════════════════════════════════════════════════════════════════"
    } | tee -a "$MAIN_LOG"
    
    # Combine all logs
    if [[ "$S1_ENABLED" == "true" ]]; then
        cat "$LOG_DIR"/S1_*.log > "$LOG_DIR/s1_combined.log" 2>/dev/null || true
        log INFO "S1 combined log: $LOG_DIR/s1_combined.log"
    fi
    
    if [[ "$S2_ENABLED" == "true" ]]; then
        cat "$LOG_DIR"/S2_*.log > "$LOG_DIR/s2_combined.log" 2>/dev/null || true
        log INFO "S2 combined log: $LOG_DIR/s2_combined.log"
    fi
    
    # Return status
    if [[ "$s1_success" == "true" && "$s2_success" == "true" ]]; then
        log SUCCESS "All processing completed successfully! 🎉"
        exit 0
    else
        log ERROR "Some processing tasks failed. Please check the logs for details."
        exit 1
    fi
}

# Capture interrupt signal
trap 'log ERROR "Process interrupted by user"; exit 130' INT TERM

# Run main program
main "$@"
