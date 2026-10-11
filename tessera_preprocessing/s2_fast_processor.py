#!/usr/bin/env python3
"""
s2_fast_processor.py — Sentinel-2 L2A Fast Download & ROI Mosaic (Optimized Version)
Updated: 2025-05-21
Supports flexible parallel partition processing with comprehensive error handling, timeout control, and memory optimization
Features: SCL band preserves original values during quality assessment, ensuring SCL for each date is generated based on actual observation data
Optimization: Uses efficient vectorized operations instead of pixel-by-pixel processing, significantly improving SCL processing and band generation performance
"""

from __future__ import annotations
import os, sys, argparse, logging, datetime, time, warnings, signal
from pathlib import Path
import multiprocessing
from contextlib import contextmanager
import concurrent.futures
import uuid
import tempfile
import shutil
import gc
import random

import numpy as np
import psutil, rasterio, xarray as xr, rioxarray
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject
from rasterio.transform import from_origin
import pystac_client, planetary_computer, stackstac

import dask
from dask.distributed import Client, LocalCluster, performance_report, wait

# ▶ distributed version compatibility
try:
    from distributed.comm.core import CommClosedError
except ImportError:
    from distributed import CommClosedError

warnings.filterwarnings("ignore", category=RuntimeWarning, module="dask.core")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*The array is being split into many small chunks.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in true_divide.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in log10.*")

# ─── Constants ──────────────────────────────────────────────────────────────────────
# NOTE: Band/asset keys differ by data source:
# - MPC (planetarycomputer sentinel-2-l2a): assets are band codes like B02/B03/.../SCL
# - AWS (earth-search sentinel-2-l2a): assets are names like blue/green/.../scl
MPC_BAND_MAPPING = {
    "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08",
    "B11": "swir16", "B12": "swir22",
    "SCL": "scl",
}
AWS_BAND_MAPPING = {
    "blue": "blue", "green": "green", "red": "red",
    "rededge1": "rededge1", "rededge2": "rededge2", "rededge3": "rededge3",
    "nir": "nir", "nir08": "nir08",
    "swir16": "swir16", "swir22": "swir22",
    "scl": "scl",
}

# Will be set in main() based on --data_source
BAND_MAPPING = MPC_BAND_MAPPING
S2_BANDS = list(BAND_MAPPING.keys())
DATA_SOURCE = "mpc"            # set in main(); controls baseline-offset handling
BASELINE_CUTOFF = datetime.datetime(2022, 1, 25)
BASELINE_OFFSET = 1000

# SCL invalid value list (cloud-free/shadow-free/non-water etc. are valid)
SCL_INVALID = {0, 1, 2, 3, 8, 9, np.nan}

# SCL value descriptions for logging
SCL_DESCRIPTIONS = {
    0: "No data",
    1: "Saturated or defective",
    2: "Dark area pixels",
    3: "Cloud shadows",
    4: "Vegetation",
    5: "Not vegetated",
    6: "Water",
    7: "Unclassified",
    8: "Cloud medium probability",
    9: "Cloud high probability",
    10: "Thin cirrus",
    11: "Snow"
}

# Valid coverage threshold (skip processing below this value)
MIN_VALID_COVERAGE = 5.0  # percentage

# Temporary file directory setting (default uses system temp directory)
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())

# Timeout settings (seconds)
PROCESS_TIMEOUT = 25 * 60  # Overall processing timeout
DAY_TIMEOUT = 15 * 60       # Single day processing timeout
ITEM_TIMEOUT = 10 * 60      # Single item processing timeout
BAND_TIMEOUT = 8 * 60      # Single band processing timeout
SCL_BAND_TIMEOUT = 2 * 60   # SCL band processing timeout
SCL_MAX_ATTEMPTS = 2        # SCL processing maximum attempts

# Network request retry configuration
MAX_RETRIES = 5
RETRY_BACKOFF_FACTOR = 1.5  # Exponential backoff factor

# Parallel processing configuration
DEFAULT_MAX_WORKERS = 2     # Default number of threads for processing items, avoid too many concurrent requests

# ─── Timeout Control ──────────────────────────────────────────────────────────────────
class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
    """Timeout context manager"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutException(f"Operation timeout ({seconds} seconds)")
    
    # Check if running in main thread (Unix signals can only be handled in main thread)
    import threading
    if threading.current_thread() is not threading.main_thread():
        # If not main thread, just yield without setting signal
        yield
        return
    
    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore original signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ─── CLI ───────────────────────────────────────────────────────────────────────
def get_args():
    P = argparse.ArgumentParser("Fast Sentinel-2 L2A Processor (Optimized Parallel Edition)")
    P.add_argument("--input_tiff",   required=True, help="ROI mask or template raster")
    P.add_argument("--start_date",   required=True, help="Start date (YYYY-MM-DD[THH:MM:SS]) - inclusive")
    P.add_argument("--end_date",     required=True, help="End date (YYYY-MM-DD[THH:MM:SS]) - inclusive")
    P.add_argument("--output",       default="sentinel2_output", help="Output directory")
    P.add_argument("--max_cloud",    type=float, default=90, help="Maximum cloud cover percentage")
    P.add_argument("--dask_workers", type=int,   default=8, help="Number of Dask workers for this partition")
    P.add_argument("--worker_memory",type=int,   default=16, help="Memory per worker in GB")
    P.add_argument("--chunksize",    type=int,   default=1024, help="stackstac x/y chunk size")
    P.add_argument("--resolution",   type=float,   default=10.0, help="Output resolution (meters)")
    P.add_argument("--overwrite",    action="store_true", help="Overwrite existing files")
    P.add_argument("--debug",        action="store_true", help="Output debug logs")
    P.add_argument("--min_coverage", type=float, default=MIN_VALID_COVERAGE,
                   help="Minimum valid pixel coverage percentage")
    P.add_argument("--partition_id", default="unknown",
                   help="Partition ID (for log identification)")
    P.add_argument("--temp_dir",     default=TEMP_DIR,
                   help="Temporary file storage directory, default uses system temp directory")
    P.add_argument("--data_source", default="mpc", choices=["mpc", "aws"],
                   help="Data source backend: mpc (Planetary Computer) or aws (Earth-search on AWS)")
    return P.parse_args()

# ─── logging ──────────────────────────────────────────────────────────────────
def setup_logging(debug: bool, out_dir: Path, partition_id: str):
    """Setup logging with partition ID identifier"""
    fmt = f"%(asctime)s [{partition_id}] [%(levelname)s] %(message)s"
    lvl = logging.DEBUG if debug else logging.INFO
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(lvl)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(fmt)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (partition-specific log file)
    file_handler = logging.FileHandler(
        out_dir / f"s2_{partition_id}_detail.log", 
        "a", 
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_sys(partition_id: str):
    m = psutil.virtual_memory()
    logging.info(f"[{partition_id}] System info - CPU {os.cpu_count()} | "
                 f"RAM {m.total/1e9:.1f} GB (free {m.available/1e9:.1f} GB)")

def fmt_bbox(b):
    return f"{b[0]:.5f},{b[1]:.5f} ⇢ {b[2]:.5f},{b[3]:.5f}"

# ─── Dask ─────────────────────────────────────────────────────────────────────
def make_client(req_workers:int, req_mem:int, partition_id: str):
    """Create Dask client with partition-specific dashboard port"""
    total_mem = psutil.virtual_memory().total / 1e9
    workers = min(req_workers, os.cpu_count(),
                  max(1, int(total_mem // (req_mem*1.2))))
    if workers < req_workers:
        logging.warning(f"⚠️  worker count {req_workers}→{workers} (resource limitation)")
    
    # Automatically assign dashboard port for different partitions
    # Use hash of partition ID to determine port, ensuring same ID always uses same port
    # Limit ports to 8780-8899 range
    port_base = 8780
    port_range = 120
    dashboard_port = port_base + (hash(partition_id) % port_range)
    
    # Configure Dask performance optimization options
    dask_config = {
        "distributed.worker.memory.target": 0.75,  # Lower to reduce memory pressure
        "distributed.worker.memory.spill": 0.85,   # Lower to reduce memory pressure
        "distributed.worker.memory.pause": 0.95,
        "array.slicing.split_large_chunks": True,  # Optimize large array slicing
        "optimization.fuse.active": True,         # Activate fusion optimization
        "optimization.fuse.ave-width": 4          # Accelerate dask graph optimization
    }
    
    dask.config.set(dask_config)
    
    cluster = LocalCluster(
        n_workers         = workers,
        threads_per_worker= 4,
        processes         = True,
        memory_limit      = f"{req_mem}GB",
        dashboard_address = f":{dashboard_port}",
        silence_logs      = "ERROR",
    )
    
    cli = Client(cluster, asynchronous=False)
    logging.info(f"[{partition_id}] Dask dashboard → {cli.dashboard_link}")
    return cli

# ─── ROI & Mask ────────────────────────────────────────────────────────────────
def load_roi(tiff: Path, partition_id: str, out_resolution_m: float):
    """
    Load ROI bounds + CRS from the input TIFF, but build an OUTPUT template grid at out_resolution_m.
    The ROI mask is resampled (nearest) into that OUTPUT grid so all downstream mosaics use
    the same extent but fixed resolution (e.g. 10m).
    """
    with rasterio.open(tiff) as src:
        if src.crs is None:
            raise ValueError(f"ROI has no CRS: {tiff}")
        bbox_src = src.bounds
        bbox_ll = transform_bounds(
            src.crs,
            "EPSG:4326",
            bbox_src.left,
            bbox_src.bottom,
            bbox_src.right,
            bbox_src.top,
            densify_pts=21,
        )

        # Read ROI mask in source grid
        mask_src = (src.read(1) > 0).astype(np.uint8)

        # Build OUTPUT grid: same bounds, forced pixel size out_resolution_m
        left, bottom, right, top = bbox_src.left, bbox_src.bottom, bbox_src.right, bbox_src.top
        res = float(out_resolution_m)
        width = int(np.ceil((right - left) / res))
        height = int(np.ceil((top - bottom) / res))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid output grid computed from bounds/res: width={width} height={height}")
        transform = from_origin(left, top, res, res)

        tpl = dict(crs=src.crs, transform=transform, width=width, height=height)
        bbox_proj = rasterio.coords.BoundingBox(left=left, bottom=bottom, right=right, top=top)

        # Resample ROI mask into OUTPUT grid (nearest)
        mask_np = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=mask_src,
            destination=mask_np,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=src.crs,
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )
        
    # Check ROI size and log info
    roi_size_mb = (mask_np.size * mask_np.itemsize) / (1024 * 1024)
    logging.info(f"[{partition_id}] ROI/output grid (CRS={tpl['crs']}): {tpl['width']}×{tpl['height']} @ {out_resolution_m}m ({roi_size_mb:.2f} MB)")
    logging.info(f"[{partition_id}] ROI bbox proj: {fmt_bbox(bbox_proj)}")
    logging.info(f"[{partition_id}] ROI bbox lon/lat: {fmt_bbox(bbox_ll)}")
    
    return tpl, bbox_proj, bbox_ll, mask_np


def tpl_resolution_tuple(tpl) -> tuple:
    """
    Return (xres, yres) from the ROI template transform.
    Using a tuple prevents subtle grid mismatch when ROI pixels are not
    perfectly square, which can otherwise manifest as 'zoomed' outputs.
    """
    tr = tpl["transform"]
    return (abs(float(tr.a)), abs(float(tr.e)))

def mask_to_xr(mask_np, tpl):
    """Convert mask to xarray object"""
    da = xr.DataArray(mask_np, dims=("y", "x"))
    return da.rio.write_crs(tpl["crs"]).rio.write_transform(tpl["transform"])

# ─── STAC ─────────────────────────────────────────────────────────────────────
def search_items(bbox_ll, date_range:str, max_cloud, partition_id: str):
    """
    Search STAC items with enhanced exception handling and retry logic
    """
    # Parse start and end times
    start_date, end_date = date_range.split("/")
    
    # Parse end time and add one second to ensure end time is included
    try:
        end_dt = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00').replace(' ', 'T'))
        end_dt_plus = end_dt + datetime.timedelta(seconds=1)
        search_date_range = f"{start_date}/{end_dt_plus.isoformat()}"
    except ValueError:
        # If time format parsing fails, use original date range
        logging.warning(f"[{partition_id}] Unable to parse end date format, using original range: {date_range}")
        search_date_range = date_range
    
    logging.info(f"[{partition_id}] STAC search date range: {search_date_range}")
    
    # Add retry logic
    retries = 0
    max_retries = MAX_RETRIES
    retry_delay = 1  # Initial delay 1 second
    
    while retries <= max_retries:
        try:
            # Backend switch:
            # - mpc: planetarycomputer STAC with signing
            # - aws: earth-search STAC (public)
            if getattr(search_items, "_data_source", "mpc") == "aws":
                cat = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
                q = cat.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox_ll,
                    datetime=search_date_range,
                    query={"eo:cloud_cover": {"lt": max_cloud}},
                )
            else:
                cat = pystac_client.Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                    modifier=planetary_computer.sign_inplace,
                )
                q = cat.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox_ll,
                    datetime=search_date_range,
                    query={"eo:cloud_cover": {"lt": max_cloud}},
                )
            items = list(q.get_items())
            logging.info(f"[{partition_id}] STAC found {len(items)} items (cloud < {max_cloud}%)")
            if items:
                b = np.array([it.bbox for it in items])
                union = [b[:,0].min(), b[:,1].min(), b[:,2].max(), b[:,3].max()]
                logging.info(f"[{partition_id}] All item union lon/lat: {fmt_bbox(union)}")
            return items
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logging.error(f"[{partition_id}] STAC search failed (attempt {retries}/{max_retries+1}): {e}")
                raise
            
            # Calculate exponential backoff delay
            retry_delay = min(60, retry_delay * RETRY_BACKOFF_FACTOR)  # Max delay 60 seconds
            jitter = random.uniform(0.8, 1.2)  # Add random jitter
            actual_delay = retry_delay * jitter
            
            logging.warning(f"[{partition_id}] STAC search failed (attempt {retries}/{max_retries+1}): {e}, retrying after {actual_delay:.1f}s...")
            time.sleep(actual_delay)

def group_by_date(items, partition_id: str):
    """Group items by date"""
    g = {}
    for it in items:
        d = it.properties["datetime"][:10]
        g.setdefault(d, []).append(it)
    logging.info(f"[{partition_id}] ⇒ {len(g)} observation days")
    return dict(sorted(g.items()))

# ─── baseline correction ─────────────────────────────────────────────────────────────
def harmonize_arr(arr: np.ndarray, date_key: str):
    """Subtract the PB-04.00 BOA_ADD_OFFSET (1000) introduced on 2022-01-25.

    Source-specific behaviour:
      - MPC (planetarycomputer sentinel-2-l2a): COGs are stored RAW, i.e. with
        the +1000 offset still present for post-cutoff acquisitions. We must
        subtract 1000 here to harmonise with pre-04.00 scenes.
      - AWS (Element84 earth-search sentinel-2-l2a): the offset has already
        been removed at COG-generation time (per Element84 earth-search README:
        "As part of the conversion to COGs, the offset has been applied"). The
        data is therefore already harmonised; subtracting again would shift
        post-2022-01-25 acquisitions ~1000 too low for bands whose value
        exceeds the offset (matches the empirical ~1000 MPC↔AWS gap reported
        by users).
    """
    if DATA_SOURCE == "aws":
        return arr
    if datetime.datetime.strptime(date_key, "%Y-%m-%d") > BASELINE_CUTOFF:
        # Handle NaN values to avoid warnings
        valid_mask = ~np.isnan(arr) & (arr >= BASELINE_OFFSET)
        np.subtract(arr, BASELINE_OFFSET, out=arr, where=valid_mask)
    return arr

# ─── Memory Check ──────────────────────────────────────────────────────────────────
def check_memory_requirements(shape, dtype=np.uint16):
    """Check if array memory requirements are reasonable"""
    try:
        # Calculate required memory (GB)
        element_size = np.dtype(dtype).itemsize
        total_elements = np.prod(shape)
        memory_gb = (total_elements * element_size) / (1024**3)
        
        logging.debug(f"Calculating memory requirements: shape{shape}, type{dtype}, size{memory_gb:.2f}GB")
        
        # Get current available memory
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        # Use 50% of current available memory as threshold
        threshold_gb = min(available_gb * 0.5, 32)  # Not exceeding 32GB
        
        if memory_gb > threshold_gb:
            logging.warning(f"⚠️  Memory requirement {memory_gb:.2f}GB exceeds available threshold {threshold_gb:.2f}GB, skipping processing")
            return False
        return True
    except (OverflowError, ValueError) as e:
        logging.warning(f"⚠️  Memory calculation error: {e}, skipping processing")
        return False

# ─── GeoTIFF Write ──────────────────────────────────────────────────────────────
def write_tiff(np_arr, out_path: Path, tpl, dtype, metadata=None):
    """Write GeoTIFF with optimized compression and configuration, adding metadata"""
    # Handle NaN values
    if np.isnan(np_arr).any():
        np_arr = np.nan_to_num(np_arr, nan=0)
        
    profile = dict(driver="GTiff", dtype=dtype, count=1,
                   width=tpl["width"], height=tpl["height"],
                   crs=tpl["crs"], transform=tpl["transform"],
                   tiled=True,
                   blockxsize=256, blockysize=256,
                   nodata=0)
    
    # Write file
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np_arr.astype(dtype, copy=False), 1)
        
        # Add metadata
        if metadata:
            dst.update_tags(**metadata)

# ─── Validate TIFF ─────────────────────────────────────────────────────────────
def validate_tiff(file_path, expected_shape, expected_crs, expected_transform):
    """Validate if TIFF file is valid"""
    try:
        with rasterio.open(file_path) as src:
            # Check basic attributes
            if src.shape != expected_shape:
                logging.warning(f"Validation failed: {file_path} shape mismatch. Expected {expected_shape}, got {src.shape}")
                return False
            
            if src.crs != expected_crs:
                logging.warning(f"Validation failed: {file_path} CRS mismatch. Expected {expected_crs}, got {src.crs}")
                return False
            
            # Check data existence (through statistics, avoid reading entire array)
            stats = [src.statistics(i) for i in range(1, src.count + 1)]
            if any(s.max == 0 and s.min == 0 for s in stats):
                logging.warning(f"Validation failed: {file_path} all bands are zero")
                return False
            
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            logging.debug(f"TIFF validation passed: {file_path}, shape={src.shape}, size={file_size_mb:.2f}MB")
            return True
            
    except Exception as e:
        logging.error(f"Error validating TIFF {file_path}: {e}")
        return False

# ─── SCL Quality Assessment ─────────────────────────────────────────────────────────
def is_valid_scl(scl_arr):
    """Determine if SCL value is valid observation (non-cloud/shadow/water etc.)"""
    # Vectorized operation: fill nan with 0, then check if in invalid values list
    return ~np.isin(np.nan_to_num(scl_arr, nan=0), list(SCL_INVALID - {np.nan}))

def process_scl(scl_arr, roi_mask, partition_id="unknown"):
    """
    Process SCL data, highly optimized vectorized version
    Returns:
        valid_mask: validity mask (boolean array)
        tile_selection: selected tile index array (for subsequent band processing)
        valid_pct: valid coverage percentage
    """
    # Get array shape and handle single tile and multi-tile cases
    if len(scl_arr.shape) == 3:
        n_tiles, scl_height, scl_width = scl_arr.shape
    else:
        scl_height, scl_width = scl_arr.shape
        n_tiles = 1
        scl_arr = scl_arr.reshape(1, scl_height, scl_width)
    
    # Ensure ROI mask is boolean type
    roi_mask = roi_mask.astype(bool)
    roi_height, roi_width = roi_mask.shape
    
    # Check shape matching, if not matching, adjust SCL data
    if scl_height != roi_height or scl_width != roi_width:
        logging.warning(f"[{partition_id}] Detected shape mismatch: SCL shape{(scl_height, scl_width)} differs from ROI shape{(roi_height, roi_width)}")
        
        # Determine final shape to use (take smaller values to avoid index overflow)
        use_height = min(scl_height, roi_height)
        use_width = min(scl_width, roi_width)
        
        logging.info(f"[{partition_id}] Using common area: {use_height}x{use_width}")
        
        # Crop data to common area
        scl_arr = scl_arr[:, :use_height, :use_width]
        roi_mask = roi_mask[:use_height, :use_width]
    
    # Calculate valid mask for each tile - vectorized operation
    valid_mask = is_valid_scl(scl_arr)
    
    # Calculate valid pixel positions and ROI pixel count
    roi_pixel_count = np.sum(roi_mask)
    
    # Create tile selection array, initialize to -1 (no valid tile)
    tile_selection = np.full(roi_mask.shape, -1, dtype=np.int8)
    
    # Optimized vectorized algorithm for selecting first valid tile
    # 1. Create a boolean mask indicating which pixels have been assigned a valid tile
    assigned = np.zeros(roi_mask.shape, dtype=bool)
    
    # 2. Loop through each tile (this loop cannot be completely avoided as we need to select first valid tile in order)
    for tile_idx in range(n_tiles):
        # 2.1 Find pixels that are valid in current tile and within ROI and not yet assigned
        # valid_mask[tile_idx] is the valid mask for current tile
        # roi_mask is the ROI area mask
        # ~assigned is mask for pixels not yet assigned a tile
        current_valid = valid_mask[tile_idx] & roi_mask & ~assigned
        
        # 2.2 Assign current tile_idx to current valid pixels in tile_selection
        tile_selection[current_valid] = tile_idx
        
        # 2.3 Update assigned flag
        assigned |= current_valid
    
    # Calculate valid coverage - number of pixels assigned a valid tile divided by total ROI pixels
    valid_pixel_count = np.sum(tile_selection >= 0)
    valid_pct = 100.0 * valid_pixel_count / roi_pixel_count if roi_pixel_count > 0 else 0.0
    
    logging.info(f"[{partition_id}] SCL processing result: Valid pixels in ROI {valid_pixel_count}/{roi_pixel_count}, coverage {valid_pct:.2f}%")
    
    return valid_mask, tile_selection, valid_pct

def create_scl_mosaic(scl_arr, tile_selection, roi_mask, target_shape, date_key=None, partition_id="unknown"):
    """
    Create SCL mosaic based on tile selection results, preserving original SCL values - vectorized version
    """
    try:
        # Get shape information
        if len(scl_arr.shape) == 3:
            n_tiles, arr_height, arr_width = scl_arr.shape
        else:
            arr_height, arr_width = scl_arr.shape
            n_tiles = 1
            scl_arr = scl_arr.reshape(1, arr_height, arr_width)
        
        target_height, target_width = target_shape
        
        # Create target size result array (initial value 0, meaning no data)
        result = np.zeros(target_shape, dtype=np.uint8)
        
        # Determine common area size
        common_height = min(arr_height, roi_mask.shape[0], target_height, tile_selection.shape[0])
        common_width = min(arr_width, roi_mask.shape[1], target_width, tile_selection.shape[1])
        
        # Crop to common area
        roi_crop = roi_mask[:common_height, :common_width]
        tile_sel_crop = tile_selection[:common_height, :common_width]
        
        # Vectorized implementation: use valid tile indices within ROI range to select SCL values
        # 1. Create a valid ROI mask indicating which pixels need processing
        valid_roi = (roi_crop & (tile_sel_crop >= 0))
        
        if np.any(valid_roi):
            # 2. Get coordinates of valid ROI
            y_coords, x_coords = np.where(valid_roi)
            
            # 3. Get tile indices corresponding to these coordinates
            tile_indices = tile_sel_crop[y_coords, x_coords]
            
            # 4. Create mapping to extract corresponding values from scl_arr
            result[y_coords, x_coords] = scl_arr[tile_indices, y_coords, x_coords]
        
        # Count SCL value distribution
        unique_values, unique_counts = np.unique(result, return_counts=True)
        value_counts = dict(zip(unique_values, unique_counts))
        
        # Log SCL value distribution
        if date_key:
            total_pixels = np.sum(roi_mask)
            logging.info(f"[{partition_id}] SCL value distribution statistics ({date_key}):")
            for val in sorted(value_counts.keys()):
                count = value_counts[val]
                desc = SCL_DESCRIPTIONS.get(val, "Unknown")
                percent = 100 * count / total_pixels if total_pixels > 0 else 0
                logging.info(f"[{partition_id}]   SCL value {val} ({desc}): {count} pixels ({percent:.2f}%)")
        
        return result
    except Exception as e:
        logging.error(f"[{partition_id}] SCL mosaic creation failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logging.debug(traceback.format_exc())
        raise

# ─── Smart Mosaic ──────────────────────────────────────────────────────────────────
def smart_mosaic(data_arr, tile_selection, roi_mask, partition_id="unknown"):
    """
    Smart mosaic based on tile_selection, using best tiles determined in SCL processing - vectorized version
    """
    try:
        # Single tile case return directly
        if len(data_arr.shape) < 3 or data_arr.shape[0] == 1:
            result = data_arr[0] if len(data_arr.shape) == 3 else data_arr
            
            # Check shape matching
            if result.shape != roi_mask.shape:
                logging.debug(f"[{partition_id}] Single tile mosaic shape mismatch: data{result.shape}, ROI mask{roi_mask.shape}")
                
                # Determine common size
                common_height = min(result.shape[0], roi_mask.shape[0])
                common_width = min(result.shape[1], roi_mask.shape[1])
                
                # Create target size result array
                final_result = np.zeros(roi_mask.shape, dtype=result.dtype)
                
                # Crop data and mask
                result_cropped = result[:common_height, :common_width] 
                roi_mask_cropped = roi_mask[:common_height, :common_width]
                
                # Apply ROI mask - vectorized operation
                final_result[:common_height, :common_width] = result_cropped * roi_mask_cropped
                return final_result
            
            # Apply ROI mask - vectorized operation
            return result * roi_mask
        
        # Get dimensions
        n_tiles, data_height, data_width = data_arr.shape
        
        # Ensure ROI mask is boolean type
        roi_mask = roi_mask.astype(bool)
        
        # Create output array (using original ROI size)
        result = np.zeros(roi_mask.shape, dtype=data_arr.dtype)
        
        # Determine common area size
        common_height = min(data_height, roi_mask.shape[0], tile_selection.shape[0])
        common_width = min(data_width, roi_mask.shape[1], tile_selection.shape[1])
        
        # Crop to common area
        roi_crop = roi_mask[:common_height, :common_width]
        tile_sel_crop = tile_selection[:common_height, :common_width]
        
        # Vectorized implementation: use valid tile indices within ROI range to select data values
        # 1. Create a valid ROI mask indicating which pixels need processing
        valid_roi = (roi_crop & (tile_sel_crop >= 0))
        
        if np.any(valid_roi):
            # 2. Get coordinates of valid ROI
            y_coords, x_coords = np.where(valid_roi)
            
            # 3. Get tile indices corresponding to these coordinates
            tile_indices = tile_sel_crop[y_coords, x_coords]
            
            # 4. Extract corresponding values from data_arr
            result[y_coords, x_coords] = data_arr[tile_indices, y_coords, x_coords]
        
        # For ROI pixels without valid tiles, use random selection logic
        # 1. Create mask for pixels within ROI but without valid tiles
        invalid_roi = (roi_crop & (tile_sel_crop < 0))
        
        if np.any(invalid_roi):
            # 2. Get these coordinates
            y_coords_invalid, x_coords_invalid = np.where(invalid_roi)
            
            # 3. Randomly select a tile for each invalid pixel
            random_tiles = np.random.randint(0, n_tiles, size=len(y_coords_invalid))
            
            # 4. Fill these pixels using randomly selected tiles
            result[y_coords_invalid, x_coords_invalid] = data_arr[random_tiles, y_coords_invalid, x_coords_invalid]
        
        return result
    except Exception as e:
        logging.error(f"[{partition_id}] Smart mosaic failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logging.debug(traceback.format_exc())
        raise

# ─── Process Single Band ───────────────────────────────────────────────────────────
def process_band(items, band_name, date_key, tpl, bbox_proj, mask_np, tile_selection,
               res, chunksize, out_path, partition_id="unknown", retries=3):
    """Process single band with retry and error handling, handle shape mismatch issues, skip SCL band"""
    t0 = time.time()

    # SCL band has been processed in quality assessment phase, skip directly
    if band_name == "SCL":
        logging.info(f"[{partition_id}]     Band {band_name} already processed in quality assessment phase, skipping")
        return True

    logging.info(f"[{partition_id}]     Processing band {band_name}")
   
   # Check if output path already exists
    if out_path.exists():
        if validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.info(f"[{partition_id}]     {band_name} valid file already exists, skipping")
            return True
        else:
            logging.warning(f"[{partition_id}]     {band_name} file exists but invalid, reprocessing")
            out_path.unlink()
   
    # Retry loop
    for attempt in range(retries + 1):
        try:
            with timeout_handler(BAND_TIMEOUT):
                # Use stackstac.stack to load single band
                assets = [band_name]
                
                # Bilinear for spectral bands so the native-20m bands (B05/B06/
                # B07/B8A/B11/B12) are smoothly upsampled to the 10m output grid
                # instead of being block-replicated. SCL takes a separate code
                # path (`process_scl_assessment_and_generation`) which keeps
                # Resampling.nearest because SCL values are categorical class
                # IDs and must never be averaged.
                da = stackstac.stack(
                    items=items,
                    assets=assets,
                    resolution=tpl_resolution_tuple(tpl),
                    epsg=tpl["crs"].to_epsg(),
                    bounds=bbox_proj,
                    chunksize=chunksize,
                    rescale=False,
                    resampling=Resampling.bilinear
                )
                
                # Flatten useless dimensions but keep multi-item dimension
                item_dim = None
                for dim in da.dims:
                    if dim not in ('band', 'x', 'y'):
                        if da.sizes[dim] > 1:
                            item_dim = dim  # Find multi-item dimension
                        elif da.sizes[dim] == 1:
                            da = da.squeeze(dim, drop=True)
                
                # Extract band data
                band_da = da.sel(band=band_name)
                
                # Convert to numpy array for processing
                if item_dim:
                    # Multiple items case
                    band_arr = band_da.values
                    
                    # Log and output shape info for debugging
                    if logging.getLogger().level <= logging.DEBUG:
                        logging.debug(f"[{partition_id}]     {band_name} array shape: {band_arr.shape}, ROI shape: {mask_np.shape}")
                    
                    # Check if array size is reasonable
                    if not check_memory_requirements(band_arr.shape, band_arr.dtype):
                        logging.warning(f"[{partition_id}]     {band_name} array too large, skipping")
                        return False
                    
                    # Apply SCL-based smart mosaic - vectorized version
                    if tile_selection is not None:
                        arr = smart_mosaic(band_arr, tile_selection, mask_np, partition_id)
                    else:
                        # If no valid tile_selection, use random selection
                        logging.warning(f"[{partition_id}]     No valid tile_selection, using random tile selection")
                        n_tiles = band_arr.shape[0]
                        
                        # Create random selection matrix - vectorized implementation
                        random_selection = np.random.randint(0, n_tiles, size=mask_np.shape)
                        arr = np.zeros(mask_np.shape, dtype=band_arr.dtype)
                        
                        # Calculate common area
                        common_height = min(mask_np.shape[0], band_arr.shape[1])
                        common_width = min(mask_np.shape[1], band_arr.shape[2])
                        
                        # Create index matrix
                        for i in range(n_tiles):
                            # Find pixels that should use current tile
                            use_tile_i = (random_selection[:common_height, :common_width] == i) & (mask_np[:common_height, :common_width] > 0)
                            if np.any(use_tile_i):
                                # Direct assignment using boolean indexing - vectorized operation
                                arr[:common_height, :common_width][use_tile_i] = band_arr[i, :common_height, :common_width][use_tile_i]
                    
                    # Check if output shape matches target
                    if arr.shape != (tpl["height"], tpl["width"]):
                        logging.warning(f"[{partition_id}]     Mosaic result shape {arr.shape} does not match target shape {(tpl['height'], tpl['width'])}, adjusting")
                        # Create target size array
                        full_arr = np.zeros((tpl["height"], tpl["width"]), dtype=arr.dtype)
                        # Copy common area
                        h = min(arr.shape[0], tpl["height"])
                        w = min(arr.shape[1], tpl["width"])
                        full_arr[:h, :w] = arr[:h, :w]
                        arr = full_arr
                    
                    # Apply baseline correction
                    harmonize_arr(arr, date_key)
                    
                else:
                    # Single item case
                    band_arr = band_da.values
                    
                    # Log and output shape info for debugging
                    if logging.getLogger().level <= logging.DEBUG:
                        logging.debug(f"[{partition_id}]     {band_name} array shape: {band_arr.shape}, ROI shape: {mask_np.shape}")
                    
                    # Check if array size is reasonable
                    if not check_memory_requirements(band_arr.shape, band_arr.dtype):
                        logging.warning(f"[{partition_id}]     {band_name} array too large, skipping")
                        return False
                    
                    # Check shape matching
                    if band_arr.shape != mask_np.shape:
                        logging.warning(f"[{partition_id}]     Data shape {band_arr.shape} does not match ROI shape {mask_np.shape}, adjusting")
                        # Create target size array
                        arr = np.zeros((tpl["height"], tpl["width"]), dtype=band_arr.dtype)
                        # Copy common area
                        h = min(band_arr.shape[0], tpl["height"])
                        w = min(band_arr.shape[1], tpl["width"])
                        # Apply mask to common area - vectorized operation
                        mask_crop = mask_np[:h, :w]
                        arr[:h, :w] = band_arr[:h, :w] * mask_crop
                    else:
                        # Apply mask - vectorized operation
                        arr = band_arr * mask_np
                    
                    # Apply baseline correction
                    harmonize_arr(arr, date_key)
                
                # Create metadata
                metadata = {
                    "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    "DATE_ACQUIRED": date_key,
                    "BAND_NAME": band_name,
                    "ITEMS_COUNT": len(items)
                }
                
                # Write GeoTIFF
                dtype = "uint16"  # SCL has been skipped, only process other bands here
                write_tiff(arr, out_path, tpl, dtype, metadata)
                
                # Validate output file
                if not validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                    logging.error(f"[{partition_id}]     ✗ Band {band_name} validation failed")
                    if out_path.exists():
                        out_path.unlink()
                    continue
                
                logging.info(f"[{partition_id}]     ✓ {band_name:9s}  "
                            f"{os.path.getsize(out_path)/1e6:.2f} MB, time {time.time()-t0:.1f}s")
                
                # Successfully completed
                return True
                
        except TimeoutException as e:
            if attempt < retries:
                retry_delay = min(30, (attempt + 1) * 2)  # Retry delay, max 30 seconds
                logging.warning(f"[{partition_id}]     Band {band_name} processing timeout, retrying after {retry_delay}s ({attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"[{partition_id}]     ✗ Band {band_name} processing timeout: {e}")
                return False
                
        except Exception as e:
            if attempt < retries:
                retry_delay = min(30, (attempt + 1) * 2)  # Retry delay, max 30 seconds
                logging.warning(f"[{partition_id}]     Band {band_name} processing error: {e}, retrying after {retry_delay}s ({attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"[{partition_id}]     ✗ Band {band_name} processing failed: {e}")
                return False

        # All retries failed
        return False

# ─── Process SCL Assessment and Band Generation ───────────────────────────────────────────────────────────
def process_scl_assessment_and_generation(items, date_key, tpl, bbox_proj, mask_np, res, chunksize,
                                      min_coverage, out_root, overwrite, partition_id="unknown"):
    """
    Process SCL band, perform quality assessment and generate output file with actual SCL values - optimized vectorized version
    """
    t0 = time.time()
    logging.info(f"[{partition_id}]   Processing SCL band for quality assessment and generating SCL output file")

    # Build SCL output path (key differs by backend: SCL vs scl)
    scl_key = "scl" if BAND_MAPPING is AWS_BAND_MAPPING else "SCL"
    scl_out_name = BAND_MAPPING[scl_key]
    scl_dir = out_root / scl_out_name
    scl_dir.mkdir(parents=True, exist_ok=True)
    scl_out_path = scl_dir / f"{date_key}_mosaic.tiff"

    # A saved SCL mosaic contains classes, not the source index for each pixel.
    # Recompute selection from the source SCLs when a day needs processing;
    # process_day already skips fully completed days and retains valid bands.
    if not overwrite and scl_out_path.exists():
        logging.info(f"[{partition_id}]   Recomputing source selection from SCL assets for incomplete day")

    # Determine SCL asset key for this backend
    scl_asset_key = "scl" if BAND_MAPPING is AWS_BAND_MAPPING else "SCL"

    # Check if SCL assets are included in items
    if not all(scl_asset_key in item.assets for item in items):
        logging.warning(f"[{partition_id}]   Some items missing SCL assets, trying to use only available SCL")
        # Filter items with SCL assets
        scl_items = [item for item in items if scl_asset_key in item.assets]
        if not scl_items:
            logging.warning(f"[{partition_id}]   All items missing SCL assets, cannot perform quality assessment!")
            # Return failure result
            return False, 0.0, None
        items = scl_items

    try:
        with timeout_handler(SCL_BAND_TIMEOUT):
            # Use stackstac.stack to load SCL band
            da = stackstac.stack(
                items=items,
                assets=[scl_asset_key],
                resolution=tpl_resolution_tuple(tpl),
                epsg=tpl["crs"].to_epsg(),
                bounds=bbox_proj,
                chunksize=chunksize,
                rescale=False,
                resampling=Resampling.nearest
            )
            
            # Flatten useless dimensions but keep multi-item dimension
            item_dim = None
            for dim in da.dims:
                if dim not in ('band', 'x', 'y'):
                    if da.sizes[dim] > 1:
                        item_dim = dim  # Find multi-item dimension
                    elif da.sizes[dim] == 1:
                        da = da.squeeze(dim, drop=True)
            
            # Extract SCL data
            scl_da = da.sel(band=scl_asset_key)
            
            # Get numpy array
            scl_arr = scl_da.values
            
            # Check if array size is reasonable
            if not check_memory_requirements(scl_arr.shape, scl_arr.dtype):
                logging.warning(f"[{partition_id}]   SCL array too large, cannot perform quality assessment")
                return False, 0.0, None
            
            # Perform SCL processing - vectorized version
            valid_mask, tile_selection, valid_pct = process_scl(scl_arr, mask_np, partition_id)
            
            # Check if valid coverage meets threshold
            if valid_pct < min_coverage:
                logging.warning(f"[{partition_id}]   ⚠️ {date_key} valid coverage {valid_pct:.2f}% < {min_coverage}%, skipping SCL generation")
                return False, valid_pct, None  # Return insufficient coverage result but don't generate file
            
            # Create SCL mosaic output based on valid_mask and tile_selection, preserving original SCL values - vectorized version
            scl_output = create_scl_mosaic(scl_arr, tile_selection, mask_np, (tpl["height"], tpl["width"]), date_key, partition_id)
            
            # Create metadata
            metadata = {
                "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                "DATE_ACQUIRED": date_key,
                "BAND_NAME": scl_asset_key,
                "ITEMS_COUNT": len(items),
                "VALID_COVERAGE_PCT": f"{valid_pct:.2f}"
            }
            
            # Write SCL GeoTIFF
            write_tiff(scl_output, scl_out_path, tpl, "uint8", metadata)
            
            # Validate output file
            if not validate_tiff(scl_out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                logging.error(f"[{partition_id}]   ✗ SCL file validation failed")
                if scl_out_path.exists():
                    scl_out_path.unlink()
                return False, 0.0, None
            
            # Finally analyze generated SCL file again
            try:
                with rasterio.open(scl_out_path) as src:
                    scl_data = src.read(1)
                    # Check SCL value distribution - vectorized operation
                    unique_values, unique_counts = np.unique(scl_data, return_counts=True)
                    value_counts = dict(zip(unique_values, unique_counts))
                    
                    # Log SCL value distribution
                    total_pixels = np.sum(mask_np)
                    logging.info(f"[{partition_id}]   Generated SCL file value distribution statistics ({date_key}):")
                    for val in sorted(value_counts.keys()):
                        count = value_counts[val]
                        desc = SCL_DESCRIPTIONS.get(val, "Unknown")
                        percent = 100 * count / total_pixels if total_pixels > 0 else 0
                        logging.info(f"[{partition_id}]     SCL value {val} ({desc}): {count} pixels ({percent:.2f}%)")
                    
                    scl_stats = src.statistics(1)
                    logging.info(f"[{partition_id}]   SCL file statistics: min={scl_stats.min}, max={scl_stats.max}, mean={scl_stats.mean:.2f}")
            except Exception as e:
                logging.warning(f"[{partition_id}]   Cannot read SCL file statistics: {e}")
            
            logging.info(f"[{partition_id}]   ✓ SCL quality assessment and file generation complete, validity: {valid_pct:.2f}%, "
                        f"file size: {os.path.getsize(scl_out_path)/1e6:.2f} MB, time {time.time()-t0:.1f}s")
            
            return True, valid_pct, tile_selection
            
    except TimeoutException as e:
        logging.error(f"[{partition_id}]   ✗ SCL processing timeout: {e}")
        return False, 0.0, None
            
    except Exception as e:
        logging.error(f"[{partition_id}]   ✗ SCL processing failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logging.debug(traceback.format_exc())
        return False, 0.0, None

# ─── Single Day Task ──────────────────────────────────────────────────────────────────
def process_day(date_key:str, items, tpl, bbox_proj, mask_np,
               out_root:Path, res:int, chunksize:int,
               overwrite:bool, min_coverage:float=5.0,
               partition_id:str="unknown") -> bool:
    """Process single day data, optimized SCL processing logic, preserving original SCL values"""
    logging.info(f"[{partition_id}] → {date_key} (item={len(items)})")
    t0 = time.time()

    try:
        # Use 60-minute timeout control for single day processing
        with timeout_handler(DAY_TIMEOUT):
            # Create band output directories
            for outname in BAND_MAPPING.values():
                band_dir = out_root / outname
                band_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if all processing is already complete
            if not overwrite:
                all_exist = True
                for band_name in S2_BANDS:
                    out_name = BAND_MAPPING[band_name]
                    out_path = out_root / out_name / f"{date_key}_mosaic.tiff"
                    if not out_path.exists() or not validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                        all_exist = False
                        break
                
                if all_exist:
                    logging.info(f"[{partition_id}]   All bands already exist with valid files, skipping")
                    return True
            
            # Process SCL band, perform quality assessment and generate SCL file, while keeping tile_selection for other bands
            scl_success, valid_pct, tile_selection = process_scl_assessment_and_generation(
                items, date_key, tpl, bbox_proj, mask_np, res, chunksize,
                min_coverage, out_root, overwrite, partition_id
            )
            
            # Check if SCL processing was successful
            if not scl_success:
                if valid_pct < min_coverage:
                    logging.warning(f"[{partition_id}]   {date_key} valid coverage {valid_pct:.2f}% < {min_coverage}%, skipping other band processing")
                    return True  # Insufficient coverage is not failure, just skip
                else:
                    logging.error(f"[{partition_id}]   {date_key} SCL processing failed, skipping this date processing")
                    return False
            
            # Create temporary directory for processing
            day_temp_dir = tempfile.mkdtemp(prefix=f"s2_{date_key}_", dir=TEMP_DIR)
            logging.debug(f"[{partition_id}]   Temporary directory: {day_temp_dir}")
            
            try:
                # Use thread pool to process bands in parallel (excluding SCL)
                scl_key = "scl" if BAND_MAPPING is AWS_BAND_MAPPING else "SCL"
                other_bands = [band for band in S2_BANDS if band != scl_key]
                max_workers = min(DEFAULT_MAX_WORKERS, os.cpu_count())
                logging.info(f"[{partition_id}]   Using {max_workers} threads to process {len(other_bands)} bands in parallel (excluding SCL)")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    futures = {}
                    
                    for band_name in other_bands:
                        out_name = BAND_MAPPING[band_name]
                        out_path = out_root / out_name / f"{date_key}_mosaic.tiff"
                        
                        # If file already exists and is valid, skip
                        if not overwrite and out_path.exists() and validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                            logging.info(f"[{partition_id}]     Band {band_name} valid file already exists, skipping")
                            continue
                        
                        # Create temporary output path
                        temp_path = Path(day_temp_dir) / f"{band_name}_{date_key}.tiff"
                        
                        # Submit task
                        future = executor.submit(
                            process_band, 
                            items, band_name, date_key, tpl, bbox_proj, mask_np, tile_selection,
                            res, chunksize, temp_path, partition_id
                        )
                        futures[future] = (band_name, out_path, temp_path)
                    
                    # Process results
                    success_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        band_name, out_path, temp_path = futures[future]
                        try:
                            success = future.result()
                            if success:
                                # Check if temporary file is valid
                                if temp_path.exists() and validate_tiff(temp_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                                    # Move temporary file to final location
                                    shutil.copy2(temp_path, out_path)
                                    success_count += 1
                                    logging.info(f"[{partition_id}]     ✓ {band_name} processing complete")
                                else:
                                    logging.error(f"[{partition_id}]     ✗ {band_name} temporary file invalid or does not exist")
                            else:
                                logging.warning(f"[{partition_id}]     ✗ {band_name} processing failed")
                        except Exception as e:
                            logging.error(f"[{partition_id}]     ✗ {band_name} processing exception: {e}")
            finally:
                # Clean up daily temporary directory
                try:
                    shutil.rmtree(day_temp_dir)
                    logging.debug(f"[{partition_id}]   Cleaned temporary directory: {day_temp_dir}")
                except Exception as e:
                    logging.warning(f"[{partition_id}]   Failed to clean temporary directory: {e}")
            
            # Log processing results
            total_other_bands = len(other_bands)
            total_bands = len(S2_BANDS)  # Including SCL
            proc_time = time.time() - t0
            
            # Calculate total success: SCL success (1) + other bands success count
            total_success = (1 if scl_success else 0) + success_count
            
            if total_success == total_bands:
                logging.info(f"[{partition_id}] ← {date_key} all bands processed successfully ({total_success}/{total_bands}), time {proc_time:.1f}s")
                return True
            elif total_success > 0:
                logging.warning(f"[{partition_id}] ← {date_key} partial bands processed successfully ({total_success}/{total_bands}), time {proc_time:.1f}s")
                return True  # Partial success is also success
            else:
                logging.error(f"[{partition_id}] ← {date_key} all bands processing failed, time {proc_time:.1f}s")
                return False
                
    except TimeoutException as e:
        proc_time = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {date_key} processing timeout ({proc_time:.1f}s): {e}")
        return False
    except Exception as e:
        proc_time = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {date_key} processing failed: {type(e).__name__} - {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logging.debug(traceback.format_exc())
        return False

# ─── Main Program ───────────────────────────────────────────────────────────────────
def main():
    a = get_args()
    out_dir = Path(a.output).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # Use command line specified temporary directory
    global TEMP_DIR
    TEMP_DIR = a.temp_dir

    # Configure backend-dependent band mapping + STAC backend selection.
    # DATA_SOURCE also gates the baseline-offset correction in `harmonize_arr`:
    # MPC stores the raw +1000 offset, AWS earth-search has already removed it.
    global BAND_MAPPING, S2_BANDS, DATA_SOURCE
    DATA_SOURCE = a.data_source
    if a.data_source == "aws":
        BAND_MAPPING = AWS_BAND_MAPPING
        S2_BANDS = list(BAND_MAPPING.keys())
        search_items._data_source = "aws"  # type: ignore[attr-defined]
    else:
        BAND_MAPPING = MPC_BAND_MAPPING
        S2_BANDS = list(BAND_MAPPING.keys())
        search_items._data_source = "mpc"  # type: ignore[attr-defined]

    setup_logging(a.debug, out_dir, a.partition_id)
    logging.info(f"[{a.partition_id}] ⚡ S2 Fast Processor startup (Optimized Parallel Edition - SCL Original Value Preservation Version)"); 
    logging.info(f"[{a.partition_id}] Data source: {a.data_source}")
    log_sys(a.partition_id)
    logging.info(f"[{a.partition_id}] Processing timeout settings: Overall {PROCESS_TIMEOUT//60} minutes, Single day {DAY_TIMEOUT//60} minutes, Single band {BAND_TIMEOUT//60} minutes, SCL assessment {SCL_BAND_TIMEOUT//60} minutes")
    logging.info(f"[{a.partition_id}] SCL assessment attempt count: {SCL_MAX_ATTEMPTS}")
    logging.info(f"[{a.partition_id}] Temporary directory: {TEMP_DIR}")
    logging.info(f"[{a.partition_id}] Processing time period: {a.start_date} → {a.end_date}")

    tpl, bbox_proj, bbox_ll, mask_np = load_roi(Path(a.input_tiff), a.partition_id, a.resolution)

    # Search STAC items
    search_date_range = f"{a.start_date}/{a.end_date}"

    items = search_items(bbox_ll, search_date_range, a.max_cloud, a.partition_id)
    if not items:
        logging.warning(f"[{a.partition_id}] No images meeting criteria, exiting")
        return

    # Group by date
    groups = group_by_date(items, a.partition_id)

    # Create temporary directory for processing
    base_temp_dir = tempfile.mkdtemp(prefix=f"s2_proc_{a.partition_id}_", dir=TEMP_DIR)
    logging.info(f"[{a.partition_id}] Main temporary directory: {base_temp_dir}")

    try:
        # Create initial client
        dask_client = make_client(a.dask_workers, a.worker_memory, a.partition_id)
        
        report_path = out_dir / f"dask-report-{a.partition_id}.html"
        with performance_report(filename=report_path):
            # Process each day's data
            results = []
            for i, (d, its) in enumerate(groups.items()):
                # Perform garbage collection
                gc.collect()
                
                # Try to restart or recreate client before processing each new date
                if i > 0:  # Skip first date as client was just created
                    try:
                        logging.info(f"[{a.partition_id}] Attempting to restart Dask client, setting timeout to 240 seconds, not waiting for workers...")
                        # Set longer timeout and no wait for workers option
                        dask_client.restart(timeout=240, wait_for_workers=False)
                    except (TimeoutError, Exception) as e:
                        logging.warning(f"[{a.partition_id}] Dask client restart failed: {e}, trying to recreate client...")
                        try:
                            # Close existing client
                            try:
                                dask_client.close(timeout=30)
                            except:
                                pass
                            
                            # Force memory cleanup
                            gc.collect()
                            
                            # Brief pause to let system release resources
                            time.sleep(5)
                            
                            # Recreate client
                            dask_client = make_client(a.dask_workers, a.worker_memory, a.partition_id)
                            logging.info(f"[{a.partition_id}] Dask client recreated successfully")
                        except Exception as recreate_error:
                            logging.error(f"[{a.partition_id}] Cannot recreate Dask client: {recreate_error}")
                            # Continue processing, let function run normally even if performance may be reduced
                
                # Process current day data
                try:
                    success = process_day(
                        d, its, tpl, bbox_proj, mask_np,
                        out_dir, a.resolution, a.chunksize,
                        a.overwrite, a.min_coverage, a.partition_id
                    )
                    results.append(success)
                except Exception as day_error:
                    logging.error(f"[{a.partition_id}] Exception occurred while processing date {d}: {day_error}")
                    # Add a failure result
                    results.append(False)
                    
                    # If exception occurs, try to recreate client to restore state
                    try:
                        # Close existing client
                        try:
                            dask_client.close(timeout=30)
                        except:
                            pass
                        
                        # Recreate client
                        time.sleep(5)  # Wait for resource release
                        dask_client = make_client(a.dask_workers, a.worker_memory, a.partition_id)
                        logging.info(f"[{a.partition_id}] Dask client recreated successfully after exception")
                    except:
                        # If unable to recreate client, continue trying to process next day
                        pass
        
        # Try to close client
        try:
            dask_client.close(timeout=30)
        except:
            pass
        
        # Summary statistics
        success_count = sum(results)
        total_count = len(results)
        
        logging.info(f"[{a.partition_id}] ✅ Partition processing complete: successful {success_count}/{total_count} days")
        logging.info(f"[{a.partition_id}] 📊 Dask performance report saved: {report_path}")
        
        # Return appropriate exit code
        if success_count == 0 and total_count > 0:
            sys.exit(1)  # All failed
        elif success_count < total_count:
            logging.warning(f"[{a.partition_id}] ⚠️  Some dates processing failed ({total_count - success_count}/{total_count})")
            sys.exit(2)  # Partial failure
        else:
            sys.exit(0)  # All successful

    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(base_temp_dir)
            logging.info(f"[{a.partition_id}] Cleaned main temporary directory")
        except Exception as e:
            logging.warning(f"[{a.partition_id}] Failed to clean main temporary directory: {e}")

if __name__ == "__main__":
    main()
