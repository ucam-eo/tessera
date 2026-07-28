import os
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time

def check_intersection(folder_info):
    """Check if the patch intersects with the downstream TIFF"""
    folder, d_pixel_retiled_path, downstream_bounds, downstream_crs = folder_info
    
    roi_path = os.path.join(d_pixel_retiled_path, folder, 'roi.tiff')
    if not os.path.exists(roi_path):
        return None
    
    with rasterio.open(roi_path) as src:
        bounds = src.bounds
        transform = src.transform
        roi_width = src.width
        roi_height = src.height
        roi_crs = src.crs
    
    # Check if bounding boxes intersect
    if (bounds.right < downstream_bounds.left or 
        bounds.left > downstream_bounds.right or 
        bounds.top < downstream_bounds.bottom or 
        bounds.bottom > downstream_bounds.top):
        return None
    
    # Warn if CRS are different, might cause alignment issues
    if roi_crs != downstream_crs:
        print(f"Warning: CRS of {folder} doesn't match downstream TIFF, may cause alignment issues")
    
    return {
        'folder': folder,
        'bounds': bounds,
        'transform': transform,
        'roi_width': roi_width,
        'roi_height': roi_height,
    }

def stitch_and_crop_representations(d_pixel_retiled_path, representation_retiled_path, downstream_tiff_path, out_dir):
    """
    Optimized version: Stitch representation vectors and crop to the downstream TIFF extent.
    
    Parameters:
    d_pixel_retiled_path: Directory path containing roi.tiff files
    representation_retiled_path: Directory path containing representation vector npy files
    downstream_tiff_path: Path to the downstream TIFF file
    out_dir: Output directory
    """
    start_time = time.time()
    
    # Read downstream TIFF to determine target extent and dimensions
    print(f"Reading downstream TIFF: {downstream_tiff_path}")
    with rasterio.open(downstream_tiff_path) as src:
        downstream_bounds = src.bounds
        downstream_width = src.width
        downstream_height = src.height
        downstream_transform = src.transform
        downstream_crs = src.crs
        
        # Read the downstream TIFF data to count non-zero pixels
        downstream_data = src.read(1)  # Read first band
        nonzero_downstream_pixels = np.count_nonzero(downstream_data)
        print(f"Downstream TIFF has {nonzero_downstream_pixels} non-zero pixels out of {downstream_height * downstream_width} total pixels")
    
    print(f"Downstream TIFF dimensions: {downstream_height} x {downstream_width}")
    print(f"Downstream TIFF bounds: {downstream_bounds}")
    
    # Get list of all patch folders
    patch_folders = [f for f in os.listdir(d_pixel_retiled_path) 
                    if os.path.isdir(os.path.join(d_pixel_retiled_path, f))]
    
    # Analyze a sample representation vector to get the number of bands
    first_folder = patch_folders[0]
    first_rep_path = os.path.join(representation_retiled_path, f"{first_folder}.npy")
    rep_sample = np.load(first_rep_path)
    num_bands = rep_sample.shape[2]
    patch_height, patch_width, _ = rep_sample.shape
    
    print(f"Representation vector dimensions: {patch_height} x {patch_width} x {num_bands}")
    
    # Initialize target array (direct size of downstream TIFF)
    target_array = np.zeros((downstream_height, downstream_width, num_bands), dtype=np.float32)
    count_array = np.zeros((downstream_height, downstream_width), dtype=np.int32)
    
    # Find patches that intersect with the downstream TIFF using parallel processing
    print("Filtering patches that intersect with downstream TIFF...")
    
    # Prepare parameters
    folder_infos = [(folder, d_pixel_retiled_path, downstream_bounds, downstream_crs) for folder in patch_folders]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        intersection_results = list(tqdm(executor.map(check_intersection, folder_infos), total=len(patch_folders)))
    
    # Filter out None results
    intersecting_patches = [result for result in intersection_results if result is not None]
    print(f"Found {len(intersecting_patches)} patches intersecting with downstream TIFF (out of {len(patch_folders)} total)")
    
    if not intersecting_patches:
        raise ValueError("No patches found that intersect with the downstream TIFF!")
    
    # Calculate the scale relationship between patch ROI and representation vector
    first_patch = intersecting_patches[0]
    first_roi_path = os.path.join(d_pixel_retiled_path, first_patch['folder'], 'roi.tiff')
    
    with rasterio.open(first_roi_path) as src:
        first_roi_height = src.height
        first_roi_width = src.width
    
    # Calculate scale factors: ratio of representation vector size to ROI size
    height_scale = patch_height / first_roi_height
    width_scale = patch_width / first_roi_width
    
    print(f"ROI to representation vector scale: height={height_scale}, width={width_scale}")
    
    # Process each intersecting patch sequentially and place in target array
    print("Processing intersecting patches...")
    processed_count = 0
    
    for patch_index, patch_info in tqdm(enumerate(intersecting_patches), total=len(intersecting_patches)):
        folder = patch_info['folder']
        bounds = patch_info['bounds']
        roi_transform = patch_info['transform']
        
        # Load corresponding representation vector
        rep_path = os.path.join(representation_retiled_path, f"{folder}.npy")
        if not os.path.exists(rep_path):
            continue
        
        rep_data = np.load(rep_path)
        
        scale_path = rep_path.replace('.npy', '_scales.npy')
        if os.path.exists(scale_path):
            scales = np.load(scale_path)  # (H, W) per-pixel scale factors
            rep_data = rep_data.astype(np.float32) * scales[:, :, np.newaxis]
        
        # Calculate ROI window in downstream TIFF
        # First determine pixel coordinates of ROI bounds in downstream TIFF
        window = rasterio.windows.from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, 
            downstream_transform
        )
        
        # Convert to integer coordinates (position on downstream TIFF)
        dst_row_start = max(0, int(round(window.row_off)))
        dst_row_end = min(downstream_height, int(round(window.row_off + window.height)))
        dst_col_start = max(0, int(round(window.col_off)))
        dst_col_end = min(downstream_width, int(round(window.col_off + window.width)))
        
        # Skip if window is outside downstream TIFF range
        if dst_row_start >= downstream_height or dst_col_start >= downstream_width or dst_row_end <= 0 or dst_col_end <= 0:
            continue
        
        # Calculate corresponding pixel area in representation vector
        # First determine which part of original ROI corresponds to window on downstream TIFF
        roi_window = rasterio.windows.from_bounds(
            downstream_transform.c + dst_col_start * downstream_transform.a,
            downstream_transform.f + dst_row_end * downstream_transform.e,
            downstream_transform.c + dst_col_end * downstream_transform.a,
            downstream_transform.f + dst_row_start * downstream_transform.e,
            roi_transform
        )
        
        # Convert ROI window to window on representation vector (considering scale factors)
        src_row_start = max(0, int(round(roi_window.row_off * height_scale)))
        src_row_end = min(rep_data.shape[0], int(round((roi_window.row_off + roi_window.height) * height_scale)))
        src_col_start = max(0, int(round(roi_window.col_off * width_scale)))
        src_col_end = min(rep_data.shape[1], int(round((roi_window.col_off + roi_window.width) * width_scale)))
        
        # Ensure source and target areas match in size (adjust target area to fit source area)
        src_height = src_row_end - src_row_start
        src_width = src_col_end - src_col_start
        
        # Adjust target area size to match source area (maintaining aspect ratio)
        dst_height = dst_row_end - dst_row_start
        dst_width = dst_col_end - dst_col_start
        
        # Skip if source or target area is empty
        if src_height <= 0 or src_width <= 0 or dst_height <= 0 or dst_width <= 0:
            continue
        
        # If source and target areas have different sizes, adjust
        if src_height != dst_height or src_width != dst_width:
            # Create temporary array for resampled data
            resampled_data = np.zeros((dst_height, dst_width, num_bands), dtype=rep_data.dtype)
            
            # Calculate sampling ratios
            row_ratio = src_height / dst_height
            col_ratio = src_width / dst_width
            
            # Use simple bilinear interpolation for resampling
            for y in range(dst_height):
                for x in range(dst_width):
                    # Calculate corresponding source position (floating point)
                    src_y = src_row_start + y * row_ratio
                    src_x = src_col_start + x * col_ratio
                    
                    # Get four nearest source pixels
                    y0 = int(src_y)
                    y1 = min(y0 + 1, src_row_end - 1)
                    x0 = int(src_x)
                    x1 = min(x0 + 1, src_col_end - 1)
                    
                    # Calculate weights
                    wy1 = src_y - y0
                    wy0 = 1 - wy1
                    wx1 = src_x - x0
                    wx0 = 1 - wx1
                    
                    # Bilinear interpolation
                    if y0 >= src_row_start and x0 >= src_col_start:
                        for b in range(num_bands):
                            resampled_data[y, x, b] = (
                                wy0 * wx0 * rep_data[y0, x0, b] +
                                wy0 * wx1 * rep_data[y0, x1, b] +
                                wy1 * wx0 * rep_data[y1, x0, b] +
                                wy1 * wx1 * rep_data[y1, x1, b]
                            )
            
            # Add resampled data to target array (direct assignment as no overlap)
            target_array[dst_row_start:dst_row_end, dst_col_start:dst_col_end, :] = resampled_data
        else:
            # If sizes match, copy directly
            # Ensure region shapes match
            max_height = min(dst_height, src_height)
            max_width = min(dst_width, src_width)
            
            # Add source region data to target array (direct assignment as no overlap)
            target_array[dst_row_start:dst_row_start+max_height, 
                         dst_col_start:dst_col_start+max_width, :] = rep_data[
                         src_row_start:src_row_start+max_height, 
                         src_col_start:src_col_start+max_width, :]
        
        # Update count array (for tracking coverage statistics only)
        count_array[dst_row_start:dst_row_end, dst_col_start:dst_col_end] += 1
        processed_count += 1
    
    print(f"Successfully processed {processed_count} patches")
    
    # Note: Skipping overlap handling as requested since the TIFFs don't overlap
    
    # Create output directory (if it doesn't exist)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as numpy file
    out_path = os.path.join(out_dir, "stitched_representation.npy")
    print(f"Saving to: {out_path}")
    np.save(out_path, target_array)
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    # Output statistics - modified to calculate coverage based on non-zero pixels in downstream TIFF
    non_zero_pixels = np.count_nonzero(count_array)
    total_pixels = downstream_height * downstream_width
    nonzero_coverage_percent = (non_zero_pixels / nonzero_downstream_pixels) * 100
    
    print(f"Stitching coverage: {nonzero_coverage_percent:.2f}% of non-zero downstream pixels ({non_zero_pixels}/{nonzero_downstream_pixels} pixels)")
    print(f"Stitching coverage relative to total area: {(non_zero_pixels / total_pixels) * 100:.2f}% ({non_zero_pixels}/{total_pixels} pixels)")
    
    return out_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Stitch representation vectors and crop to downstream TIFF extent')
    parser.add_argument('--d_pixel_retiled_path', required=True, help='Directory path containing roi.tiff files')
    parser.add_argument('--representation_retiled_path', required=True, help='Directory path containing representation vector npy files')
    parser.add_argument('--downstream_tiff', required=True, help='Path to the downstream TIFF file')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    stitch_and_crop_representations(
        args.d_pixel_retiled_path,
        args.representation_retiled_path,
        args.downstream_tiff,
        args.out_dir
    )

if __name__ == "__main__":
    main()
    
"""
Usage:
python stitch_tiled_representation.py \
--d_pixel_retiled_path /maps/usr/btfm_project/my_data/d_pixel_retiled \
--representation_retiled_path /maps/usr/btfm_project/my_data/representation_retiled \
--downstream_tiff /maps/usr/btfm_project/my_data/downstream.tiff \
--out_dir /maps/usr/btfm_project/my_data
"""