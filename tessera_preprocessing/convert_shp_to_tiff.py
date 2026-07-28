#!/usr/bin/env python3
import logging
import os
import sys

import fiona
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def determine_utm_zone(lon, lat):
    """
    Determine the best UTM zone based on longitude and latitude.

    Args:
        lon (float): Longitude (degrees)
        lat (float): Latitude (degrees)

    Returns:
        tuple: (epsg code, zone number, is northern hemisphere)
    """
    # UTM zone width is 6 degrees
    zone_number = int((lon + 180) / 6) + 1

    # Special case for Norway
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone_number = 32

    # Special case for Svalbard
    if 72 <= lat < 84:
        if 0 <= lon < 9:
            zone_number = 31
        elif 9 <= lon < 21:
            zone_number = 33
        elif 21 <= lon < 33:
            zone_number = 35
        elif 33 <= lon < 42:
            zone_number = 37

    is_northern = lat >= 0

    # EPSG code: Northern Hemisphere EPSG:326xx, Southern Hemisphere EPSG:327xx
    epsg_code = 32600 + zone_number if is_northern else 32700 + zone_number

    return epsg_code, zone_number, is_northern


def determine_best_utm_crs(geometries, src_crs):
    """
    Determine the best UTM coordinate reference system based on the centroid of the geometry collection.

    Args:
        geometries (list): List of geometries in the source CRS.
        src_crs (CRS): Source coordinate reference system.

    Returns:
        CRS: Target UTM coordinate reference system.
    """
    logger.info("Starting to determine the best UTM zone...")

    # If not WGS84 (EPSG:4326), convert first
    if src_crs.to_epsg() != 4326 and src_crs != CRS.from_epsg(4326):
        logger.info(f"Source CRS is not WGS84, converting to WGS84...")
        wgs84_crs = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(src_crs, wgs84_crs, always_xy=True).transform
        # Convert all geometries to WGS84
        wgs84_geoms = [shp_transform(transformer, shape(geom)) for geom in geometries]
        logger.info(f"WGS84 conversion completed.")
    else:
        logger.info(f"Source CRS is already WGS84, no conversion needed.")
        wgs84_geoms = [shape(geom) for geom in geometries]

    # Create the union of all geometries and get its centroid
    union_geom = unary_union(wgs84_geoms)
    centroid = union_geom.centroid
    cent_lon, cent_lat = centroid.x, centroid.y

    # Determine UTM zone based on the centroid
    epsg_code, zone_number, is_northern = determine_utm_zone(cent_lon, cent_lat)

    hemisphere = "Northern Hemisphere" if is_northern else "Southern Hemisphere"
    logger.info(f"Data centroid location: {cent_lon:.6f}°E, {cent_lat:.6f}°N")
    logger.info(f"Selected UTM zone: {zone_number}{hemisphere[0]} (EPSG:{epsg_code})")

    return CRS.from_epsg(epsg_code)


def shp_to_tiff(shp_path, tiff_path=None, pixel_size=100, force_crs=None):
    """
    Convert a shapefile to a TIFF raster.

    Args:
        shp_path (str): Input shapefile path.
        tiff_path (str, optional): Output TIFF path. If None, derived from shp_path.
        pixel_size (float, optional): Output pixel size (meters). Defaults to 100.
        force_crs (CRS, optional): Force a specific target CRS. If None, automatically determines the best UTM zone.

    Returns:
        tuple: (Path to the main TIFF file, Path to the convex hull TIFF file)
    """
    # Set default output path if not provided
    if tiff_path is None:
        tiff_path = os.path.splitext(shp_path)[0] + ".tiff"

    logger.info(f"Starting conversion of shapefile: {shp_path}")
    logger.info(f"Output TIFF will be saved as: {tiff_path}")
    logger.info(f"Using pixel size: {pixel_size} meters")

    # Open the shapefile and read geometries
    with fiona.open(shp_path, "r") as src:
        # Get basic shapefile information
        num_features = len(src)
        src_driver = src.driver
        src_schema = src.schema

        logger.info(f"Shapefile information:")
        logger.info(f"  - Number of features: {num_features}")
        logger.info(f"  - Driver: {src_driver}")
        logger.info(f"  - Schema: {src_schema}")

        # Read all geometries
        geometries = [feature["geometry"] for feature in src]
        logger.info(f"Read {len(geometries)} geometries from the shapefile")

        # Get source CRS, default to EPSG:4326 if undefined
        if src.crs:
            src_crs = CRS(src.crs)
            logger.info(f"Source CRS: {src_crs.to_string()}")
        else:
            src_crs = CRS.from_epsg(4326)
            logger.warning(f"Source CRS is undefined, defaulting to EPSG:4326 (WGS84)")

    # Determine target CRS (if not forced)
    if force_crs:
        target_crs = force_crs
        logger.info(f"Using forced target CRS: {target_crs.to_string()}")
    else:
        target_crs = determine_best_utm_crs(geometries, src_crs)
        logger.info(f"Automatically selected target CRS: {target_crs.to_string()}")

    # Construct transformer from source CRS to target CRS
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True).transform
    logger.info("Reprojecting geometries to the target CRS...")

    # Reproject geometries to the target CRS
    try:
        reprojected_geoms = [
            mapping(shp_transform(transformer, shape(geom))) for geom in geometries
        ]
        logger.info(f"Successfully reprojected {len(reprojected_geoms)} geometries")
    except Exception as e:
        logger.error(f"Error reprojecting geometries: {str(e)}")
        raise

    # Calculate the union and convex hull of the reprojected geometries
    try:
        union_geom = unary_union([shape(geom) for geom in reprojected_geoms])
        convex_hull = union_geom.convex_hull
        logger.info("Created union and convex hull of geometries")
    except Exception as e:
        logger.error(f"Error creating union or convex hull: {str(e)}")
        raise

    # Get the bounds of the convex hull
    minx, miny, maxx, maxy = convex_hull.bounds
    logger.info(f"Convex hull bounds: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")

    # Calculate the dimensions of the output raster
    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))

    if width <= 0 or height <= 0:
        error_msg = f"Invalid raster dimensions: width={width}, height={height}. Check pixel size."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Output raster dimensions: Width={width}, Height={height} pixels")

    # Create the affine transformation matrix
    transform_affine = from_origin(minx, maxy, pixel_size, pixel_size)
    logger.info(f"Affine transform matrix: {transform_affine}")

    # Rasterize the geometries
    logger.info("Rasterizing geometries...")
    try:
        raster = rasterize(
            reprojected_geoms,
            out_shape=(height, width),
            transform=transform_affine,
            fill=0,
            default_value=255,
            dtype="uint8",
        )
        logger.info(f"Rasterization complete. Raster shape: {raster.shape}")
    except Exception as e:
        logger.error(f"Error during rasterization: {str(e)}")
        raise

    # Write the TIFF file
    logger.info(f"Writing raster to TIFF file: {tiff_path}")
    try:
        with rasterio.open(
            tiff_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            crs=target_crs,
            transform=transform_affine,
        ) as dst:
            dst.write(raster, 1)
        logger.info("Successfully wrote TIFF file")
    except Exception as e:
        logger.error(f"Error writing TIFF file: {str(e)}")
        raise

    # Create convex hull TIFF
    hull_tiff_path = os.path.splitext(tiff_path)[0] + "_convex_hull.tiff"
    logger.info(f"Creating convex hull TIFF: {hull_tiff_path}")

    try:
        # Convert convex hull to GeoJSON compatible format
        hull_geometry = mapping(convex_hull)

        # Rasterize the convex hull
        hull_raster = rasterize(
            [(hull_geometry, 255)],
            out_shape=(height, width),
            transform=transform_affine,
            fill=0,
            dtype="uint8",
        )

        # Write the convex hull to TIFF
        with rasterio.open(
            hull_tiff_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=hull_raster.dtype,
            crs=target_crs,
            transform=transform_affine,
        ) as dst:
            dst.write(hull_raster, 1)
        logger.info("Successfully wrote convex hull TIFF file")
    except Exception as e:
        logger.error(f"Error creating convex hull TIFF: {str(e)}")
        raise

    return tiff_path, hull_tiff_path


def main():
    """
    Main function to run the conversion process.
    """
    # Input shapefile path
    if len(sys.argv) < 2:
        # uv run ./convert_shp_to_tiff.py /path/to/shapefile.shp
        print("Usage: python ./convert_shp_to_tiff.py /path/to/shapefile.shp")
        sys.exit(1)
    shp_path = sys.argv[1]

    # Call the conversion function
    try:
        tiff_path, hull_tiff_path = shp_to_tiff(shp_path, pixel_size=10, force_crs=None)
        logger.info(f"Conversion complete!")
        logger.info(f"TIFF file saved at: {tiff_path}")
        logger.info(f"Convex hull TIFF saved at: {hull_tiff_path}")
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
