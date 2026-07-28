#!/usr/bin/env python3
"""
Identify tiles that need regeneration for an AOI and create processing ROIs.

Performs three steps:
  1. Query the GeoTessera registry to find which 0.1-degree tiles are available
     and which are missing entirely for the given AOI.
  2. Optionally download available GeoTessera tiles, then scan them for blank
     (near-zero) pixels that indicate masked ocean/coastal areas.
  3. Group all tiles needing regeneration into spatially adjacent clusters and
     write per-group ROI TIFFs plus a manifest for the inference pipeline.

Outputs:
  <output-dir>/manifest.json         Processing manifest
  <output-dir>/keep_tiles.json       Tiles that can be used as-is from GeoTessera
  <output-dir>/group_NNN/roi.tiff    Per-group ROI rasters
"""

import argparse
import json
import os

import geopandas as gpd
import numpy as np
import rasterio
from geotessera import GeoTessera
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
from shapely.ops import transform as shp_transform, unary_union

PIXEL_SIZE = 10


def determine_utm_epsg(lon, lat):
    zone_number = int((lon + 180) / 6) + 1
    return 32600 + zone_number if lat >= 0 else 32700 + zone_number


def query_registry(aoi_path, year):
    """Return sets of (lon, lat) for tiles the registry has vs tiles it lacks."""
    aoi = gpd.read_file(aoi_path)
    bounds = aoi.total_bounds

    gt = GeoTessera()
    available = list(gt.registry.load_blocks_for_region(
        bounds=tuple(bounds), year=year,
    ))
    available_coords = {(round(lon, 2), round(lat, 2)) for _, lon, lat in available}

    step = 0.1
    min_lon = np.floor(bounds[0] / step) * step + step / 2
    max_lon = np.ceil(bounds[2] / step) * step + step / 2
    min_lat = np.floor(bounds[1] / step) * step + step / 2
    max_lat = np.ceil(bounds[3] / step) * step + step / 2

    all_tiles = []
    for lon in np.arange(min_lon, max_lon, step):
        for lat in np.arange(min_lat, max_lat, step):
            lr, latr = round(lon, 2), round(lat, 2)
            tile_box = box(lr - 0.05, latr - 0.05, lr + 0.05, latr + 0.05)
            if aoi.geometry.intersects(tile_box).any():
                all_tiles.append((lr, latr))

    missing = {c for c in all_tiles if c not in available_coords}
    covered = {c for c in all_tiles if c in available_coords}
    return covered, missing


def download_geotessera(aoi_path, output_dir, year):
    """Download precomputed GeoTessera embeddings for the AOI."""
    aoi = gpd.read_file(aoi_path)
    bounds = tuple(aoi.total_bounds)

    gt = GeoTessera()
    tiles = list(gt.registry.load_blocks_for_region(bounds=bounds, year=year))
    print(f"Found {len(tiles)} available tiles to download")

    downloaded, failed = [], []
    for i, (yr, lon, lat) in enumerate(tiles):
        tile_name = f"grid_{lon:.2f}_{lat:.2f}"
        out_dir = os.path.join(output_dir, tile_name)
        npy_path = os.path.join(out_dir, f"{tile_name}.npy")

        if os.path.exists(npy_path):
            downloaded.append(tile_name)
            continue

        try:
            embedding, crs, transform = gt.fetch_embedding(lon=lon, lat=lat, year=yr)
            os.makedirs(out_dir, exist_ok=True)
            np.save(npy_path, embedding)

            meta = {
                "tile_name": tile_name,
                "center_lon": float(lon),
                "center_lat": float(lat),
                "year": yr,
                "shape": list(embedding.shape),
                "dtype": str(embedding.dtype),
                "crs": str(crs),
                "transform": list(transform)[:6],
            }
            with open(os.path.join(out_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

            downloaded.append(tile_name)
            print(f"  [{i+1}/{len(tiles)}] {tile_name} -- shape={embedding.shape}")
        except Exception as e:
            failed.append((tile_name, str(e)))
            print(f"  [{i+1}/{len(tiles)}] {tile_name} -- FAILED: {e}")

    manifest = {"year": year, "downloaded": downloaded,
                "failed": [{"name": n, "error": e} for n, e in failed]}
    with open(os.path.join(output_dir, "download_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Downloaded: {len(downloaded)}, Failed: {len(failed)}")
    return downloaded, failed


def scan_tiles_for_blanks(geotessera_dir, missing_coords, zero_threshold):
    """Scan downloaded tiles for blank ocean pixels and return regen/keep lists."""
    regen_set = set(missing_coords)
    keep_tiles = []

    for d in sorted(os.listdir(geotessera_dir)):
        if not d.startswith("grid_"):
            continue
        npy_path = os.path.join(geotessera_dir, d, f"{d}.npy")
        if not os.path.exists(npy_path):
            continue
        parts = d.replace("grid_", "").split("_")
        lon, lat = float(parts[0]), float(parts[1])

        data = np.load(npy_path, mmap_mode="r")
        mag = np.abs(data).sum(axis=2)
        zero_frac = np.sum(mag < 0.01) / mag.size

        if zero_frac > zero_threshold:
            regen_set.add((round(lon, 2), round(lat, 2)))
        else:
            keep_tiles.append((lon, lat))

    return sorted(regen_set), keep_tiles


def group_adjacent_tiles(tiles, max_group_extent_deg):
    """Group tiles into spatially adjacent clusters within an extent limit."""
    if not tiles:
        return []

    remaining = list(tiles)
    groups = []

    while remaining:
        seed = remaining.pop(0)
        group = [seed]

        changed = True
        while changed:
            changed = False
            new_remaining = []
            for tile in remaining:
                group_lons = [t[0] for t in group]
                group_lats = [t[1] for t in group]
                candidate_lons = group_lons + [tile[0]]
                candidate_lats = group_lats + [tile[1]]

                is_adjacent = any(
                    abs(tile[0] - t[0]) <= 0.1 + 1e-6
                    and abs(tile[1] - t[1]) <= 0.1 + 1e-6
                    for t in group
                )

                if (
                    is_adjacent
                    and max(candidate_lons) - min(candidate_lons) <= max_group_extent_deg
                    and max(candidate_lats) - min(candidate_lats) <= max_group_extent_deg
                ):
                    group.append(tile)
                    changed = True
                else:
                    new_remaining.append(tile)
            remaining = new_remaining

        groups.append(group)

    return groups


def write_roi_and_manifest(groups, output_dir, group_prefix):
    """Write ROI TIFFs and manifest.json for the grouped tiles."""
    manifest = []

    for i, group in enumerate(groups):
        lons = [t[0] for t in group]
        lats = [t[1] for t in group]

        min_lon, max_lon = min(lons) - 0.05, max(lons) + 0.05
        min_lat, max_lat = min(lats) - 0.05, max(lats) + 0.05

        tile_boxes = [box(lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
                      for lon, lat in group]
        roi_geom = unary_union(tile_boxes)

        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        epsg = determine_utm_epsg(center_lon, center_lat)
        target_crs = CRS.from_epsg(epsg)

        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        corners_x, corners_y = transformer.transform(
            [min_lon, max_lon, min_lon, max_lon],
            [min_lat, min_lat, max_lat, max_lat],
        )
        utm_minx, utm_maxx = min(corners_x), max(corners_x)
        utm_miny, utm_maxy = min(corners_y), max(corners_y)

        width = int(np.ceil((utm_maxx - utm_minx) / PIXEL_SIZE))
        height = int(np.ceil((utm_maxy - utm_miny) / PIXEL_SIZE))
        transform_affine = from_origin(utm_minx, utm_maxy, PIXEL_SIZE, PIXEL_SIZE)

        roi_utm = shp_transform(transformer.transform, roi_geom)
        raster = rasterize(
            [(mapping(roi_utm), 1)],
            out_shape=(height, width),
            transform=transform_affine,
            fill=0, dtype="uint8",
        )

        group_name = f"{group_prefix}_{i:03d}"
        data_dir = os.path.join(output_dir, group_name)
        os.makedirs(data_dir, exist_ok=True)
        roi_tiff = os.path.join(data_dir, "roi.tiff")

        with rasterio.open(
            roi_tiff, "w", driver="GTiff",
            height=height, width=width, count=1, dtype="uint8",
            crs=target_crs, transform=transform_affine,
        ) as dst:
            dst.write(raster, 1)

        valid_pixels = int(np.sum(raster > 0))
        total_pixels = height * width
        extent_km = (round((utm_maxx - utm_minx) / 1000, 1),
                     round((utm_maxy - utm_miny) / 1000, 1))

        entry = {
            "group_name": group_name,
            "tiles": [[float(lon), float(lat)] for lon, lat in group],
            "roi_tiff": roi_tiff,
            "data_dir": data_dir,
            "epsg": epsg,
            "extent_km": extent_km,
            "raster_size": (width, height),
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
            "valid_fraction": round(valid_pixels / total_pixels, 3) if total_pixels > 0 else 0,
        }
        manifest.append(entry)

        print(
            f"  {group_name}: {len(group)} tiles, "
            f"{extent_km[0]}x{extent_km[1]} km, "
            f"{width}x{height} px, "
            f"valid={valid_pixels}/{total_pixels} ({entry['valid_fraction']*100:.1f}%)"
        )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Identify coastal/gap tiles and create ROIs for regeneration.",
    )
    parser.add_argument("--aoi", required=True,
                        help="Path to AOI GeoJSON file")
    parser.add_argument("--geotessera-dir", required=True,
                        help="Directory containing downloaded GeoTessera tiles")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for ROIs and manifest")
    parser.add_argument("--year", type=int, default=2024,
                        help="Year for GeoTessera query (default: 2024)")
    parser.add_argument("--zero-threshold", type=float, default=0.001,
                        help="Fraction of blank pixels to trigger regeneration (default: 0.001)")
    parser.add_argument("--max-group-extent", type=float, default=0.2,
                        help="Max group extent in degrees (default: 0.2)")
    parser.add_argument("--group-prefix", default="group",
                        help="Prefix for group directory names (default: group)")
    parser.add_argument("--download", action="store_true",
                        help="Download GeoTessera tiles before scanning")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Querying GeoTessera registry...")
    covered, missing = query_registry(args.aoi, args.year)
    print(f"Covered by GeoTessera: {len(covered)}")
    print(f"Missing from GeoTessera: {len(missing)}")

    if args.download:
        print("\nDownloading GeoTessera tiles...")
        download_geotessera(args.aoi, args.geotessera_dir, args.year)

    print("\nScanning tiles for blank ocean pixels...")
    regen_tiles, keep_tiles = scan_tiles_for_blanks(
        args.geotessera_dir, missing, args.zero_threshold,
    )
    print(f"Tiles to regenerate: {len(regen_tiles)}")
    print(f"Tiles to keep (no blanks): {len(keep_tiles)}")

    print("\nGrouping into processing clusters...")
    groups = group_adjacent_tiles(regen_tiles, args.max_group_extent)
    print(f"Grouped into {len(groups)} clusters")

    print("\nWriting ROIs and manifest...")
    write_roi_and_manifest(groups, args.output_dir, args.group_prefix)

    keep_path = os.path.join(args.output_dir, "keep_tiles.json")
    with open(keep_path, "w") as f:
        json.dump(
            [{"center_lon": lon, "center_lat": lat} for lon, lat in keep_tiles],
            f, indent=2,
        )

    print(f"\nManifest saved to {os.path.join(args.output_dir, 'manifest.json')}")
    print(f"Keep-list saved to {keep_path}")
    print(f"Total groups: {len(groups)}")
    print(f"Total tiles to regenerate: {len(regen_tiles)}")
    print(f"Total tiles to keep: {len(keep_tiles)}")


if __name__ == "__main__":
    main()
