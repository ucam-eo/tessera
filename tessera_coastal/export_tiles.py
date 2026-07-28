#!/usr/bin/env python3
"""
Export TESSERA embeddings as sliver-free 0.1-degree GeoTIFF tiles.

Combines two sources:
  - GeoTessera precomputed tiles (copied first; these have proper overlapping
    extents and take priority)
  - Gap-fill tiles from the coastal inference pipeline (extracted from stitched
    representations with a configurable pixel overlap buffer to prevent slivers
    between adjacent tiles)

Output:
  <output-dir>/npy/<tile_name>/<tile_name>.npy + metadata.json
  <output-dir>/geotiff/<tile_name>.tif
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer


def tile_name(lon, lat):
    return f"grid_{lon:.2f}_{lat:.2f}"


def copy_geotessera_tiles(geotessera_dir, export_npy, keep_set=None):
    """Copy GeoTessera tiles into the export directory.

    These tiles have proper overlapping extents and are preferred over
    gap-fill extractions for the same tile centre.
    """
    copied = []
    geotessera_dir = Path(geotessera_dir)
    for entry in sorted(geotessera_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("grid_"):
            continue
        if keep_set is not None and entry.name not in keep_set:
            continue
        npy_file = entry / f"{entry.name}.npy"
        meta_file = entry / "metadata.json"
        if not npy_file.exists():
            continue

        out_dir = export_npy / entry.name
        if out_dir.exists() and (out_dir / f"{entry.name}.npy").exists():
            copied.append(entry.name)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        data = np.load(npy_file)
        np.save(out_dir / f"{entry.name}.npy", data)

        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            meta["source"] = "geotessera"
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

        copied.append(entry.name)

    return copied


def split_gap_group(group, gaps_dir, export_npy, overlap_px, year, skip_tiles=None):
    """Split a gap group's stitched representation into individual 0.1-deg tiles.

    Each tile is extended by overlap_px pixels beyond the exact 0.1-degree
    boundary (clamped to the stitched raster extent) so that adjacent tiles
    overlap and no slivers appear between them.
    """
    group_name = group["group_name"]
    data_dir = Path(group["data_dir"])
    if not data_dir.is_absolute():
        data_dir = Path(gaps_dir) / group_name
    stitch_path = data_dir / "stitched_representation.npy"
    roi_path = data_dir / "roi.tiff"
    epsg = group["epsg"]
    tiles = group["tiles"]

    if not stitch_path.exists():
        print(f"  SKIP {group_name}: no stitched_representation.npy")
        return []

    if group.get("valid_fraction", 1) < 0.001:
        print(f"  SKIP {group_name}: valid_fraction={group['valid_fraction']:.3f}")
        return []

    embedding = np.load(stitch_path)
    with rasterio.open(roi_path) as ds:
        roi_transform = ds.transform

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    results = []
    for lon, lat in tiles:
        name = tile_name(lon, lat)
        if skip_tiles and name in skip_tiles:
            continue

        utm_west, utm_south = transformer.transform(lon - 0.05, lat - 0.05)
        utm_east, utm_north = transformer.transform(lon + 0.05, lat + 0.05)

        col_start = int(round((utm_west - roi_transform.c) / roi_transform.a)) - overlap_px
        col_end = int(round((utm_east - roi_transform.c) / roi_transform.a)) + overlap_px
        row_start = int(round((utm_north - roi_transform.f) / roi_transform.e)) - overlap_px
        row_end = int(round((utm_south - roi_transform.f) / roi_transform.e)) + overlap_px

        col_start = max(0, col_start)
        col_end = min(embedding.shape[1], col_end)
        row_start = max(0, row_start)
        row_end = min(embedding.shape[0], row_end)

        if row_end <= row_start or col_end <= col_start:
            print(f"  SKIP tile {name} in {group_name}: empty extraction window")
            continue

        tile_data = embedding[row_start:row_end, col_start:col_end, :]

        tile_origin_x = roi_transform.c + col_start * roi_transform.a
        tile_origin_y = roi_transform.f + row_start * roi_transform.e
        tile_transform = Affine(10.0, 0.0, tile_origin_x, 0.0, -10.0, tile_origin_y)

        out_dir = export_npy / name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{name}.npy", tile_data)

        meta = {
            "tile_name": name,
            "center_lon": lon,
            "center_lat": lat,
            "year": year,
            "shape": list(tile_data.shape),
            "dtype": "float32",
            "crs": f"EPSG:{epsg}",
            "transform": [
                tile_transform.a, tile_transform.b, tile_transform.c,
                tile_transform.d, tile_transform.e, tile_transform.f,
            ],
            "source": "tessera_coastal_gapfill",
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        results.append(name)

    return results


def export_tile_to_geotiff(tile_dir, export_tiff):
    """Convert a single tile from npy + metadata to a multi-band GeoTIFF."""
    name = tile_dir.name
    npy_file = tile_dir / f"{name}.npy"
    meta_file = tile_dir / "metadata.json"

    if not npy_file.exists() or not meta_file.exists():
        return False

    data = np.load(npy_file)
    with open(meta_file) as f:
        meta = json.load(f)

    h, w, bands = data.shape
    t = meta["transform"]
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

    out_path = export_tiff / f"{name}.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=h, width=w, count=bands, dtype="float32",
        crs=meta["crs"], transform=transform,
        compress="zstd", predictor=2,
        tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        for b in range(bands):
            dst.write(data[:, :, b], b + 1)
        dst.update_tags(
            source=meta.get("source", "unknown"),
            year=str(meta.get("year", "")),
            tile_name=name,
        )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export TESSERA embeddings as sliver-free GeoTIFF tiles.",
    )
    parser.add_argument("--geotessera-dir", required=True,
                        help="Directory with downloaded GeoTessera tiles")
    parser.add_argument("--gaps-dir", required=True,
                        help="Directory with gap-fill groups (contains manifest.json)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for exported tiles")
    parser.add_argument("--overlap-px", type=int, default=20,
                        help="Pixel overlap buffer for gap-fill tiles (default: 20)")
    parser.add_argument("--year", type=int, default=2024,
                        help="Year tag for metadata (default: 2024)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tiles that already exist in the output")
    args = parser.parse_args()

    export_npy = Path(args.output_dir) / "npy"
    export_tiff = Path(args.output_dir) / "geotiff"
    export_npy.mkdir(parents=True, exist_ok=True)
    export_tiff.mkdir(parents=True, exist_ok=True)

    gaps_dir = Path(args.gaps_dir)
    manifest_path = gaps_dir / "manifest.json"
    keep_path = gaps_dir / "keep_tiles.json"

    keep_set = None
    if keep_path.exists():
        with open(keep_path) as f:
            keep_list = json.load(f)
        keep_set = {tile_name(t["center_lon"], t["center_lat"]) for t in keep_list}
        print(f"Inland tiles to keep from GeoTessera: {len(keep_set)}")

    # Phase 1: GeoTessera tiles (these have proper overlapping extents)
    print("=" * 60)
    print("PHASE 1: Copying GeoTessera tiles")
    print("=" * 60)
    gt_tiles = copy_geotessera_tiles(args.geotessera_dir, export_npy, keep_set)
    gt_set = set(gt_tiles)
    print(f"GeoTessera tiles: {len(gt_tiles)}")

    # Phase 2: Gap-fill tiles (skip those already covered by GeoTessera)
    print("\n" + "=" * 60)
    print("PHASE 2: Splitting gap-fill embeddings into 0.1-degree tiles")
    print("=" * 60)
    with open(manifest_path) as f:
        manifest = json.load(f)

    gap_tiles = []
    for group in manifest:
        print(f"\nProcessing {group['group_name']} ({len(group['tiles'])} tiles)...")
        names = split_gap_group(
            group, str(gaps_dir), export_npy, args.overlap_px, args.year,
            skip_tiles=gt_set,
        )
        gap_tiles.extend(names)
        for n in names:
            print(f"  -> {n}")

    print(f"\nGap-fill tiles created: {len(gap_tiles)}")

    # Phase 3: Export to GeoTIFF
    print("\n" + "=" * 60)
    print("PHASE 3: Exporting all tiles to GeoTIFF")
    print("=" * 60)
    all_tile_dirs = sorted(export_npy.iterdir())
    written = 0
    skipped = 0
    for i, tile_dir in enumerate(all_tile_dirs):
        if not tile_dir.is_dir():
            continue
        if args.skip_existing and (export_tiff / f"{tile_dir.name}.tif").exists():
            skipped += 1
            continue
        if export_tile_to_geotiff(tile_dir, export_tiff):
            written += 1
        if (i + 1) % 20 == 0 or i == len(all_tile_dirs) - 1:
            print(f"  {i + 1}/{len(all_tile_dirs)} tiles processed...")

    print(f"\nGeoTIFFs written: {written}")
    if skipped:
        print(f"GeoTIFFs skipped (already existed): {skipped}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"GeoTessera tiles:  {len(gt_tiles)}")
    print(f"Gap-fill tiles:    {len(gap_tiles)}")
    print(f"Total tiles:       {len(gt_tiles) + len(gap_tiles)}")
    print(f"GeoTIFFs written:  {written + skipped} ({written} new, {skipped} existing)")
    print(f"\nOutput directories:")
    print(f"  NumPy tiles: {export_npy}")
    print(f"  GeoTIFFs:    {export_tiff}")


if __name__ == "__main__":
    main()
