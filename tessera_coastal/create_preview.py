#!/usr/bin/env python3
"""
Create 3-band PCA false-colour previews from multi-band TESSERA embeddings.

Projects the embedding dimensions onto their top-3 principal components to
produce an RGBA GeoTIFF per tile: RGB channels are the PCA projection
stretched to 0--255, and the alpha channel masks truly empty pixels (exact-zero
padding at tile edges).

These previews are intended for quick visual inspection in GIS software.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio


def compute_pca_from_sample(tiff_files, n_components=3, sample_pixels=100000):
    """Compute PCA components and global stretch from a random pixel sample."""
    rng = np.random.default_rng(42)
    samples = []
    per_file = max(sample_pixels // len(tiff_files), 200)

    for f in tiff_files:
        with rasterio.open(f) as ds:
            data = ds.read()
        pixels = data.reshape(data.shape[0], -1).T
        mag = np.abs(pixels).sum(axis=1)
        valid = mag > 0.01
        valid_pixels = pixels[valid]
        if len(valid_pixels) == 0:
            continue
        idx = rng.choice(len(valid_pixels), min(per_file, len(valid_pixels)),
                         replace=False)
        samples.append(valid_pixels[idx])

    all_samples = np.concatenate(samples, axis=0)
    mean = all_samples.mean(axis=0)
    centered = all_samples - mean
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    top_idx = np.argsort(eigenvalues)[::-1][:n_components]
    components = eigenvectors[:, top_idx].T

    projected = centered @ components.T
    global_lo = np.percentile(projected, 1, axis=0)
    global_hi = np.percentile(projected, 99, axis=0)

    return mean, components, global_lo, global_hi


def project_tile(tiff_path, mean, components, global_lo, global_hi, out_path):
    """Project a single tile to a 3-band RGBA GeoTIFF."""
    with rasterio.open(tiff_path) as ds:
        data = ds.read()
        profile = ds.profile.copy()

    bands, h, w = data.shape
    pixels = data.reshape(bands, -1).T
    mag = np.abs(pixels).sum(axis=1)
    valid = mag > 0.01

    projected = (pixels - mean) @ components.T

    rgb = np.zeros((len(pixels), 3), dtype=np.float32)
    for c in range(3):
        span = max(global_hi[c] - global_lo[c], 1e-8)
        rgb[:, c] = np.clip((projected[:, c] - global_lo[c]) / span, 0, 1)

    rgb_img = (rgb * 255).astype(np.uint8).reshape(h, w, 3)
    alpha = np.where(valid, 255, 0).astype(np.uint8).reshape(h, w)

    profile.update(
        count=4, dtype="uint8",
        compress="deflate", predictor=2,
        tiled=True, blockxsize=256, blockysize=256,
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        for b in range(3):
            dst.write(rgb_img[:, :, b], b + 1)
        dst.write(alpha, 4)


def main():
    parser = argparse.ArgumentParser(
        description="Create PCA false-colour preview tiles from TESSERA embeddings.",
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing multi-band GeoTIFF tiles")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for preview tiles")
    parser.add_argument("--sample-pixels", type=int, default=100000,
                        help="Number of pixels to sample for PCA (default: 100000)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(input_dir.glob("*.tif"))
    print(f"Found {len(tiff_files)} GeoTIFF tiles")

    if not tiff_files:
        print("No tiles found. Nothing to do.")
        return

    print("Computing PCA from sample pixels...")
    mean, components, global_lo, global_hi = compute_pca_from_sample(
        tiff_files, sample_pixels=args.sample_pixels,
    )
    np.savez(output_dir / "pca_params.npz", mean=mean, components=components,
             global_lo=global_lo, global_hi=global_hi)
    print(f"Global stretch: lo={global_lo}, hi={global_hi}")

    print("Projecting tiles to 3-band RGBA...")
    for i, f in enumerate(tiff_files):
        out = output_dir / f.name
        project_tile(f, mean, components, global_lo, global_hi, out)
        if (i + 1) % 20 == 0 or i == len(tiff_files) - 1:
            print(f"  {i + 1}/{len(tiff_files)}")

    total = sum(f.stat().st_size for f in output_dir.glob("*.tif"))
    print(f"\nPreview tiles saved to {output_dir}")
    print(f"Total preview size: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
