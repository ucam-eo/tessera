"""Offline S2 resume regression tests using real STAC Items and GeoTIFFs.

Run from the repository root with the preprocessing dependencies and pytest:
    python -m pytest tests/test_s2_resume.py
"""
import datetime
from collections import Counter

import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from tessera_preprocessing import s2_fast_processor as s2


@pytest.fixture(params=["mpc", "aws"])
def mosaic_case(request, tmp_path, monkeypatch):
    mapping = s2.MPC_BAND_MAPPING if request.param == "mpc" else s2.AWS_BAND_MAPPING
    monkeypatch.setattr(s2, "BAND_MAPPING", mapping)
    monkeypatch.setattr(s2, "S2_BANDS", list(mapping))
    monkeypatch.setattr(s2, "DATA_SOURCE", request.param)
    monkeypatch.setattr(s2, "TEMP_DIR", str(tmp_path))

    crs = rasterio.crs.CRS.from_epsg(32631)
    transform = from_origin(500000, 5800000, 10, 10)
    bounds = (500000, 5799960, 500040, 5800000)
    bbox = transform_bounds(crs, "EPSG:4326", *bounds)
    left, bottom, right, top = bbox
    geometry = {"type": "Polygon", "coordinates": [[
        [left, bottom], [right, bottom], [right, top],
        [left, top], [left, bottom],
    ]]}
    template = dict(driver="GTiff", width=4, height=4, count=1,
                    crs=crs, transform=transform)
    # The first source has clouds and no-data where the second has clear land.
    classifications = [
        np.array([[4, 4, 9, 9], [4, 4, 9, 9],
                  [0, 0, 9, 9], [0, 0, 9, 9]], dtype=np.uint8),
        np.array([[5, 5, 5, 5], [5, 5, 5, 5],
                  [5, 5, 5, 5], [5, 5, 5, 9]], dtype=np.uint8),
    ]
    items = []
    for index, scl in enumerate(classifications):
        item = pystac.Item(
            id=f"source_{index}", geometry=geometry, bbox=list(bbox),
            datetime=datetime.datetime(2021, 6, 1, 12, index,
                                       tzinfo=datetime.timezone.utc),
            properties={"proj:epsg": 32631, "proj:shape": [4, 4],
                        "proj:transform": list(transform)[:6]},
        )
        for band_index, (asset, name) in enumerate(mapping.items()):
            path = tmp_path / f"{index}_{asset}.tif"
            if name == "scl":
                data = scl
            else:
                data = (1000 + index * 2000 + band_index * 100 +
                        np.arange(16).reshape(4, 4) * 10).astype(np.uint16)
                data[scl == 0] = 0
            with rasterio.open(path, "w", **template, dtype=data.dtype, nodata=0) as dst:
                dst.write(data, 1)
            item.add_asset(asset, pystac.Asset(
                href=str(path), media_type=pystac.MediaType.GEOTIFF,
                extra_fields={"raster:bands": [{"nodata": 0,
                    "data_type": str(data.dtype), "scale": 1, "offset": 0}]},
            ))
        items.append(item)
    mask = np.ones((4, 4), dtype=np.uint8)
    mask[0, 0] = 0
    return dict(date_key="2021-06-01", items=items, tpl=template,
                bbox_proj=bounds, mask_np=mask, out_root=tmp_path / "output",
                res=10, chunksize=16, overwrite=False, min_coverage=5.0,
                partition_id="test")


def output_path(case, name):
    return case["out_root"] / name / f"{case['date_key']}_mosaic.tiff"


def snapshot(case):
    result = {}
    for name in s2.BAND_MAPPING.values():
        with rasterio.open(output_path(case, name)) as src:
            result[name] = src.read(1)
    return result


def record_stack_calls(monkeypatch):
    calls = []
    stack = s2.stackstac.stack

    def recording_stack(*args, **kwargs):
        calls.append(tuple(kwargs["assets"]))
        return stack(*args, **kwargs)

    monkeypatch.setattr(s2.stackstac, "stack", recording_stack)
    return calls


@pytest.mark.parametrize("incomplete", ["missing_band", "missing_bands", "corrupt_band"])
def test_partial_resume_matches_fresh_valid_pixels(mosaic_case, monkeypatch, incomplete):
    case = mosaic_case
    assert s2.process_day(**case)
    before = snapshot(case)
    np.testing.assert_array_equal(before["scl"], [
        [0, 4, 5, 5], [4, 4, 5, 5], [5, 5, 5, 5], [5, 5, 5, 0],
    ])
    assert before["blue"][0, 2] == 3020  # First source is cloudy here.
    assert before["blue"][2, 0] == 3080  # First source has no data here.
    missing = {"red", "swir16"} if incomplete == "missing_bands" else {"blue"}
    for name in missing:
        path = output_path(case, name)
        if incomplete == "corrupt_band":
            path.write_bytes(b"interrupted GeoTIFF write")
        else:
            path.unlink()
    retained = {name: (output_path(case, name).read_bytes(),
                       output_path(case, name).stat().st_mtime_ns)
                for name in before if name not in missing and name != "scl"}
    calls = record_stack_calls(monkeypatch)

    assert s2.process_day(**case)
    after = snapshot(case)
    # SCL-invalid pixels deliberately use random sources; compare valid pixels.
    valid = before["scl"] > 0
    for name in before:
        np.testing.assert_array_equal(after[name][valid], before[name][valid],
                                      err_msg=f"Resuming changed {name}")
    np.testing.assert_array_equal(after["scl"], before["scl"])
    for name, (data, mtime) in retained.items():
        assert output_path(case, name).read_bytes() == data
        assert output_path(case, name).stat().st_mtime_ns == mtime
    # Source SCLs may be reread; completed spectral bands must not be fetched.
    assert Counter(calls) == Counter((asset,) for asset, name in s2.BAND_MAPPING.items()
                                    if name in missing | {"scl"})


def test_completed_day_skips_source_reads(mosaic_case, monkeypatch):
    case = mosaic_case
    assert s2.process_day(**case)
    mtimes = {name: output_path(case, name).stat().st_mtime_ns
              for name in s2.BAND_MAPPING.values()}
    calls = record_stack_calls(monkeypatch)
    assert s2.process_day(**case)
    assert calls == []
    assert mtimes == {name: output_path(case, name).stat().st_mtime_ns for name in mtimes}


def test_single_source_resume_matches_fresh_valid_pixels(mosaic_case):
    case = dict(mosaic_case, items=mosaic_case["items"][1:])
    assert s2.process_day(**case)
    before = snapshot(case)
    output_path(case, "blue").unlink()
    assert s2.process_day(**case)
    after = snapshot(case)
    valid = before["scl"] > 0
    for name in before:
        np.testing.assert_array_equal(after[name][valid], before[name][valid])


def test_overwrite_rebuilds_all_bands(mosaic_case, monkeypatch):
    case = mosaic_case
    assert s2.process_day(**case)
    before = snapshot(case)
    with rasterio.open(output_path(case, "blue"), "r+") as dst:
        dst.write(np.full((4, 4), 9999, dtype=np.uint16), 1)
    calls = record_stack_calls(monkeypatch)
    assert s2.process_day(**dict(case, overwrite=True))
    after = snapshot(case)
    valid = before["scl"] > 0
    for name in before:
        np.testing.assert_array_equal(after[name][valid], before[name][valid])
    assert Counter(calls) == Counter((asset,) for asset in s2.BAND_MAPPING)


def test_resume_does_not_guess_sources_when_scl_read_fails(mosaic_case, monkeypatch):
    case = mosaic_case
    assert s2.process_day(**case)
    output_path(case, "blue").unlink()
    retained = {name: output_path(case, name).read_bytes()
                for name in s2.BAND_MAPPING.values() if name != "blue"}
    calls = []

    def unavailable_stack(*args, **kwargs):
        calls.append(tuple(kwargs["assets"]))
        raise RuntimeError("Source SCL is temporarily unavailable")

    monkeypatch.setattr(s2.stackstac, "stack", unavailable_stack)
    s2.process_day(**case)
    assert not output_path(case, "blue").exists()
    assert all(output_path(case, name).read_bytes() == data
               for name, data in retained.items())
    scl_key = next(asset for asset, name in s2.BAND_MAPPING.items() if name == "scl")
    assert calls == [(scl_key,)]
