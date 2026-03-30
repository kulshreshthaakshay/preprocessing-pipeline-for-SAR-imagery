"""
sar_pipeline/utils/io.py
------------------------
GeoTIFF save/load helpers shared across all pipeline steps.

Usage:
    from utils.io import save_geotiff, load_geotiff

Dependencies:
    pip install rasterio numpy
"""

from __future__ import annotations
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS


def save_geotiff(
    path: str,
    arrays: dict[str, np.ndarray],
    transform: Affine,
    crs: CRS,
    dtype: str = "float32",
    nodata: float = -9999.0,
) -> None:
    """
    Save one or more 2D arrays to a multi-band GeoTIFF.

    Parameters
    ----------
    path : str
        Output file path, e.g. "output/calibrated.tif"
    arrays : dict[str, np.ndarray]
        Mapping of band name → 2D array. Band order follows dict insertion order.
        e.g. {"VV": vv_arr, "VH": vh_arr}
    transform : Affine
        Geospatial transform (from ingest SARScene).
    crs : CRS
        Coordinate reference system (from ingest SARScene).
    dtype : str
        Output rasterio dtype string. Default "float32".
    nodata : float
        NoData sentinel value written to the file metadata.
    """
    bands = list(arrays.values())
    names = list(arrays.keys())
    h, w = bands[0].shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=len(bands),
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
        nodata=nodata,
        BIGTIFF="YES",
    ) as dst:
        for i, (name, arr) in enumerate(zip(names, bands), start=1):
            dst.write(arr.astype(dtype), i)
            dst.update_tags(i, name=name)

    print(f"[io] Saved {len(bands)}-band GeoTIFF → {path}")


def load_geotiff(path: str) -> tuple[dict[str, np.ndarray], Affine, CRS]:
    """
    Load a multi-band GeoTIFF back into numpy arrays.

    Returns
    -------
    arrays : dict[str, np.ndarray]
        Band index (1-based str) → array. e.g. {"band_1": arr, "band_2": arr}
    transform : Affine
    crs : CRS
    """
    with rasterio.open(path) as src:
        transform = src.transform
        crs = src.crs
        arrays = {}
        for i in range(1, src.count + 1):
            tag = src.tags(i).get("name", f"band_{i}")
            arrays[tag] = src.read(i)
    return arrays, transform, crs
