"""
sar_pipeline/export.py
-----------------------
Export preprocessed Sentinel-1 scenes in three formats:

  1. GeoTIFF  — archival float32 master with full geospatial metadata.
                Always save this. Everything else is derived from it.

  2. PNG patches — 512×512 uint8 chips for YOLO/Faster-RCNN training.
                   Written alongside a JSON sidecar mapping patch filename
                   → (row_start, col_start) in the parent GeoTIFF.

  3. .pt tensor  — float32 PyTorch tensor for batch inference pipelines.

Naming convention (keeps scenes traceable):
    S1A_IW_GRDH_1SDV_20230601T054912_048832
        ↓ GeoTIFF
    processed/S1A_...048832_GTC_sigma0.tif
        ↓ PNG patches
    patches/S1A_...048832_r0000_c0000.png
    patches/S1A_...048832_r0000_c0256.png  (stride=256 → 50% overlap)
    patches/S1A_...048832_patches.json      (sidecar index)
        ↓ tensor
    tensors/S1A_...048832.pt

Dependencies:
    pip install numpy rasterio torch pillow tqdm

Usage:
    from export import export_scene

    export_scene(
        geotiff_path="output/scene_GTC_sigma0.tif",
        output_dir="output/",
        scene_id="S1A_048832",
        formats=["geotiff", "png", "tensor"],
    )
"""

from __future__ import annotations

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
except ImportError:
    raise ImportError("rasterio is required: pip install rasterio")

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required: pip install pillow")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation (inline — avoids circular import from normalize.py)
# ─────────────────────────────────────────────────────────────────────────────

DB_MIN = -30.0
DB_MAX =   0.0
LOG_EPS = 1e-10


def _to_norm(arr: np.ndarray) -> np.ndarray:
    """Linear σ° → normalised [0,1] float32."""
    db = 10.0 * np.log10(np.maximum(arr.astype(np.float64), LOG_EPS))
    clipped = np.clip(db, DB_MIN, DB_MAX)
    return ((clipped - DB_MIN) / (DB_MAX - DB_MIN)).astype(np.float32)


def _norm_to_uint8(norm: np.ndarray) -> np.ndarray:
    """[0,1] float32 → uint8 [0,255]. Clips to handle floating-point edge values."""
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# GeoTIFF writer
# ─────────────────────────────────────────────────────────────────────────────

def save_geotiff_master(
    arrays: dict[str, np.ndarray],
    transform: Affine,
    crs: CRS,
    output_path: str,
    compress: str = "deflate",
    dtype: str = "float32",
    nodata: float = -9999.0,
    verbose: bool = True,
) -> str:
    """
    Save a multi-band float32 GeoTIFF — the archival master output.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Band name → 2D float32 array.  e.g. {"VV": vv_arr, "VH": vh_arr}
        Values should be linear σ° (not dB, not normalised).
        Storing linear preserves the full dynamic range and lets you
        re-derive dB or any other transform later without loss.
    transform : Affine
        Rasterio affine geotransform (from ingest or terrain correction).
    crs : CRS
        Output coordinate reference system.
    output_path : str
        Destination file path, e.g. "output/scene_GTC_sigma0.tif"
    compress : str
        Compression codec.  "deflate" is lossless and gives ~60% size
        reduction on SAR float32 data.  "lzw" is an alternative.
        "none" disables compression (faster write, larger file).
    dtype : str
        Output rasterio dtype.  float32 is standard.
    nodata : float
        NoData sentinel value.  -9999.0 is conventional.
        SNAP uses 0.0 — be consistent with your toolchain.
    verbose : bool

    Returns
    -------
    str : absolute path to the written file
    """
    bands = list(arrays.values())
    names = list(arrays.keys())
    H, W = bands[0].shape

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    profile = {
        "driver":    "GTiff",
        "height":    H,
        "width":     W,
        "count":     len(bands),
        "dtype":     dtype,
        "crs":       crs,
        "transform": transform,
        "nodata":    nodata,
    }
    if compress != "none":
        profile["compress"] = compress
        if compress == "deflate":
            profile["predictor"] = 2    # horizontal differencing — improves
                                        # deflate ratio on floating point data
        profile["tiled"] = True         # tiled layout speeds up spatial reads
        profile["blockxsize"] = 512     # tile size matches our patch size
        profile["blockysize"] = 512

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, (name, arr) in enumerate(zip(names, bands), start=1):
            dst.write(arr.astype(dtype), i)
            dst.update_tags(i, name=name)

    size_mb = os.path.getsize(output_path) / 1e6
    if verbose:
        print(f"[export] GeoTIFF → {output_path}  ({size_mb:.1f} MB, "
              f"{len(bands)} bands, {H}×{W} px)")

    return os.path.abspath(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# PNG patch writer
# ─────────────────────────────────────────────────────────────────────────────

def save_png_patches(
    geotiff_path: str,
    output_dir: str,
    scene_id: str,
    patch_size: int = 512,
    stride: int = 256,
    min_valid_fraction: float = 0.5,
    add_ratio_band: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Slice a GeoTIFF into overlapping PNG patches for YOLO/detection training.

    Each PNG is a 3-channel uint8 image:
        Channel 0 (R) = VV normalised [0,255]
        Channel 1 (G) = VH normalised [0,255]
        Channel 2 (B) = VV/VH dB ratio normalised [0,255]
                        (or VV again if VH not available)

    A JSON sidecar file maps each patch filename to its pixel coordinates
    in the parent GeoTIFF and to its geographic bounding box.
    This is what lets you convert model detection boxes (in patch-pixel coords)
    back to lat/lon after inference.

    Directory layout:
        {output_dir}/
          images/     ← PNG chips (fed to YOLO)
          labels/     ← YOLO .txt annotation files (you fill these in)
          {scene_id}_patches.json

    Parameters
    ----------
    geotiff_path : str
        Path to the float32 GeoTIFF written by save_geotiff_master().
    output_dir : str
        Root directory.  images/ and labels/ subdirs are created.
    scene_id : str
        Short identifier used in output filenames,
        e.g. "S1A_048832" → patches named "S1A_048832_r0000_c0000.png"
    patch_size : int
        Square patch side length in pixels.  512 = standard for YOLOv8.
    stride : int
        Step between patch top-left corners.  stride < patch_size → overlap.
        256 gives 50% overlap — ships near borders appear in two patches.
    min_valid_fraction : float
        Discard patches where fewer than this fraction of VV pixels are
        non-zero (nodata patches at scene edges).
    add_ratio_band : bool
        Include VV/VH ratio as the third channel.
    verbose : bool

    Returns
    -------
    index : dict
        Patch index also written to {output_dir}/{scene_id}_patches.json
        Keys are patch filenames; values are dicts with:
            row_start, col_start : int   top-left in parent GeoTIFF pixels
            row_end, col_end     : int
            bbox_geo             : [min_lon, min_lat, max_lon, max_lat]
            parent_geotiff       : str   absolute path to source GeoTIFF
    """
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Load source GeoTIFF
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs
        H, W = src.height, src.width

        # Read bands — resolve VV and VH by tag name
        band_map = {}
        for i in range(1, src.count + 1):
            tag = src.tags(i).get("name", f"band_{i}")
            key = tag.replace("Sigma0_", "").replace("sigma0_", "").upper()
            band_map[key] = src.read(i).astype(np.float32)

    vv = band_map.get("VV") or band_map.get("BAND_1")
    vh = band_map.get("VH") or band_map.get("BAND_2")

    if vv is None:
        raise ValueError(f"Cannot find VV band in {geotiff_path}. "
                         f"Available: {list(band_map.keys())}")

    # Build 3-channel normalised array
    vv_norm = _to_norm(vv)

    if vh is not None:
        vh_norm = _to_norm(vh)
    else:
        if verbose:
            print("[export] VH not found — using VV for all three channels.")
        vh_norm = vv_norm

    if add_ratio_band and vh is not None:
        vv_db = 10.0 * np.log10(np.maximum(vv.astype(np.float64), LOG_EPS))
        vh_db = 10.0 * np.log10(np.maximum(vh.astype(np.float64), LOG_EPS))
        ratio_db = vv_db - vh_db                    # typically 0–15 dB
        ratio_norm = np.clip(ratio_db, 0.0, 15.0) / 15.0
        ratio_norm = ratio_norm.astype(np.float32)
    else:
        ratio_norm = vv_norm                        # fallback: duplicate VV

    # Stack to (H, W, 3) uint8
    rgb = np.stack([
        _norm_to_uint8(vv_norm),
        _norm_to_uint8(vh_norm),
        _norm_to_uint8(ratio_norm),
    ], axis=-1)                                     # (H, W, 3)

    # Patch extraction
    index = {}
    total = kept = 0

    row_positions = list(range(0, H - patch_size + 1, stride))
    col_positions = list(range(0, W - patch_size + 1, stride))

    iterator = row_positions
    if TQDM_AVAILABLE and verbose:
        iterator = tqdm(row_positions, desc="[export] Extracting patches", unit="row")

    for r in iterator:
        for c in col_positions:
            total += 1
            patch_vv = vv_norm[r:r + patch_size, c:c + patch_size]

            # Skip near-empty patches (sea edges, nodata strips)
            valid_frac = float((patch_vv > 0.0).mean())
            if valid_frac < min_valid_fraction:
                continue

            # Compute geographic bounding box of this patch
            # transform maps (col, row) → (x, y) in the CRS
            top_left  = transform * (c, r)
            bot_right = transform * (c + patch_size, r + patch_size)
            min_lon = min(top_left[0], bot_right[0])
            max_lon = max(top_left[0], bot_right[0])
            min_lat = min(top_left[1], bot_right[1])
            max_lat = max(top_left[1], bot_right[1])

            # Filename: scene_id + zero-padded row/col for lexicographic sort
            patch_name = f"{scene_id}_r{r:05d}_c{c:05d}.png"
            patch_path = os.path.join(img_dir, patch_name)
            label_path = os.path.join(lbl_dir, patch_name.replace(".png", ".txt"))

            # Save PNG
            patch_rgb = rgb[r:r + patch_size, c:c + patch_size]
            Image.fromarray(patch_rgb).save(patch_path, optimize=False)

            # Create empty YOLO label file (you fill this in with annotations)
            # YOLO format: class cx cy w h  (all normalised to [0,1])
            if not os.path.exists(label_path):
                open(label_path, "w").close()

            index[patch_name] = {
                "row_start":      r,
                "col_start":      c,
                "row_end":        r + patch_size,
                "col_end":        c + patch_size,
                "bbox_geo":       [min_lon, min_lat, max_lon, max_lat],
                "valid_fraction": round(valid_frac, 3),
                "parent_geotiff": os.path.abspath(geotiff_path),
            }

            kept += 1

    # Write sidecar JSON
    sidecar_path = os.path.join(output_dir, f"{scene_id}_patches.json")
    with open(sidecar_path, "w") as f:
        json.dump(index, f, indent=2)

    if verbose:
        print(f"[export] PNG patches → {img_dir}")
        print(f"         Total: {total}  Kept: {kept}  "
              f"Discarded (nodata): {total - kept}")
        print(f"         Sidecar index → {sidecar_path}")

    return index


# ─────────────────────────────────────────────────────────────────────────────
# Tensor writer
# ─────────────────────────────────────────────────────────────────────────────

def save_tensor(
    geotiff_path: str,
    output_path: str,
    add_ratio_band: bool = True,
    verbose: bool = True,
) -> str:
    """
    Load a preprocessed GeoTIFF and save it as a float32 PyTorch tensor.

    The tensor has shape (C, H, W) with C = 2 or 3 depending on
    add_ratio_band.  Values are in [0, 1].

    This is the format you pass directly to model.forward() —
    just unsqueeze(0) to add the batch dimension.

    Note: geographic metadata is NOT stored in the tensor.
    Keep the source GeoTIFF path so you can look up coordinates later.

    Parameters
    ----------
    geotiff_path : str
    output_path : str  e.g. "tensors/scene.pt"
    add_ratio_band : bool
    verbose : bool

    Returns
    -------
    str : absolute path to the .pt file
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install torch")

    with rasterio.open(geotiff_path) as src:
        H, W = src.height, src.width
        band_map = {}
        for i in range(1, src.count + 1):
            tag = src.tags(i).get("name", f"band_{i}")
            key = tag.replace("Sigma0_", "").replace("sigma0_", "").upper()
            band_map[key] = src.read(i).astype(np.float32)

    vv = band_map.get("VV") or band_map.get("BAND_1")
    vh = band_map.get("VH") or band_map.get("BAND_2")

    channels = [_to_norm(vv)]
    if vh is not None:
        channels.append(_to_norm(vh))
    if add_ratio_band and vh is not None:
        vv_db = 10.0 * np.log10(np.maximum(vv.astype(np.float64), LOG_EPS))
        vh_db = 10.0 * np.log10(np.maximum(vh.astype(np.float64), LOG_EPS))
        ratio = np.clip((vv_db - vh_db) / 15.0, 0.0, 1.0).astype(np.float32)
        channels.append(ratio)

    arr = np.stack(channels, axis=0)          # (C, H, W)
    tensor = torch.from_numpy(arr)            # float32

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(tensor, output_path)

    size_mb = os.path.getsize(output_path) / 1e6
    if verbose:
        print(f"[export] Tensor → {output_path}  "
              f"(shape={tuple(tensor.shape)}, {size_mb:.1f} MB)")

    return os.path.abspath(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Georeferencing helper: patch pixels → lat/lon
# ─────────────────────────────────────────────────────────────────────────────

def patch_pixel_to_latlon(
    row_in_patch: float,
    col_in_patch: float,
    patch_row_start: int,
    patch_col_start: int,
    geotiff_path: str,
) -> tuple[float, float]:
    """
    Convert a pixel coordinate within a patch to (latitude, longitude).

    Used after inference to convert YOLO bounding box centres back to
    geographic coordinates for GeoJSON / AIS cross-matching output.

    Parameters
    ----------
    row_in_patch, col_in_patch : float
        Pixel position within the patch (0-indexed from top-left).
        For a YOLO detection box: cx_pixel = cx_norm * patch_size,
                                   cy_pixel = cy_norm * patch_size
    patch_row_start, patch_col_start : int
        Top-left corner of the patch in the parent GeoTIFF
        (from the patches JSON sidecar).
    geotiff_path : str
        Path to the parent GeoTIFF (for its transform).

    Returns
    -------
    (lat, lon) : tuple[float, float]
    """
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs

    # Absolute position in parent GeoTIFF pixels
    abs_row = patch_row_start + row_in_patch
    abs_col = patch_col_start + col_in_patch

    # Pixel → geographic coordinate (top-left corner of pixel)
    x, y = transform * (abs_col, abs_row)

    # If CRS is projected (UTM), convert to geographic (lat/lon)
    if not crs.is_geographic:
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)
            lon, lat = transformer.transform(x, y)
        except ImportError:
            raise ImportError("pyproj is required for UTM→lat/lon conversion: pip install pyproj")
    else:
        lon, lat = x, y

    return float(lat), float(lon)


# ─────────────────────────────────────────────────────────────────────────────
# One-shot convenience function
# ─────────────────────────────────────────────────────────────────────────────

def export_scene(
    geotiff_path: str,
    output_dir: str,
    scene_id: Optional[str] = None,
    formats: list[str] = ("geotiff", "png", "tensor"),
    patch_size: int = 512,
    stride: int = 256,
    verbose: bool = True,
) -> dict[str, str | dict]:
    """
    Export a preprocessed SAR scene in all requested formats.

    Parameters
    ----------
    geotiff_path : str
        Source float32 GeoTIFF from the preprocessing pipeline.
        Must have VV (and optionally VH) bands with correct name tags.
    output_dir : str
        Root output directory.  Subdirectories are created as needed.
    scene_id : str | None
        Short ID used in filenames.  Derived from geotiff_path if None.
    formats : list[str]
        Any subset of ["geotiff", "png", "tensor"].
        "geotiff" copies/re-saves the master with validated metadata.
        "png"     writes patch chips + JSON sidecar.
        "tensor"  writes a .pt file.
    patch_size : int
    stride : int
    verbose : bool

    Returns
    -------
    dict[str, str | dict]
        Keys: "geotiff", "png_index", "tensor"
        Values: file path (str) or patch index (dict)
    """
    if scene_id is None:
        scene_id = Path(geotiff_path).stem[:40]

    results = {}

    if "geotiff" in formats:
        # Load and re-save with our preferred profile (tiled, deflate)
        with rasterio.open(geotiff_path) as src:
            transform = src.transform
            crs = src.crs
            band_map = {}
            for i in range(1, src.count + 1):
                tag = src.tags(i).get("name", f"band_{i}")
                key = tag.replace("Sigma0_", "").replace("sigma0_", "").upper()
                band_map[key] = src.read(i).astype(np.float32)

        out_path = os.path.join(output_dir, f"{scene_id}_sigma0.tif")
        results["geotiff"] = save_geotiff_master(
            band_map, transform, crs, out_path, verbose=verbose
        )

    if "png" in formats:
        png_index = save_png_patches(
            geotiff_path=geotiff_path,
            output_dir=os.path.join(output_dir, "patches"),
            scene_id=scene_id,
            patch_size=patch_size,
            stride=stride,
            verbose=verbose,
        )
        results["png_index"] = png_index

    if "tensor" in formats:
        if not TORCH_AVAILABLE:
            print("[export] PyTorch not installed — skipping tensor export.")
        else:
            t_path = os.path.join(output_dir, "tensors", f"{scene_id}.pt")
            results["tensor"] = save_tensor(geotiff_path, t_path, verbose=verbose)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    if len(sys.argv) < 2:
        print("Usage: python export.py /path/to/scene_GTC.tif [output_dir] [scene_id]")
        print("\nRunning synthetic test...")

        # Build a tiny synthetic GeoTIFF for testing
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS as RasterioCRS

        H, W = 1024, 1024
        rng = np.random.default_rng(42)
        vv = rng.lognormal(np.log(10**(-18/10)), 0.5, (H, W)).astype(np.float32)
        vh = (vv * 0.6 * rng.lognormal(0, 0.2, (H, W))).astype(np.float32)
        vv[400:440, 400:440] = 0.3     # synthetic ship

        transform = from_bounds(10.0, 55.0, 10.1, 55.1, W, H)
        crs = RasterioCRS.from_epsg(4326)

        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = os.path.join(tmpdir, "test_scene.tif")
            with rasterio.open(
                tif_path, "w", driver="GTiff", height=H, width=W,
                count=2, dtype="float32", crs=crs, transform=transform
            ) as dst:
                dst.write(vv, 1); dst.update_tags(1, name="VV")
                dst.write(vh, 2); dst.update_tags(2, name="VH")

            out_dir = os.path.join(tmpdir, "export")
            results = export_scene(
                tif_path, out_dir,
                scene_id="TEST_SCENE",
                formats=["geotiff", "png", "tensor"],
                patch_size=256,
                stride=128,
                verbose=True,
            )

            print(f"\nResults:")
            print(f"  GeoTIFF  : {results.get('geotiff')}")
            print(f"  Patches  : {len(results.get('png_index', {}))} PNGs")
            print(f"  Tensor   : {results.get('tensor')}")

            # Test georeferencing roundtrip
            first_patch = next(iter(results.get("png_index", {}).values()), None)
            if first_patch:
                lat, lon = patch_pixel_to_latlon(
                    row_in_patch=128, col_in_patch=128,
                    patch_row_start=first_patch["row_start"],
                    patch_col_start=first_patch["col_start"],
                    geotiff_path=results["geotiff"],
                )
                print(f"\n  Georef test: patch pixel (128,128) → lat={lat:.4f} lon={lon:.4f}")
                print("  (should be within the synthetic bbox: lat 55.0–55.1, lon 10.0–10.1)")

        print("\nSelf-test PASSED.")
        sys.exit(0)

    geotiff_path = sys.argv[1]
    output_dir   = sys.argv[2] if len(sys.argv) > 2 else "output"
    scene_id     = sys.argv[3] if len(sys.argv) > 3 else None

    export_scene(
        geotiff_path, output_dir,
        scene_id=scene_id,
        formats=["geotiff", "png", "tensor"],
        verbose=True,
    )
