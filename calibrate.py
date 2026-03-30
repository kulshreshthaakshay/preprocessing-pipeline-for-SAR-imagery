"""
sar_pipeline/calibrate.py
--------------------------
Step 2: Radiometric calibration — convert raw DN values to sigma-nought (σ°).

Formula:
    σ°(i,j) = DN(i,j)² / A(i,j)²

where A(i,j) is the sigmaNought calibration LUT bilinearly interpolated
from the sparse calibration annotation XML to the full image grid.

This module is snappy-free and operates entirely in numpy/scipy.

Dependencies:
    pip install numpy scipy lxml

Usage:
    from ingest import ingest_safe
    from calibrate import calibrate_scene

    scene = ingest_safe("/data/S1A_....SAFE")
    scene = calibrate_scene(scene)
    # scene.vv and scene.vh now contain σ° in linear power (float32)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

try:
    from lxml import etree
except ImportError:
    raise ImportError("lxml is required: pip install lxml")

try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:
    raise ImportError("scipy is required: pip install scipy")

from ingest import SARScene


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _parse_calibration_xml(xml_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a Sentinel-1 calibration annotation XML and extract the
    sigmaNought LUT as arrays suitable for interpolation.

    The XML stores one <calibrationVector> per azimuth line, each containing
    sparse range sample positions and LUT values.

    Returns
    -------
    lines : np.ndarray  shape (L,)
        Azimuth line indices (one per calibration vector).
    pixels : np.ndarray  shape (P,)
        Range pixel indices (same for all vectors in a GRD product).
    lut : np.ndarray  shape (L, P)
        sigmaNought calibration LUT values. These are amplitudes (NOT power),
        so the calibration formula is σ° = DN² / lut².
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    vectors = root.findall(".//calibrationVector")
    if not vectors:
        raise ValueError(f"No calibrationVector elements found in {xml_path}")

    lines_list = []
    pixels_ref = None
    lut_rows = []

    for vec in vectors:
        # Azimuth line index
        line_el = vec.find("line")
        if line_el is None or line_el.text is None:
            continue
        line_idx = int(line_el.text.strip())

        # Range pixel positions (same across all vectors for GRD)
        pixel_el = vec.find("pixel")
        if pixel_el is None or pixel_el.text is None:
            continue
        pixels = np.array(pixel_el.text.strip().split(), dtype=np.float64)

        # sigmaNought LUT values
        sigma_el = vec.find("sigmaNought")
        if sigma_el is None or sigma_el.text is None:
            continue
        sigma_vals = np.array(sigma_el.text.strip().split(), dtype=np.float64)

        if len(pixels) != len(sigma_vals):
            raise ValueError(
                f"Pixel count {len(pixels)} != LUT count {len(sigma_vals)} "
                f"at line {line_idx} in {xml_path}"
            )

        if pixels_ref is None:
            pixels_ref = pixels
        elif not np.allclose(pixels, pixels_ref, atol=1):
            # GRD products should have consistent pixel positions
            # If they differ, use the first vector's positions and hope for the best
            pass

        lines_list.append(line_idx)
        lut_rows.append(sigma_vals)

    lines = np.array(lines_list, dtype=np.float64)
    lut = np.array(lut_rows, dtype=np.float64)  # shape (L, P)

    return lines, pixels_ref, lut


# ---------------------------------------------------------------------------
# LUT interpolation
# ---------------------------------------------------------------------------

def _build_lut_interpolator(
    lines: np.ndarray,
    pixels: np.ndarray,
    lut: np.ndarray,
) -> RegularGridInterpolator:
    """
    Build a bilinear interpolator from the sparse calibration LUT.

    Parameters
    ----------
    lines : (L,) azimuth line indices in the image
    pixels : (P,) range pixel indices in the image
    lut : (L, P) sigmaNought values at those positions

    Returns
    -------
    RegularGridInterpolator that accepts (azimuth_coords, range_coords)
    and returns interpolated LUT values.
    """
    # RegularGridInterpolator requires strictly increasing axes
    if not np.all(np.diff(lines) > 0):
        sort_idx = np.argsort(lines)
        lines = lines[sort_idx]
        lut = lut[sort_idx, :]

    if not np.all(np.diff(pixels) > 0):
        sort_idx = np.argsort(pixels)
        pixels = pixels[sort_idx]
        lut = lut[:, sort_idx]

    interp = RegularGridInterpolator(
        (lines, pixels),
        lut,
        method="linear",
        bounds_error=False,
        fill_value=None,   # extrapolate at edges using nearest
    )
    return interp


def _interpolate_lut_to_grid(
    interp: RegularGridInterpolator,
    image_shape: tuple[int, int],
    block_size: int = 512,
) -> np.ndarray:
    """
    Evaluate the calibration LUT interpolator over the full image grid,
    processing in blocks to avoid a single massive meshgrid allocation.

    Parameters
    ----------
    interp : RegularGridInterpolator built from the sparse LUT
    image_shape : (height, width) of the full image
    block_size : number of rows to process per block (trade memory for speed)

    Returns
    -------
    lut_grid : np.ndarray  shape (H, W), dtype float32
        Calibration LUT values at every pixel, ready for the formula.
    """
    H, W = image_shape
    lut_grid = np.empty((H, W), dtype=np.float32)
    col_coords = np.arange(W, dtype=np.float64)

    for row_start in range(0, H, block_size):
        row_end = min(row_start + block_size, H)
        row_coords = np.arange(row_start, row_end, dtype=np.float64)

        # Build (n_rows * n_cols, 2) query array
        rr, cc = np.meshgrid(row_coords, col_coords, indexing="ij")
        query = np.stack([rr.ravel(), cc.ravel()], axis=-1)

        vals = interp(query).reshape(row_end - row_start, W)
        lut_grid[row_start:row_end, :] = vals.astype(np.float32)

    return lut_grid


# ---------------------------------------------------------------------------
# Core calibration function
# ---------------------------------------------------------------------------

def calibrate_band(
    dn: np.ndarray,
    lut_grid: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Apply the radiometric calibration formula to a single band.

    Formula:  σ°(i,j) = DN(i,j)² / A(i,j)²

    Equivalently written as:  σ°(i,j) = (DN(i,j) / A(i,j))²

    Parameters
    ----------
    dn : np.ndarray  shape (H, W)
        Raw DN values (float32, already cast from uint16 in ingest.py).
    lut_grid : np.ndarray  shape (H, W)
        Calibration LUT values interpolated to every pixel.
    epsilon : float
        Minimum clamp value for lut_grid to prevent division by zero.

    Returns
    -------
    sigma0 : np.ndarray  shape (H, W), dtype float32
        Sigma-nought in linear power (dimensionless ratio).
        Values are typically in range [1e-5, 1.0] for ocean/ship scenes.
    """
    # Clamp LUT to avoid divide-by-zero in malformed edge pixels
    np.maximum(lut_grid, epsilon, out=lut_grid)

    # Core formula: σ° = DN² / A²
    # Compute in-place to save memory on 400M+ pixel arrays
    # 1. dn = dn / A
    np.divide(dn, lut_grid, out=dn)
    
    # 2. dn = dn^2
    np.square(dn, out=dn)

    # Note: Zero-DN pixels (masked/fill pixels) naturally stay zero 
    # since 0 / A = 0 and 0^2 = 0.

    return dn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate_scene(
    scene: SARScene,
    polarisations: Optional[list[str]] = None,
    verbose: bool = True,
) -> SARScene:
    """
    Radiometrically calibrate all available bands in a SARScene.

    Modifies scene.vv and/or scene.vh in-place, replacing raw DN arrays
    with sigma-nought (σ°) in linear power.  Also stores the interpolated
    LUT grids in scene.meta for optional inspection/debugging.

    Parameters
    ----------
    scene : SARScene
        Output of ingest_safe(). Must have calib_xml paths set.
    polarisations : list[str] | None
        Which polarisations to calibrate. Default: all available.
    verbose : bool

    Returns
    -------
    SARScene
        Same object, with .vv/.vh updated to σ° values.
    """
    if polarisations is None:
        polarisations = scene.available_polarisations()

    for pol in polarisations:
        xml_path = scene.calib_xml.get(pol)
        if xml_path is None:
            print(f"[calibrate] No calibration XML for {pol}, skipping.")
            continue

        arr = getattr(scene, pol.lower())
        if arr is None:
            print(f"[calibrate] No {pol} band data in scene, skipping.")
            continue

        if verbose:
            print(f"[calibrate] Parsing calibration XML for {pol}...")

        # 1. Parse the sparse LUT from XML
        lines, pixels, lut_sparse = _parse_calibration_xml(xml_path)

        if verbose:
            print(f"[calibrate]   LUT shape (sparse): {lut_sparse.shape}  "
                  f"lines={len(lines)}, pixels={len(pixels)}")
            print(f"[calibrate]   LUT value range: "
                  f"{lut_sparse.min():.1f} – {lut_sparse.max():.1f}")

        # 2. Build interpolator
        interp = _build_lut_interpolator(lines, pixels, lut_sparse)

        # 3. Interpolate to full image grid
        if verbose:
            print(f"[calibrate]   Interpolating LUT to {scene.shape} grid...")
        lut_grid = _interpolate_lut_to_grid(interp, scene.shape)

        # 4. Apply calibration formula
        if verbose:
            raw_min, raw_max = arr.min(), arr.max()
            print(f"[calibrate]   DN range before calibration: "
                  f"{raw_min:.1f} – {raw_max:.1f}")

        sigma0 = calibrate_band(arr, lut_grid)

        if verbose:
            # Report in dB for human-readable sanity check
            valid = sigma0[sigma0 > 0]
            if len(valid):
                sigma0_db = 10 * np.log10(valid)
                print(f"[calibrate]   σ° range (dB): "
                      f"{sigma0_db.min():.1f} – {sigma0_db.max():.1f} dB")
                print(f"[calibrate]   Ocean median: "
                      f"~{np.median(sigma0_db):.1f} dB  "
                      f"(expect −20 to −10 dB for open water)")

        # Store calibrated array back into scene
        setattr(scene, pol.lower(), sigma0)

        # Optionally stash the LUT grid for debugging
        scene.meta[f"lut_grid_{pol}"] = lut_grid

    if verbose:
        print("[calibrate] Done.")

    return scene


# ---------------------------------------------------------------------------
# Utility: save calibrated scene to GeoTIFF
# ---------------------------------------------------------------------------

def save_calibrated(scene: SARScene, output_path: str) -> None:
    """
    Save the calibrated sigma-nought bands to a 2-band GeoTIFF.

    Parameters
    ----------
    scene : SARScene  (after calibrate_scene has been called)
    output_path : str  e.g. "output/calibrated_sigma0.tif"
    """
    import os
    from utils.io import save_geotiff

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    bands = {}
    if scene.vv is not None:
        bands["VV"] = scene.vv
    if scene.vh is not None:
        bands["VH"] = scene.vh

    if not bands:
        raise ValueError("No calibrated bands available to save.")

    save_geotiff(output_path, bands, scene.transform, scene.crs)
    print(f"[calibrate] Saved σ° GeoTIFF → {output_path}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from ingest import ingest_safe

    if len(sys.argv) < 2:
        print("Usage: python calibrate.py /path/to/S1A_....SAFE [output.tif]")
        sys.exit(1)

    safe_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/calibrated.tif"

    print("=== Step 1: Ingest ===")
    scene = ingest_safe(safe_path, verbose=True)

    print("\n=== Step 2: Calibrate ===")
    scene = calibrate_scene(scene, verbose=True)

    print(f"\n=== Saving to {output_path} ===")
    save_calibrated(scene, output_path)

    print("\n=== Quick-look ===")
    from utils.viz import quicklook_scene
    quicklook_scene(scene, save_dir="output/quicklooks")
