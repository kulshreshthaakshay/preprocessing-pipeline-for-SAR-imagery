"""
sar_pipeline/denoise.py
------------------------
Step 3: Thermal noise removal for Sentinel-1 GRD.

Subtracts the Noise Equivalent Sigma Zero (NESZ) from calibrated sigma-nought
using the noise LUT stored in the SAFE annotation XMLs.

Formula:
    σ°_denoised(i,j) = max(σ°_calibrated(i,j) − N(i,j), ε)

where N(i,j) is the bilinearly interpolated noise power at pixel (i,j).

MUST run AFTER calibrate.py — the noise LUT is in linear power units
that match the calibrated σ°, not raw DN.

Two noise vector types exist depending on IPF processor version:
    IPF < 2.90   : noiseRangeVector only (range direction)
    IPF ≥ 2.90   : noiseRangeVector + noiseAzimuthVector (both directions)
                   Combined as: N(i,j) = N_range(i,j) × N_azimuth(i,j)

This module auto-detects which format the SAFE file uses and applies
the correct combination.

Dependencies:
    pip install numpy scipy lxml

Usage:
    from ingest import ingest_safe
    from calibrate import calibrate_scene
    from denoise import remove_thermal_noise

    scene = ingest_safe("/data/S1A_....SAFE")
    scene = calibrate_scene(scene)
    scene = remove_thermal_noise(scene)
    # scene.vv / scene.vh now contain denoised σ° in linear power
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


# ─────────────────────────────────────────────────────────────────────────────
# XML parsing — range noise vectors
# ─────────────────────────────────────────────────────────────────────────────

def _parse_noise_range_vectors(xml_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse <noiseRangeVector> elements from a Sentinel-1 noise annotation XML.

    Each vector covers one azimuth line and contains sparse range sample
    positions plus the corresponding NESZ values in linear power.

    Structure (IPF ≥ 2.90 schema name change: noiseVector → noiseRangeVector):

        <noiseRangeVectorList>
          <noiseRangeVector>
            <azimuthTime>2023-06-01T05:49:12.1</azimuthTime>
            <line>0</line>
            <pixel count="21">0 1270 2540 ... 25192</pixel>
            <noiseRangeLut count="21">4.2e-4 3.9e-4 ...</noiseRangeLut>
          </noiseRangeVector>
          ...
        </noiseRangeVectorList>

    Older IPF (< 2.90) uses <noiseVectorList> / <noiseVector> / <noiseLut>.

    Returns
    -------
    lines  : np.ndarray (L,)   azimuth line indices
    pixels : np.ndarray (P,)   range pixel indices (from first vector)
    lut    : np.ndarray (L, P) NESZ values in linear power
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Detect IPF schema version — try new name first, fall back to old
    vectors = root.findall(".//noiseRangeVector")
    lut_tag = "noiseRangeLut"
    if not vectors:
        vectors = root.findall(".//noiseVector")   # IPF < 2.90
        lut_tag = "noiseLut"

    if not vectors:
        raise ValueError(
            f"No noise range vectors found in {xml_path}.\n"
            "Expected <noiseRangeVector> or <noiseVector> elements."
        )

    lines_list, lut_rows = [], []
    pixels_ref = None

    for vec in vectors:
        line_el  = vec.find("line")
        pixel_el = vec.find("pixel")
        lut_el   = vec.find(lut_tag)

        if any(el is None or el.text is None for el in [line_el, pixel_el, lut_el]):
            continue

        line_idx = int(line_el.text.strip())
        pixels   = np.array(pixel_el.text.strip().split(),  dtype=np.float64)
        lut_vals = np.array(lut_el.text.strip().split(),    dtype=np.float64)

        if len(pixels) != len(lut_vals):
            continue   # malformed vector — skip

        if pixels_ref is None:
            pixels_ref = pixels

        lines_list.append(line_idx)
        lut_rows.append(lut_vals)

    if not lines_list:
        raise ValueError(f"All noise vectors were malformed or empty in {xml_path}")

    lines = np.array(lines_list, dtype=np.float64)
    lut   = np.array(lut_rows,   dtype=np.float64)   # (L, P)

    return lines, pixels_ref, lut


# ─────────────────────────────────────────────────────────────────────────────
# XML parsing — azimuth noise vectors (IPF ≥ 2.90 only)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_noise_azimuth_vectors(
    xml_path: str,
    image_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Parse <noiseAzimuthVector> elements and build a full-image azimuth
    noise correction array (H, W).

    Azimuth noise vectors cover rectangular blocks within the image.
    Each block specifies:
        firstRangeSample, lastRangeSample   — column extent
        firstAzimuthLine, lastAzimuthLine   — row extent
        line                                — azimuth sample positions
        noiseAzimuthLut                     — NESZ values at those lines

    Returns None if no azimuth vectors are present (older IPF).

    Parameters
    ----------
    xml_path : str
    image_shape : (H, W) of the full image

    Returns
    -------
    az_grid : np.ndarray (H, W) float64, or None
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    vectors = root.findall(".//noiseAzimuthVector")
    if not vectors:
        return None   # older IPF — no azimuth correction needed

    H, W = image_shape
    az_grid = np.ones((H, W), dtype=np.float32)   # default = 1.0 (no correction)

    for vec in vectors:
        def _int(tag):
            el = vec.find(tag)
            return int(el.text.strip()) if el is not None and el.text else None

        first_range = _int("firstRangeSample")
        last_range  = _int("lastRangeSample")
        first_line  = _int("firstAzimuthLine")
        last_line   = _int("lastAzimuthLine")
        line_el     = vec.find("line")
        lut_el      = vec.find("noiseAzimuthLut")

        if any(v is None for v in [first_range, last_range, first_line, last_line]):
            continue
        if line_el is None or lut_el is None:
            continue

        az_lines = np.array(line_el.text.strip().split(), dtype=np.float64)
        az_lut   = np.array(lut_el.text.strip().split(),  dtype=np.float64)

        if len(az_lines) < 2 or len(az_lines) != len(az_lut):
            continue

        # Clamp to image bounds
        r0 = max(first_line,  0)
        r1 = min(last_line,   H - 1)
        c0 = max(first_range, 0)
        c1 = min(last_range,  W - 1)

        if r1 < r0 or c1 < c0:
            continue

        # 1-D interpolator along azimuth for this block
        interp = RegularGridInterpolator(
            (az_lines,),
            az_lut[:, np.newaxis],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        row_coords = np.arange(r0, r1 + 1, dtype=np.float64)
        az_values  = interp(row_coords[:, np.newaxis]).ravel()   # (n_rows,)

        # Broadcast azimuth correction across the column extent of this block
        az_grid[r0:r1 + 1, c0:c1 + 1] = az_values[:, np.newaxis]

    return az_grid


# ─────────────────────────────────────────────────────────────────────────────
# LUT interpolation to full image grid
# ─────────────────────────────────────────────────────────────────────────────

def _build_range_interpolator(
    lines: np.ndarray,
    pixels: np.ndarray,
    lut: np.ndarray,
) -> RegularGridInterpolator:
    """Build a bilinear interpolator for the range noise LUT."""
    # Ensure strictly increasing axes
    if not np.all(np.diff(lines) > 0):
        idx = np.argsort(lines)
        lines, lut = lines[idx], lut[idx, :]
    if not np.all(np.diff(pixels) > 0):
        idx = np.argsort(pixels)
        pixels, lut = pixels[idx], lut[:, idx]

    return RegularGridInterpolator(
        (lines, pixels),
        lut,
        method="linear",
        bounds_error=False,
        fill_value=None,    # extrapolate at image edges
    )


def _interpolate_to_grid(
    interp: RegularGridInterpolator,
    image_shape: tuple[int, int],
    block_size: int = 512,
) -> np.ndarray:
    """
    Evaluate the range noise LUT over the full image grid in row blocks.

    Block-wise evaluation prevents allocating a single (H×W, 2) query
    array that would require ~6 GB for a full Sentinel-1 IW scene.

    Returns
    -------
    grid : np.ndarray (H, W) float32
    """
    H, W = image_shape
    grid = np.empty((H, W), dtype=np.float32)
    col_coords = np.arange(W, dtype=np.float64)

    for r0 in range(0, H, block_size):
        r1 = min(r0 + block_size, H)
        row_coords = np.arange(r0, r1, dtype=np.float64)
        rr, cc = np.meshgrid(row_coords, col_coords, indexing="ij")
        query  = np.stack([rr.ravel(), cc.ravel()], axis=-1)
        vals   = interp(query).reshape(r1 - r0, W)
        grid[r0:r1, :] = vals.astype(np.float32)

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Core noise subtraction
# ─────────────────────────────────────────────────────────────────────────────

def _build_noise_grid(
    xml_path: str,
    image_shape: tuple[int, int],
    verbose: bool = True,
) -> np.ndarray:
    """
    Build the full (H, W) noise power grid N(i,j) from one noise XML.

    For IPF ≥ 2.90:
        N(i,j) = N_range(i,j) × N_azimuth(i,j)
    For IPF < 2.90:
        N(i,j) = N_range(i,j)

    Both components are interpolated from their sparse vector
    representations to the full pixel grid.

    Returns
    -------
    noise_grid : np.ndarray (H, W) float32
        NESZ noise power in the same linear units as calibrated σ°.
    """
    # ── Range noise ────────────────────────────────────────────────────────
    lines, pixels, lut_range = _parse_noise_range_vectors(xml_path)

    if verbose:
        print(f"  [denoise] Range LUT: {len(lines)} azimuth vectors × "
              f"{len(pixels)} range samples, "
              f"values [{lut_range.min():.3e}, {lut_range.max():.3e}]")

    interp  = _build_range_interpolator(lines, pixels, lut_range)
    rng_grid = _interpolate_to_grid(interp, image_shape)

    # ── Azimuth noise (IPF ≥ 2.90) ─────────────────────────────────────────
    az_grid = _parse_noise_azimuth_vectors(xml_path, image_shape)

    if az_grid is not None:
        if verbose:
            print(f"  [denoise] Azimuth LUT found (IPF ≥ 2.90) — applying combined correction")
        # Compute noise grid in-place to avoid massive 3-4 GB memory spikes on full scenes
        rng_grid = rng_grid.astype(np.float32, copy=False)
        az_grid = az_grid.astype(np.float32, copy=False)
        np.multiply(rng_grid, az_grid, out=rng_grid)
        noise_grid = rng_grid
    else:
        if verbose:
            print(f"  [denoise] No azimuth LUT (IPF < 2.90) — range correction only")
        noise_grid = rng_grid.astype(np.float32, copy=False)

    return noise_grid


def _subtract_noise(
    sigma0: np.ndarray,
    noise_grid: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Subtract noise power from calibrated sigma-nought.

    σ°_denoised = max(σ°_calibrated − N, ε)

    Subtraction must be in LINEAR power domain.
    Negative values after subtraction represent pixels where the signal
    was at or below the noise floor — clamped to ε so log10 doesn't fail.

    Parameters
    ----------
    sigma0 : np.ndarray (H, W) float32  — calibrated σ° in linear power
    noise_grid : np.ndarray (H, W)      — NESZ from _build_noise_grid()
    epsilon : float                     — clamp floor (~−100 dB, safe)

    Returns
    -------
    denoised : np.ndarray (H, W) float32
    """
    # Identify nodata pixels from upstream before we mutate sigma0
    # Boolean mask takes 1 byte/pixel (~400MB) instead of float64 copies (3GB/each)
    nodata_mask = (sigma0 <= 0)

    # In-place subtraction: sigma0 = sigma0 - noise_grid
    # We do this in float32 directly on the pre-allocated sigma0 array
    # to avoid 10GB+ memory allocation crashes on massive full-swath scenes
    np.subtract(sigma0, noise_grid, out=sigma0)
    
    # In-place clamp to epsilon (noise floor)
    np.maximum(sigma0, epsilon, out=sigma0)

    # Restore nodata pixels (masking)
    sigma0[nodata_mask] = 0.0

    return sigma0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def remove_thermal_noise(
    scene: SARScene,
    polarisations: Optional[list[str]] = None,
    epsilon: float = 1e-10,
    verbose: bool = True,
) -> SARScene:
    """
    Remove thermal noise from all available polarisation bands.

    Must be called AFTER calibrate_scene() — operates on σ° arrays,
    not raw DN.  Modifies scene.vv and/or scene.vh in-place.

    Parameters
    ----------
    scene : SARScene
        Output of calibrate_scene().  scene.noise_xml must be populated
        (it is set by ingest_safe() automatically).
    polarisations : list[str] | None
        Which polarisations to process. Default: all in scene.
    epsilon : float
        Minimum output value after subtraction. Protects log10(0) in
        the normalisation step. Default 1e-10 ≈ −100 dB, well below
        the [−30, 0] dB clip window in normalize.py.
    verbose : bool

    Returns
    -------
    SARScene  (same object, vv/vh updated to denoised σ°)
    """
    if polarisations is None:
        polarisations = scene.available_polarisations()

    for pol in polarisations:
        xml_path = scene.noise_xml.get(pol)
        if xml_path is None:
            if verbose:
                print(f"[denoise] No noise XML for {pol} — skipping.")
            continue

        arr = getattr(scene, pol.lower())
        if arr is None:
            if verbose:
                print(f"[denoise] No {pol} array in scene — skipping.")
            continue

        if verbose:
            print(f"[denoise] Processing {pol}  (source: {Path(xml_path).name})")

        # ── Build noise grid ────────────────────────────────────────────────
        noise_grid = _build_noise_grid(xml_path, arr.shape, verbose=verbose)

        if verbose:
            noise_db = 10.0 * np.log10(np.maximum(noise_grid, 1e-10))
            print(f"  [denoise] Noise floor: "
                  f"min={noise_db.min():.1f} dB  "
                  f"max={noise_db.max():.1f} dB  "
                  f"mean={noise_db.mean():.1f} dB")

        # ── Subtract ────────────────────────────────────────────────────────
        before_db_median = float(np.median(
            10.0 * np.log10(np.maximum(arr[arr > 0], 1e-10))
        ))

        denoised = _subtract_noise(arr, noise_grid, epsilon=epsilon)

        after_db_median = float(np.median(
            10.0 * np.log10(np.maximum(denoised[denoised > 0], 1e-10))
        ))

        if verbose:
            change = after_db_median - before_db_median
            print(f"  [denoise] Median σ° before: {before_db_median:.2f} dB  "
                  f"after: {after_db_median:.2f} dB  "
                  f"(Δ {change:+.2f} dB)")

            # Fraction clamped (were at or below noise floor)
            # Since array was already clamped in-place by _subtract_noise,
            # we simply count pixels <= epsilon to avoid massive allocations
            n_clamped = int(np.sum(denoised <= epsilon))
            frac = n_clamped / denoised.size
            print(f"  [denoise] Pixels at noise floor: "
                  f"{n_clamped:,} / {denoised.size:,}  ({frac*100:.2f}%)")

        # Write back
        setattr(scene, pol.lower(), denoised)

        # Store noise grid in meta for optional inspection
        scene.meta[f"noise_grid_{pol}"] = noise_grid

    if verbose:
        print("[denoise] Done.")

    return scene


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: save denoised scene
# ─────────────────────────────────────────────────────────────────────────────

def save_denoised(scene: SARScene, output_path: str) -> None:
    """Save the denoised σ° bands to a 2-band float32 GeoTIFF."""
    import os
    from utils.io import save_geotiff

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    bands = {}
    if scene.vv is not None:
        bands["VV"] = scene.vv
    if scene.vh is not None:
        bands["VH"] = scene.vh
    if not bands:
        raise ValueError("No denoised bands in scene.")
    save_geotiff(output_path, bands, scene.transform, scene.crs)
    print(f"[denoise] Saved denoised σ° → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ingest import ingest_safe
    from calibrate import calibrate_scene

    if len(sys.argv) < 2:
        print("Usage: python denoise.py /path/to/S1A_....SAFE [output.tif]")
        sys.exit(1)

    safe_path   = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/denoised.tif"

    print("=== Step 1: Ingest ===")
    scene = ingest_safe(safe_path, verbose=True)

    print("\n=== Step 2: Calibrate ===")
    scene = calibrate_scene(scene, verbose=True)

    print("\n=== Step 3: Denoise ===")
    scene = remove_thermal_noise(scene, verbose=True)

    print(f"\n=== Saving to {output_path} ===")
    save_denoised(scene, output_path)
    print("Done.")
