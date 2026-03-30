"""
sar_pipeline/ingest.py
----------------------
Step 1: Ingest a Sentinel-1 GRD SAFE package.

Responsibilities:
  - Locate VV and VH measurement .tiff files inside the SAFE directory
  - Read raw DN arrays via rasterio (no snappy dependency)
  - Parse the annotation XML to extract:
      * acquisition start/stop time
      * orbit direction (ASCENDING / DESCENDING)
      * pass direction and platform heading
      * path to the calibration annotation XML (used by calibrate.py)
      * path to the noise annotation XML (used by denoise.py)
  - Return a clean SARScene dataclass consumed by every downstream step

Dependencies:
    pip install rasterio numpy lxml

Usage:
    from ingest import ingest_safe
    scene = ingest_safe("/data/S1A_IW_GRDH_...SAFE")
    print(scene.vv.shape)          # (height, width)  float32 DN values
    print(scene.meta)              # dict of acquisition metadata
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
except ImportError:
    raise ImportError("rasterio is required. Run: pip install rasterio")

try:
    from lxml import etree
except ImportError:
    raise ImportError("lxml is required. Run: pip install lxml")


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SARScene:
    """
    Holds everything downstream steps need from a single Sentinel-1 GRD scene.

    Attributes
    ----------
    safe_path : str
        Absolute path to the .SAFE directory.
    vv : np.ndarray | None
        Raw DN array for VV polarisation, shape (H, W), dtype float32.
        None if VV is not present in this product (rare for IW GRD).
    vh : np.ndarray | None
        Raw DN array for VH polarisation, shape (H, W), dtype float32.
    transform : Affine
        Rasterio affine transform mapping pixel → geographic coordinates.
    crs : CRS
        Coordinate reference system of the measurement tiffs (usually WGS84).
    meta : dict
        Acquisition metadata parsed from the annotation XML.
        Keys: start_time, stop_time, orbit_direction, heading,
              mission, mode, product_type, range_spacing, azimuth_spacing.
    calib_xml : dict[str, str]
        Maps polarisation → absolute path to calibration annotation XML.
        e.g. {"VV": "/.../.SAFE/annotation/calibration/calibration-...vv.xml"}
    noise_xml : dict[str, str]
        Maps polarisation → absolute path to noise annotation XML.
    """
    safe_path: str
    vv: Optional[np.ndarray] = None
    vh: Optional[np.ndarray] = None
    transform: Optional[object] = None
    crs: Optional[object] = None
    meta: dict = field(default_factory=dict)
    calib_xml: dict = field(default_factory=dict)
    noise_xml: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of whichever band is available."""
        arr = self.vv if self.vv is not None else self.vh
        if arr is None:
            raise ValueError("No bands loaded.")
        return arr.shape

    def available_polarisations(self) -> list[str]:
        pols = []
        if self.vv is not None:
            pols.append("VV")
        if self.vh is not None:
            pols.append("VH")
        return pols


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_safe_root(path: str) -> Path:
    """Resolve the root of the SAFE package regardless of trailing slash."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"SAFE path does not exist: {p}")
    if p.suffix != ".SAFE" and not (p / "manifest.safe").exists():
        raise ValueError(
            f"Path does not look like a SAFE package (no manifest.safe): {p}"
        )
    return p


def _find_measurement_tiffs(safe_root: Path) -> dict[str, Path]:
    """
    Return a dict mapping polarisation → tiff path.
    Sentinel-1 GRD IW stores one tiff per polarisation under
    <SAFE>/measurement/s1a-iw-grd-{pol}-*.tiff
    """
    meas_dir = safe_root / "measurement"
    if not meas_dir.exists():
        raise FileNotFoundError(f"No 'measurement' directory in {safe_root}")

    tiffs = list(meas_dir.glob("*.tiff")) + list(meas_dir.glob("*.tif"))
    if not tiffs:
        raise FileNotFoundError(f"No .tiff files found in {meas_dir}")

    pol_map: dict[str, Path] = {}
    for t in tiffs:
        name = t.stem.lower()          # e.g. "s1a-iw-grd-vv-20230101t..."
        for pol in ("vv", "vh", "hh", "hv"):
            # Sentinel-1 convention: "-{pol}-" inside the filename
            if f"-{pol}-" in name:
                pol_map[pol.upper()] = t
                break

    if not pol_map:
        raise ValueError(f"Could not identify any polarised tiffs in {meas_dir}")

    return pol_map


def _find_annotation_xmls(safe_root: Path, pol_map: dict[str, Path]) -> tuple[dict, dict]:
    """
    Locate calibration and noise annotation XMLs for each polarisation.

    SAFE structure:
        annotation/calibration/calibration-s1a-iw-grd-{pol}-....xml
        annotation/calibration/noise-s1a-iw-grd-{pol}-....xml
    """
    calib_dir = safe_root / "annotation" / "calibration"
    calib_xml: dict[str, str] = {}
    noise_xml: dict[str, str] = {}

    for pol in pol_map:
        pol_lower = pol.lower()

        # calibration XML
        calib_matches = list(calib_dir.glob(f"calibration-*-{pol_lower}-*.xml"))
        if calib_matches:
            calib_xml[pol] = str(calib_matches[0])
        else:
            print(f"  [warn] No calibration XML found for {pol}")

        # noise XML
        noise_matches = list(calib_dir.glob(f"noise-*-{pol_lower}-*.xml"))
        if noise_matches:
            noise_xml[pol] = str(noise_matches[0])
        else:
            print(f"  [warn] No noise XML found for {pol}")

    return calib_xml, noise_xml


def _parse_annotation_meta(safe_root: Path, pol_map: dict[str, Path]) -> dict:
    """
    Parse the primary annotation XML (first available polarisation) for
    scene-level acquisition metadata.

    Returns a flat dict with string values.
    """
    ann_dir = safe_root / "annotation"
    xml_files = list(ann_dir.glob("*.xml"))
    if not xml_files:
        print("  [warn] No annotation XML found; metadata will be empty.")
        return {}

    # Prefer VV annotation if available
    chosen = None
    for xf in xml_files:
        if "-vv-" in xf.name.lower():
            chosen = xf
            break
    if chosen is None:
        chosen = xml_files[0]

    tree = etree.parse(str(chosen))
    root = tree.getroot()

    def _text(xpath: str, default: str = "unknown") -> str:
        els = root.xpath(xpath)
        return els[0].text.strip() if els else default

    meta = {
        "mission":          _text(".//missionId"),
        "mode":             _text(".//mode"),
        "product_type":     _text(".//productType"),
        "start_time":       _text(".//startTime"),
        "stop_time":        _text(".//stopTime"),
        "orbit_direction":  _text(".//pass"),          # ASCENDING / DESCENDING
        "heading":          _text(".//platformHeading"),
        "range_spacing":    _text(".//rangePixelSpacing"),
        "azimuth_spacing":  _text(".//azimuthPixelSpacing"),
        "incidence_angle":  _text(".//incidenceAngleMidSwath"),
        "polarisations":    ", ".join(sorted(pol_map.keys())),
    }
    return meta


def _read_band(tiff_path: Path) -> tuple[np.ndarray, object, object]:
    """
    Read a single-band measurement tiff.

    Returns
    -------
    arr : np.ndarray  shape (H, W), dtype float32
    transform : rasterio Affine
    crs : rasterio CRS
    """
    with rasterio.open(str(tiff_path)) as src:
        # GRD tiffs are uint16; read as float32 immediately
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return arr, transform, crs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_safe(safe_path: str, verbose: bool = True) -> SARScene:
    """
    Ingest a Sentinel-1 GRD SAFE package.

    Parameters
    ----------
    safe_path : str
        Path to the .SAFE directory, e.g.:
        "/data/S1A_IW_GRDH_1SDV_20230601T054912_20230601T054937_048832_05DE8F.SAFE"
    verbose : bool
        Print progress messages.

    Returns
    -------
    SARScene
        Populated dataclass ready for calibrate.py.
    """
    if verbose:
        print(f"[ingest] Loading SAFE: {safe_path}")

    safe_root = _find_safe_root(safe_path)

    # 1. Locate measurement tiffs
    pol_map = _find_measurement_tiffs(safe_root)
    if verbose:
        print(f"[ingest] Found polarisations: {list(pol_map.keys())}")

    # 2. Parse scene-level metadata
    meta = _parse_annotation_meta(safe_root, pol_map)
    if verbose:
        print(f"[ingest] Orbit direction : {meta.get('orbit_direction', '?')}")
        print(f"[ingest] Acquisition     : {meta.get('start_time', '?')}")

    # 3. Locate calibration / noise XMLs
    calib_xml, noise_xml = _find_annotation_xmls(safe_root, pol_map)

    # 4. Read band arrays
    vv_arr = vh_arr = None
    transform = crs = None

    for pol, tiff_path in pol_map.items():
        if verbose:
            print(f"[ingest] Reading {pol} → {tiff_path.name}")
        arr, tf, c = _read_band(tiff_path)
        if pol == "VV":
            vv_arr = arr
        elif pol == "VH":
            vh_arr = arr
        # Use the first band's geo-reference (both bands share the same grid)
        if transform is None:
            transform = tf
            crs = c

    scene = SARScene(
        safe_path=str(safe_root),
        vv=vv_arr,
        vh=vh_arr,
        transform=transform,
        crs=crs,
        meta=meta,
        calib_xml=calib_xml,
        noise_xml=noise_xml,
    )

    if verbose:
        print(f"[ingest] Done. Scene shape: {scene.shape}")
        print(f"[ingest] VV dtype/range: {vv_arr.dtype if vv_arr is not None else 'N/A'}")
        if vv_arr is not None:
            print(f"[ingest]   DN min={vv_arr.min():.1f}  max={vv_arr.max():.1f}")

    return scene


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python ingest.py /path/to/scene.SAFE)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py /path/to/S1A_....SAFE")
        sys.exit(1)

    scene = ingest_safe(sys.argv[1], verbose=True)

    print("\n--- Scene summary ---")
    print(f"  Polarisations : {scene.available_polarisations()}")
    print(f"  Shape         : {scene.shape}")
    print(f"  CRS           : {scene.crs}")
    print(f"  Transform     : {scene.transform}")
    print(f"  Metadata      :")
    for k, v in scene.meta.items():
        print(f"    {k:<22} {v}")
    print(f"  Calib XMLs    : {list(scene.calib_xml.keys())}")
    print(f"  Noise XMLs    : {list(scene.noise_xml.keys())}")
