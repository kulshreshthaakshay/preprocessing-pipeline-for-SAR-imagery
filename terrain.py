"""
sar_pipeline/terrain.py
------------------------
Step 4: Range-Doppler Terrain Correction (GTC).

Converts the SAR image from slant-range / azimuth geometry to a
geocoded geographic grid (WGS84 or UTM), correcting for:
  - foreshortening
  - layover
  - radar shadow
  - range spreading and antenna pattern (via DEM-assisted normalisation)

Two backend paths are provided. Choose based on your environment:

  PATH A — snappy (recommended for this pipeline)
      Requires: ESA SNAP + snappy configured
      Use when: you want full operator control, clean integration
                with existing SARScene dataclass, reproducibility

  PATH B — pyroSAR
      Requires: pip install pyrosar  + ESA SNAP installed
      Use when: batch processing many scenes, don't want snappy
                configuration overhead, OK with less parameter control

DEM options (both paths):
  "SRTM 1Sec HGT"          — 30m, global, auto-downloaded by SNAP
  "SRTM 3Sec"               — 90m, global, faster download
  "Copernicus 30m Global DEM" — 30m, more accurate, requires credentials
  "ACE2_5Min"               — 5 arcmin, very fast, lower accuracy

For ship detection over ocean, SRTM 1Sec HGT is the standard choice.
Ocean pixels have near-zero elevation so DEM accuracy matters far less
than over mountainous land; the main effect on ocean is correct geolocation.

Usage (Path A):
    from terrain import apply_terrain_correction_snappy
    output = apply_terrain_correction_snappy(
        safe_path="/data/S1A_....SAFE",
        output_path="output/scene_GTC.tif",
    )

Usage (Path B):
    from terrain import apply_terrain_correction_pyrosar
    output_dir = apply_terrain_correction_pyrosar(
        safe_path="/data/S1A_....SAFE",
        output_dir="output/",
    )
"""

from __future__ import annotations
import os
from pathlib import Path


# ============================================================================
# PATH A — snappy (direct SNAP operator chain)
# ============================================================================

def apply_terrain_correction_snappy(
    safe_path: str,
    output_path: str,
    dem_name: str = "SRTM 1Sec HGT",
    pixel_spacing_m: float = 10.0,
    crs: str = "GEOGCS[\"WGS84(DD)\"]",
    polarisations: str = "VV,VH",
    resampling: str = "BILINEAR_INTERPOLATION",
    orbit_type: str = "Sentinel Precise (Auto Download)",
    verbose: bool = True,
) -> str:
    """
    Run the full Sentinel-1 GRD preprocessing + terrain correction
    operator chain inside ESA SNAP via snappy.

    SNAP operator chain:
        Apply-Orbit-File
        → ThermalNoiseRemoval
        → Calibration (→ σ° linear)
        → Terrain-Correction (Range-Doppler, SRTM DEM)

    The output is a geocoded, calibrated GeoTIFF with σ° bands in
    linear power (not dB).  Pass it to normalize.py for dB conversion.

    Parameters
    ----------
    safe_path : str
        Path to the .SAFE directory.
    output_path : str
        Output GeoTIFF path.  Parent directory is created if needed.
    dem_name : str
        DEM product name as SNAP recognises it.
    pixel_spacing_m : float
        Output pixel spacing in metres.  10m = native IW GRD resolution.
    crs : str
        Output map projection as WKT.
        "GEOGCS[\"WGS84(DD)\"]"  = geographic lat/lon (default)
        "AUTO:42001"              = auto UTM zone (good for local analysis)
        "EPSG:32636"              = explicit UTM zone (e.g. zone 36N)
    polarisations : str
        Comma-separated list, e.g. "VV,VH" or "VV".
    resampling : str
        Interpolation method.  BILINEAR_INTERPOLATION is standard.
        Options: NEAREST_NEIGHBOUR, BICUBIC_INTERPOLATION.
    orbit_type : str
        Orbit file source.  Precise orbits (POE) are available ~3 weeks
        after acquisition.  For near-real-time use "Sentinel Restituted".
    verbose : bool

    Returns
    -------
    str : absolute path to the written GeoTIFF
    """
    try:
        import os
        if "JAVA_HOME" not in os.environ:
            # Fallback for SNAP 10 / jpy to find the JVM because Windows often lacks this env var
            os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-24"

        if "SNAP_HOME" not in os.environ:
            # Fallback for SNAP_HOME if installed in the standard Windows installer path for SNAP 10
            if os.path.exists(r"C:\Program Files\esa-snap"):
                os.environ["SNAP_HOME"] = r"C:\Program Files\esa-snap"

        # SNAP >= 10.0 renamed the python API to esa_snappy to avoid Google overlap
        try:
            import esa_snappy as snappy
            from esa_snappy import ProductIO, GPF, HashMap
        except ImportError:
            import snappy
            from snappy import ProductIO, GPF, HashMap
    except ImportError:
        raise ImportError(
            "snappy is not installed or not on PYTHONPATH.\n"
            "Follow setup instructions in the project README:\n"
            "  /path/to/snap/bin/snappy-conf /path/to/python\n"
            "  export PYTHONPATH=$PYTHONPATH:~/.snap/snap-python"
        )

    if verbose:
        print(f"[terrain/snappy] Reading: {safe_path}")

    product = ProductIO.readProduct(safe_path)

    # ── Operator 1: Apply precise orbit file ─────────────────────────────────
    # Downloads the restituted/precise orbit state vectors from ESA.
    # Improves geolocation accuracy from ~10 m to ~1–2 m RMS.
    # continueOnFail=true lets the pipeline proceed even if no orbit file
    # is available yet (e.g. scene acquired within the last 3 weeks).
    params = HashMap()
    params.put("orbitType", orbit_type)
    params.put("polyDegree", "3")
    params.put("continueOnFail", "true")
    if verbose:
        print("[terrain/snappy] Applying orbit file...")
    orbit_product = GPF.createProduct("Apply-Orbit-File", params, product)

    # ── Operator 2: Thermal noise removal ────────────────────────────────────
    # Subtracts the NESZ noise floor from each band using the noise LUT
    # stored in the SAFE annotation XMLs.  SNAP does this internally and
    # handles both range and azimuth noise vectors (IPF >= 3.40).
    params = HashMap()
    params.put("removeThermalNoise", "true")
    params.put("reIntroduceThermalNoise", "false")
    if verbose:
        print("[terrain/snappy] Removing thermal noise...")
    tnr_product = GPF.createProduct("ThermalNoiseRemoval", params, orbit_product)

    # ── Operator 3: Radiometric calibration ──────────────────────────────────
    # Converts DN to σ° in linear power using the calibration LUT from
    # annotation/calibration/*.xml.  Output bands are named
    # "Sigma0_VV" and "Sigma0_VH".
    params = HashMap()
    params.put("outputSigmaBand", "true")
    params.put("outputBetaBand", "false")
    params.put("outputGammaBand", "false")
    params.put("outputDNBand", "false")
    params.put("selectedPolarisations", polarisations)
    if verbose:
        print("[terrain/snappy] Calibrating to sigma-nought...")
    cal_product = GPF.createProduct("Calibration", params, tnr_product)

    # ── Operator 4: Range-Doppler Terrain Correction ──────────────────────────
    # The core GTC step.  For each output pixel (lat, lon):
    #   1. Convert to ECEF + add DEM elevation
    #   2. Solve Doppler equation for sensor position → azimuth time
    #   3. Compute slant range R → range sample index
    #   4. Bilinearly interpolate input image → write to output
    # nodataValueAtSea=false keeps ocean pixels (needed for ship detection).
    params = HashMap()
    params.put("demName", dem_name)
    params.put("demResamplingMethod", resampling)
    params.put("imgResamplingMethod", resampling)
    params.put("pixelSpacingInMeter", str(pixel_spacing_m))
    params.put("mapProjection", crs)
    params.put("nodataValueAtSea", "false")
    params.put("saveDEM", "false")
    params.put("saveLocalIncidenceAngle", "false")
    params.put("saveProjectedLocalIncidenceAngle", "false")
    params.put("saveSigmaNought", "false")
    params.put("saveSelectedSourceBand", "true")
    params.put("applyRadiometricNormalization", "false")
    if verbose:
        print(f"[terrain/snappy] Running Range-Doppler TC "
              f"(DEM={dem_name}, spacing={pixel_spacing_m}m)...")
    tc_product = GPF.createProduct("Terrain-Correction", params, cal_product)

    # ── Write output GeoTIFF ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if verbose:
        print(f"[terrain/snappy] Writing GeoTIFF → {output_path}")
    ProductIO.writeProduct(tc_product, output_path, "GeoTIFF-BigTIFF")

    # Explicit Java object cleanup (important for memory in long pipelines)
    for p in [tc_product, cal_product, tnr_product, orbit_product, product]:
        try:
            p.dispose()
        except Exception:
            pass

    if verbose:
        print("[terrain/snappy] Done.")

    return os.path.abspath(output_path)


# ============================================================================
# PATH B — pyroSAR
# ============================================================================

def apply_terrain_correction_pyrosar(
    safe_path: str,
    output_dir: str,
    spacing: int = 10,
    polarisations: list[str] | None = None,
    dem: str = "SRTM 1Sec HGT",
    epsg: int = 4326,
    scaling: str = "linear",
    clean_edges: bool = True,
    verbose: bool = True,
) -> str:
    """
    Run Sentinel-1 GRD terrain correction via pyroSAR's geocode().

    pyroSAR internally builds a SNAP graph equivalent to Path A, but
    exposes a simpler API and writes files with auto-generated names.

    Parameters
    ----------
    safe_path : str
        Path to the .SAFE directory.
    output_dir : str
        Directory where output GeoTIFFs will be written.
    spacing : int
        Output pixel spacing in metres.
    polarisations : list[str] | None
        e.g. ["VV", "VH"]. Default: all available.
    dem : str
        DEM name (same options as Path A).
    epsg : int
        EPSG code for output projection.
        4326 = WGS84 geographic, 32636 = UTM zone 36N, etc.
    scaling : str
        "linear" → σ° in linear power.
        "db"     → σ° in dB (pyroSAR applies 10*log10 internally).
        Use "linear" to keep consistent with our normalize.py step.
    clean_edges : bool
        Remove border noise (bright artefact strips at image edges).
    verbose : bool

    Returns
    -------
    str : output directory path
    """
    try:
        from pyroSAR import identify
        from pyroSAR.snap import geocode
    except ImportError:
        raise ImportError(
            "pyroSAR is not installed.\n"
            "Install with: pip install pyrosar\n"
            "Also requires ESA SNAP to be installed and findable on PATH."
        )

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"[terrain/pyrosar] Identifying scene: {safe_path}")
    scene = identify(safe_path)

    if verbose:
        print(f"[terrain/pyrosar] Scene ID   : {scene.scene}")
        print(f"[terrain/pyrosar] Polarisations: {scene.polarizations}")
        print(f"[terrain/pyrosar] Running geocode() ...")

    geocode(
        infile=scene,
        outdir=output_dir,
        spacing=spacing,
        t_srs=epsg,
        polarizations=polarisations or scene.polarizations,
        dem=dem,
        scaling=scaling,
        refarea="sigma0",
        clean_edges=clean_edges,
        clean_edges_npixels=3,
        groupsize=1,
        allow_RES_OSV=True,       # use restituted orbits if precise unavailable
        removeS1BorderNoiseMethod="ESA",
    )

    if verbose:
        # List what was actually written
        written = list(Path(output_dir).glob("*.tif"))
        print(f"[terrain/pyrosar] Written files:")
        for f in written:
            print(f"  {f.name}")

    return output_dir


# ============================================================================
# Shared utility: load GTC output back into numpy for downstream steps
# ============================================================================

def load_gtc_geotiff(
    geotiff_path: str,
    verbose: bool = True,
) -> tuple[dict, object, object]:
    """
    Load a terrain-corrected GeoTIFF (from either path) back into numpy,
    returning the same (arrays, transform, crs) tuple as utils/io.py.

    The band names written by SNAP are "Sigma0_VV" and "Sigma0_VH".
    This function normalises them to plain "VV" / "VH" for consistency
    with the rest of the pipeline.

    Parameters
    ----------
    geotiff_path : str

    Returns
    -------
    arrays : dict[str, np.ndarray]   e.g. {"VV": arr, "VH": arr}
    transform : rasterio Affine
    crs : rasterio CRS
    """
    try:
        import rasterio
        import numpy as np
    except ImportError:
        raise ImportError("rasterio and numpy are required.")

    arrays = {}
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs
        for i in range(1, src.count + 1):
            tag = src.tags(i).get("name", f"band_{i}")
            # Normalise SNAP band names → plain polarisation labels
            key = tag.replace("Sigma0_", "").replace("sigma0_", "").upper()
            if not key or key.startswith("BAND"):
                key = f"band_{i}"
            arr = src.read(i).astype(np.float32)
            arrays[key] = arr
            if verbose:
                valid = arr[arr > 0]
                if len(valid):
                    import numpy as np2
                    db = 10 * np2.log10(valid)
                    print(f"[terrain/load] {key}: σ° range "
                          f"{db.min():.1f} – {db.max():.1f} dB, "
                          f"shape {arr.shape}")

    return arrays, transform, crs


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python terrain.py /path/to/S1A.SAFE output/scene_GTC.tif [snappy|pyrosar]")
        sys.exit(1)

    safe_path = sys.argv[1]
    output_path = sys.argv[2]
    backend = sys.argv[3] if len(sys.argv) > 3 else "snappy"

    if backend == "snappy":
        result = apply_terrain_correction_snappy(safe_path, output_path)
        print(f"\nOutput GeoTIFF: {result}")
        arrays, tf, crs = load_gtc_geotiff(result)
        print(f"Loaded bands: {list(arrays.keys())}")
        print(f"CRS: {crs}")
        print(f"Transform: {tf}")
    elif backend == "pyrosar":
        import os
        output_dir = os.path.dirname(output_path) or "output"
        result = apply_terrain_correction_pyrosar(safe_path, output_dir)
        print(f"\nOutput directory: {result}")
    else:
        print(f"Unknown backend '{backend}'. Use 'snappy' or 'pyrosar'.")
        sys.exit(1)
