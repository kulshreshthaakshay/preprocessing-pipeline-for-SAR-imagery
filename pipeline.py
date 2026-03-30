"""
sar_pipeline/pipeline.py
-------------------------
Orchestrator: runs the full preprocessing pipeline in order.

Step 1  ingest.py      — read SAFE, extract VV/VH DN arrays
Step 2  calibrate.py   — DN → σ° using calibration LUT
Step 3  denoise.py     — subtract thermal noise LUT
Step 4  terrain.py     — Range-Doppler terrain correction (snappy)
Step 5  speckle.py     — Lee / Gamma-MAP speckle filter   [TODO]
Step 6  normalize.py   — σ° → dB → clip → [0,1]          [TODO]

Usage:
    python pipeline.py /path/to/S1A_....SAFE --output-dir output/

Or from Python:
    from pipeline import run_pipeline
    result = run_pipeline("/data/S1A_....SAFE", output_dir="output/")
"""

from __future__ import annotations
import os
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the full preprocessing pipeline."""
    # I/O
    safe_path: str = ""
    output_dir: str = "output"

    # Terrain correction
    dem_name: str = "SRTM 1Sec HGT"
    pixel_spacing_m: float = 10.0
    output_crs: str = "AUTO:42001"
    tc_backend: str = "snappy"          # "snappy" or "pyrosar"

    # Speckle filter (used once speckle.py is implemented)
    speckle_filter: str = "lee"         # "lee", "gamma_map", "none"
    speckle_window: int = 7

    # Normalisation
    db_clip_min: float = -30.0
    db_clip_max: float = 0.0

    # Misc
    polarisations: list = field(default_factory=lambda: ["VV", "VH"])
    verbose: bool = True
    save_intermediates: bool = False    # save after each step for debugging


def run_pipeline(
    safe_path: str,
    output_dir: str = "output",
    config: PipelineConfig | None = None,
) -> dict[str, str]:
    """
    Run the full SAR preprocessing pipeline on a single Sentinel-1 GRD scene.

    Parameters
    ----------
    safe_path : str
        Path to the .SAFE directory.
    output_dir : str
        Root output directory.  Intermediate and final files go here.
    config : PipelineConfig | None
        Pipeline configuration.  Defaults used if None.

    Returns
    -------
    dict[str, str]
        Paths to all output files keyed by step name.
        e.g. {"gtc": "output/scene_GTC.tif", "normalized": "output/scene_norm.tif"}
    """
    if config is None:
        config = PipelineConfig(safe_path=safe_path, output_dir=output_dir)
    config.safe_path = safe_path
    config.output_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # Derive a short scene ID for output file names
    scene_id = Path(safe_path).stem[:32]
    outputs = {}

    def _elapsed():
        return f"{time.time() - t0:.1f}s"

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 1 / Ingest  [{_elapsed()}]")
    print(f"{'='*60}")
    from ingest import ingest_safe
    scene = ingest_safe(safe_path, verbose=config.verbose)

    if config.save_intermediates:
        from utils.io import save_geotiff
        p = os.path.join(output_dir, f"{scene_id}_01_raw_DN.tif")
        bands = {}
        if scene.vv is not None: bands["VV"] = scene.vv
        if scene.vh is not None: bands["VH"] = scene.vh
        save_geotiff(p, bands, scene.transform, scene.crs)
        outputs["raw"] = p

    # ── Step 2: Calibrate ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 2 / Calibrate  [{_elapsed()}]")
    print(f"{'='*60}")
    from calibrate import calibrate_scene
    scene = calibrate_scene(scene, verbose=config.verbose)

    if config.save_intermediates:
        p = os.path.join(output_dir, f"{scene_id}_02_sigma0_linear.tif")
        from calibrate import save_calibrated
        save_calibrated(scene, p)
        outputs["calibrated"] = p

    # ── Step 3: Thermal noise removal ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 3 / Thermal noise removal  [{_elapsed()}]")
    print(f"{'='*60}")
    from denoise import remove_thermal_noise
    scene = remove_thermal_noise(scene, verbose=config.verbose)
    if config.save_intermediates:
        p = os.path.join(output_dir, f"{scene_id}_03_denoised.tif")
        from utils.io import save_geotiff
        bands = {}
        if scene.vv is not None: bands["VV"] = scene.vv
        if scene.vh is not None: bands["VH"] = scene.vh
        save_geotiff(p, bands, scene.transform, scene.crs)
        outputs["denoised"] = p

    # ── Step 4: Terrain correction ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 4 / Terrain correction  [{_elapsed()}]")
    print(f"{'='*60}")
    gtc_path = os.path.join(output_dir, f"{scene_id}_04_GTC.tif")

    if config.tc_backend == "snappy":
        from terrain import apply_terrain_correction_snappy
        # Note: SNAP's operator chain re-runs calibration + TNR internally.
        # Pass the original SAFE path here, not the already-calibrated arrays,
        # because SNAP operators need the raw product metadata.
        gtc_path = apply_terrain_correction_snappy(
            safe_path=safe_path,
            output_path=gtc_path,
            dem_name=config.dem_name,
            pixel_spacing_m=config.pixel_spacing_m,
            crs=config.output_crs,
            polarisations=",".join(config.polarisations),
            verbose=config.verbose,
        )
    elif config.tc_backend == "pyrosar":
        from terrain import apply_terrain_correction_pyrosar
        gtc_dir = apply_terrain_correction_pyrosar(
            safe_path=safe_path,
            output_dir=output_dir,
            spacing=int(config.pixel_spacing_m),
            polarisations=config.polarisations,
            dem=config.dem_name,
            verbose=config.verbose,
        )
        # Locate the written tif
        tifs = list(Path(gtc_dir).glob("*.tif"))
        gtc_path = str(tifs[0]) if tifs else gtc_path
    else:
        raise ValueError(f"Unknown tc_backend: {config.tc_backend!r}")

    outputs["gtc"] = gtc_path
    print(f"[pipeline] GTC complete [{_elapsed()}]")

    # ── Step 5: Speckle filtering ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 5 / Speckle filtering  [{_elapsed()}]")
    print(f"{'='*60}")
    try:
        from speckle import apply_speckle_filter
        from terrain import load_gtc_geotiff
        from utils.io import save_geotiff

        arrays, transform, crs = load_gtc_geotiff(gtc_path, verbose=config.verbose)
        filtered = {}
        for pol, arr in arrays.items():
            filtered[pol] = apply_speckle_filter(
                arr,
                method=config.speckle_filter,
                window_size=config.speckle_window,
            )

        speckle_path = os.path.join(output_dir, f"{scene_id}_05_speckle.tif")
        save_geotiff(speckle_path, filtered, transform, crs)
        outputs["speckle"] = speckle_path
    except ImportError:
        print("[pipeline] speckle.py not yet implemented — skipping speckle filtering.")
        speckle_path = gtc_path   # pass GTC output to normalise directly

    # ── Step 6: Normalize ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 6 / Normalize  [{_elapsed()}]")
    print(f"{'='*60}")
    try:
        from normalize import normalize_to_db_and_scale
        from terrain import load_gtc_geotiff
        from utils.io import save_geotiff

        src_path = outputs.get("speckle", outputs.get("gtc"))
        arrays, transform, crs = load_gtc_geotiff(src_path, verbose=False)
        normalized = {}
        for pol, arr in arrays.items():
            normalized[pol] = normalize_to_db_and_scale(
                arr,
                db_min=config.db_clip_min,
                db_max=config.db_clip_max,
            )

        norm_path = os.path.join(output_dir, f"{scene_id}_06_normalized.tif")
        save_geotiff(norm_path, normalized, transform, crs)
        outputs["normalized"] = norm_path
        print(f"[pipeline] Normalized output → {norm_path}")
    except ImportError:
        print("[pipeline] normalize.py not yet implemented — skipping normalisation.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {_elapsed()}")
    print(f"{'='*60}")
    for step, path in outputs.items():
        size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
        print(f"  {step:<14} {path}  ({size_mb:.1f} MB)")

    return outputs


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR preprocessing pipeline")
    parser.add_argument("safe_path", help="Path to Sentinel-1 .SAFE directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--dem", default="SRTM 1Sec HGT", help="DEM product name")
    parser.add_argument("--spacing", type=float, default=10.0, help="Output pixel spacing (m)")
    parser.add_argument("--backend", choices=["snappy", "pyrosar"], default="snappy")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Save GeoTIFF after each step")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    cfg = PipelineConfig(
        dem_name=args.dem,
        pixel_spacing_m=args.spacing,
        tc_backend=args.backend,
        save_intermediates=args.save_intermediates,
        verbose=args.verbose,
    )

    run_pipeline(args.safe_path, args.output_dir, config=cfg)
