"""
sar_pipeline/normalize.py
--------------------------
Step 6: Convert sigma-nought to dB, clip to physical range, and normalise
to [0, 1] for deep learning model input.

Full chain per pixel:
    σ°_linear  (float32, calibrated + denoised + speckle-filtered)
        ↓  10 · log₁₀(σ°)
    σ°_dB      (float32, typically −35 to +5 dB for ocean/ship scenes)
        ↓  clip(σ°_dB, db_min, db_max)
    σ°_clipped (float32, range [db_min, db_max])
        ↓  (σ°_clipped − db_min) / (db_max − db_min)
    norm       (float32, range [0, 1])

Why this transform?
  - σ° is log-normally distributed in linear → log converts to ~Gaussian
  - [−30, 0] dB clip removes thermal noise floor and saturated metal targets
  - Linear rescaling to [0,1] gives well-conditioned gradients without
    fitting any per-dataset statistics (same formula works for all scenes)

Optional second stage:
  - ImageNet-style per-channel mean/std subtraction for pretrained backbones
  - SAR dataset statistics for domain-adapted normalisation

Dependencies:
    pip install numpy torch rasterio

Usage:
    from normalize import normalize_band, normalize_scene_to_tensor
    from ingest import ingest_safe
    from calibrate import calibrate_scene

    scene = calibrate_scene(ingest_safe("S1A_....SAFE"))
    tensor = normalize_scene_to_tensor(scene)   # shape (2, H, W)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# Torch is optional — only needed for tensor output
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

# Default dB clip range, physically motivated:
#   −30 dB ≈ thermal noise floor of Sentinel-1 IW GRD (NESZ)
#     0 dB = backscatter equal to incident power — only corner reflectors
#            and ship superstructures reach or exceed this
DB_MIN_DEFAULT = -30.0
DB_MAX_DEFAULT =   0.0

# Epsilon to protect log10(0) — corresponds to ~−100 dB, well below the clip
LOG_EPS = 1e-10

# ImageNet channel statistics (for pretrained backbone compatibility)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# SAR-specific statistics estimated from large Sentinel-1 IW GRD datasets
# (approximate — compute from your own training set for best results)
SAR_VV_MEAN_NORM = 0.42    # mean of normalised [0,1] VV band over ocean scenes
SAR_VH_MEAN_NORM = 0.35    # VH is ~3 dB lower than VV on average
SAR_VV_STD_NORM  = 0.12
SAR_VH_STD_NORM  = 0.11


# ─────────────────────────────────────────────────────────────────────────────
# Core normalisation
# ─────────────────────────────────────────────────────────────────────────────

def linear_to_db(arr: np.ndarray, eps: float = LOG_EPS) -> np.ndarray:
    """
    Convert linear-power sigma-nought to decibels.

        σ°_dB = 10 · log₁₀(max(σ°, eps))

    Parameters
    ----------
    arr : np.ndarray  shape (H, W) or (C, H, W), float32
        Linear power σ° — output of calibrate.py / speckle.py.
        Must be positive (zero and negative pixels are handled via eps).
    eps : float
        Floor for log protection.  Default 1e-10 ≈ −100 dB, well below clip.

    Returns
    -------
    db : np.ndarray  same shape, float32
        Values in dB.  Typical range: −35 to +5 dB for ocean/ship scenes.
    """
    safe = np.maximum(arr.astype(np.float64), eps)
    db = 10.0 * np.log10(safe)
    return db.astype(np.float32)


def clip_db(
    db: np.ndarray,
    db_min: float = DB_MIN_DEFAULT,
    db_max: float = DB_MAX_DEFAULT,
) -> np.ndarray:
    """
    Clip dB values to the physically meaningful range.

    Values below db_min are noise floor — uninformative and
    highly variable between scenes.  Values above db_max are
    saturated corner reflectors — also uninformative and rare.

    Parameters
    ----------
    db : np.ndarray  float32, dB values
    db_min : float   lower clip, default −30 dB (noise floor)
    db_max : float   upper clip, default   0 dB (saturation)

    Returns
    -------
    clipped : np.ndarray  same shape, float32
    """
    return np.clip(db, db_min, db_max).astype(np.float32)


def scale_to_unit(
    db_clipped: np.ndarray,
    db_min: float = DB_MIN_DEFAULT,
    db_max: float = DB_MAX_DEFAULT,
) -> np.ndarray:
    """
    Linearly rescale clipped dB values to [0, 1].

        norm = (db_clipped − db_min) / (db_max − db_min)

    This is a fixed, scene-independent formula:
      db_min → 0.0  (noise floor)
      db_max → 1.0  (saturation / strong metal)
      ocean  → ~0.35–0.45
      ships  → ~0.75–0.95

    Parameters
    ----------
    db_clipped : np.ndarray  float32, already clipped to [db_min, db_max]
    db_min, db_max : float   must match the values used in clip_db()

    Returns
    -------
    norm : np.ndarray  same shape, float32, range [0.0, 1.0]
    """
    span = db_max - db_min
    norm = (db_clipped.astype(np.float64) - db_min) / span
    return norm.astype(np.float32)


def normalize_band(
    arr: np.ndarray,
    db_min: float = DB_MIN_DEFAULT,
    db_max: float = DB_MAX_DEFAULT,
    eps: float = LOG_EPS,
) -> np.ndarray:
    """
    Full single-band normalisation pipeline:
        linear σ°  →  dB  →  clip  →  [0, 1]

    Parameters
    ----------
    arr : np.ndarray  shape (H, W), float32
        Linear sigma-nought from speckle.py output.
    db_min, db_max : float
        dB clip range.
    eps : float
        Log floor protection.

    Returns
    -------
    norm : np.ndarray  shape (H, W), float32, values in [0.0, 1.0]
    """
    db = linear_to_db(arr, eps=eps)
    clipped = clip_db(db, db_min, db_max)
    return scale_to_unit(clipped, db_min, db_max)


def normalize_to_db_and_scale(
    arr: np.ndarray,
    db_min: float = DB_MIN_DEFAULT,
    db_max: float = DB_MAX_DEFAULT,
) -> np.ndarray:
    """Alias of normalize_band — used by pipeline.py."""
    return normalize_band(arr, db_min=db_min, db_max=db_max)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-band and tensor output
# ─────────────────────────────────────────────────────────────────────────────

def normalize_scene_to_array(
    vv: Optional[np.ndarray],
    vh: Optional[np.ndarray],
    db_min: float = DB_MIN_DEFAULT,
    db_max: float = DB_MAX_DEFAULT,
    add_ratio_band: bool = True,
) -> np.ndarray:
    """
    Normalise VV and VH bands and stack into a multi-channel array.

    Optionally adds a third channel: the VV/VH ratio in dB, which
    highlights ships (strong in both channels) versus sea clutter
    (VH proportionally lower than VV).

    Parameters
    ----------
    vv, vh : np.ndarray | None
        Linear sigma-nought arrays from speckle.py.
    db_min, db_max : float
    add_ratio_band : bool
        If True and both VV and VH are present, add a third channel:
            ratio_dB = σ°_VV_dB − σ°_VH_dB   (normalised separately)
        Ships tend to have ratio_dB near 0–3 dB (similar polarisation).
        Ocean tends to have ratio_dB near 5–10 dB (VV >> VH).

    Returns
    -------
    stacked : np.ndarray  shape (C, H, W), float32
        C = 1 (VV only), 2 (VV+VH), or 3 (VV+VH+ratio)
    band_names : list[str]
        Names corresponding to each channel.
    """
    bands = []
    band_names = []

    if vv is not None:
        bands.append(normalize_band(vv, db_min, db_max))
        band_names.append("VV_norm")

    if vh is not None:
        bands.append(normalize_band(vh, db_min, db_max))
        band_names.append("VH_norm")

    if add_ratio_band and vv is not None and vh is not None:
        # VV/VH ratio in dB — captures cross-pol contrast
        vv_db = linear_to_db(vv)
        vh_db = linear_to_db(vh)
        ratio_db = vv_db - vh_db   # typically 3–12 dB; clip to [0, 15]
        ratio_norm = clip_db(ratio_db, 0.0, 15.0)
        ratio_norm = scale_to_unit(ratio_norm, 0.0, 15.0)
        bands.append(ratio_norm)
        band_names.append("VV_VH_ratio_norm")

    if not bands:
        raise ValueError("At least one of vv or vh must be provided.")

    return np.stack(bands, axis=0), band_names   # (C, H, W)


def normalize_scene_to_tensor(
    vv: Optional[np.ndarray] = None,
    vh: Optional[np.ndarray] = None,
    scene=None,
    db_min: float = DB_MIN_DEFAULT,
    db_max: float = DB_MAX_DEFAULT,
    add_ratio_band: bool = True,
    imagenet_normalize: bool = False,
    verbose: bool = True,
) -> "torch.Tensor":
    """
    Full scene normalisation → PyTorch float32 tensor.

    Accepts either (vv, vh) arrays directly, or a SARScene object.

    Parameters
    ----------
    vv, vh : np.ndarray | None
        Linear sigma-nought arrays.
    scene : SARScene | None
        If provided, vv/vh arrays are taken from scene.vv / scene.vh.
    db_min, db_max : float
    add_ratio_band : bool
        Add VV/VH ratio as a third channel (recommended for ship detection).
    imagenet_normalize : bool
        Apply ImageNet mean/std subtraction after [0,1] scaling.
        Use this when fine-tuning a backbone pretrained on ImageNet.
        The per-channel stats are replicated across the first 2 channels
        (VV → R, VH → G, ratio → B) as an approximation.
    verbose : bool

    Returns
    -------
    tensor : torch.Tensor  shape (C, H, W), dtype float32
        Ready to be unsqueeze(0) → (1, C, H, W) for model.forward().
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install torch")

    # Resolve input source
    if scene is not None:
        vv = vv if vv is not None else getattr(scene, "vv", None)
        vh = vh if vh is not None else getattr(scene, "vh", None)

    arr, band_names = normalize_scene_to_array(
        vv, vh, db_min=db_min, db_max=db_max, add_ratio_band=add_ratio_band
    )
    # arr: (C, H, W) float32

    if verbose:
        for i, name in enumerate(band_names):
            ch = arr[i]
            valid = ch[ch > 0]
            if len(valid):
                print(f"[normalize] {name}: "
                      f"mean={valid.mean():.3f}  std={valid.std():.3f}  "
                      f"range=[{ch.min():.3f}, {ch.max():.3f}]")

    tensor = torch.from_numpy(arr)   # (C, H, W) float32

    if imagenet_normalize:
        # Replicate ImageNet stats across SAR channels (approximation)
        n_channels = tensor.shape[0]
        means = torch.tensor(IMAGENET_MEAN[:n_channels])
        stds  = torch.tensor(IMAGENET_STD[:n_channels])
        # Reshape for broadcasting: (C,) → (C, 1, 1)
        means = means.view(-1, 1, 1)
        stds  = stds.view(-1, 1, 1)
        tensor = (tensor - means) / stds
        if verbose:
            print(f"[normalize] Applied ImageNet normalisation "
                  f"(mean={IMAGENET_MEAN[:n_channels]}, "
                  f"std={IMAGENET_STD[:n_channels]})")

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Tiling — patch extraction for training
# ─────────────────────────────────────────────────────────────────────────────

def extract_patches(
    tensor: "torch.Tensor",
    patch_size: int = 512,
    stride: int = 256,
    min_valid_fraction: float = 0.5,
) -> list[dict]:
    """
    Extract overlapping patches from a normalised scene tensor.

    Used to build training / inference chips from a full Sentinel-1 scene.
    Patches with too many nodata (zero) pixels are discarded.

    Parameters
    ----------
    tensor : torch.Tensor  shape (C, H, W)
        Normalised scene from normalize_scene_to_tensor().
    patch_size : int
        Square patch side length in pixels.  512 is standard for YOLOv8.
    stride : int
        Step between patch centres.  stride < patch_size gives overlap,
        which is important so ships near patch boundaries appear in
        multiple patches (data augmentation + no missed detections).
    min_valid_fraction : float
        Discard patches where fewer than this fraction of pixels are
        non-zero.  Removes water-edge and nodata patches.

    Returns
    -------
    patches : list[dict]
        Each dict has keys:
            "tensor"  : torch.Tensor  shape (C, patch_size, patch_size)
            "row_start", "col_start" : int  top-left pixel in scene coords
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install torch")

    C, H, W = tensor.shape
    patches = []
    total = 0
    kept = 0

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = tensor[:, r:r + patch_size, c:c + patch_size]
            total += 1

            # Validity check: fraction of non-zero pixels in first channel
            valid_frac = (patch[0] > 0).float().mean().item()
            if valid_frac < min_valid_fraction:
                continue

            patches.append({
                "tensor": patch.clone(),
                "row_start": r,
                "col_start": c,
            })
            kept += 1

    return patches


# ─────────────────────────────────────────────────────────────────────────────
# Save / load normalised GeoTIFF
# ─────────────────────────────────────────────────────────────────────────────

def save_normalized_geotiff(
    arr: np.ndarray,
    band_names: list[str],
    output_path: str,
    transform,
    crs,
) -> None:
    """
    Save a normalised multi-band array (C, H, W) to a GeoTIFF.

    Parameters
    ----------
    arr : np.ndarray  shape (C, H, W), float32, values in [0, 1]
    band_names : list[str]  length C
    output_path : str
    transform, crs : from the SARScene or load_gtc_geotiff()
    """
    import os
    from utils.io import save_geotiff

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    bands_dict = {name: arr[i] for i, name in enumerate(band_names)}
    save_geotiff(output_path, bands_dict, transform, crs)
    print(f"[normalize] Saved normalised GeoTIFF ({arr.shape[0]} bands) → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Inverse transform (for visualisation / debugging)
# ─────────────────────────────────────────────────────────────────────────────

def denormalize_to_db(
    norm: np.ndarray,
    db_min: float = DB_MIN_DEFAULT,
    db_max: float = DB_MAX_DEFAULT,
) -> np.ndarray:
    """
    Invert the [0,1] normalisation back to dB values.

    Useful for overlaying model outputs on physically interpretable imagery.

    Parameters
    ----------
    norm : np.ndarray  float32, values in [0, 1]
    db_min, db_max : float  must match the values used during normalisation

    Returns
    -------
    db : np.ndarray  float32, values in [db_min, db_max]
    """
    return (norm.astype(np.float64) * (db_max - db_min) + db_min).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Running synthetic test (no input file provided)...\n")

        # Synthetic sigma-nought values spanning the typical SAR range
        rng = np.random.default_rng(0)
        H, W = 256, 256

        # Simulate ocean + a ship patch
        sigma0_vv = rng.lognormal(mean=np.log(10**(-18/10)), sigma=0.5, size=(H,W)).astype(np.float32)
        sigma0_vh = rng.lognormal(mean=np.log(10**(-24/10)), sigma=0.5, size=(H,W)).astype(np.float32)
        sigma0_vv[100:120, 100:120] = 0.3    # ship
        sigma0_vh[100:120, 100:120] = 0.15   # ship

        print("Input (linear σ°):")
        print(f"  VV ocean mean : {sigma0_vv[50:80,50:80].mean():.5f}")
        print(f"  VV ship mean  : {sigma0_vv[100:120,100:120].mean():.5f}")

        # Step through the chain manually
        vv_db = linear_to_db(sigma0_vv)
        print(f"\nAfter dB conversion:")
        print(f"  VV ocean mean : {vv_db[50:80,50:80].mean():.1f} dB")
        print(f"  VV ship mean  : {vv_db[100:120,100:120].mean():.1f} dB")

        vv_clip = clip_db(vv_db)
        vv_norm = scale_to_unit(vv_clip)
        print(f"\nAfter clip+normalise:")
        print(f"  VV ocean mean : {vv_norm[50:80,50:80].mean():.3f}")
        print(f"  VV ship mean  : {vv_norm[100:120,100:120].mean():.3f}")
        print(f"  Contrast      : {vv_norm[100:120,100:120].mean() / vv_norm[50:80,50:80].mean():.1f}×")

        # Full multi-band normalisation
        arr, names = normalize_scene_to_array(sigma0_vv, sigma0_vh, add_ratio_band=True)
        print(f"\nMulti-band array: shape={arr.shape}, channels={names}")

        if TORCH_AVAILABLE:
            tensor = normalize_scene_to_tensor(sigma0_vv, sigma0_vh, verbose=True)
            print(f"\nOutput tensor: {tensor.shape}, dtype={tensor.dtype}")
            print(f"  Ready for model.forward(tensor.unsqueeze(0))")

            patches = extract_patches(tensor, patch_size=64, stride=32)
            print(f"\nPatches extracted: {len(patches)} patches of size 64×64")
        else:
            print("\n[note] PyTorch not installed — tensor output skipped.")

        # Roundtrip test
        norm = normalize_band(sigma0_vv)
        recovered_db = denormalize_to_db(norm)
        original_db = clip_db(linear_to_db(sigma0_vv))
        max_err = np.abs(recovered_db - original_db).max()
        print(f"\nRoundtrip error (norm → denorm vs original clipped dB): {max_err:.6f} dB")
        assert max_err < 1e-4, "Roundtrip error too large!"
        print("Roundtrip test PASSED.")
        sys.exit(0)

    # Real file mode
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".tif", "_norm.tif")

    from utils.io import load_geotiff, save_geotiff

    print(f"Loading: {input_path}")
    arrays, transform, crs = load_geotiff(input_path)

    vv = arrays.get("VV") or arrays.get("Sigma0_VV") or arrays.get("band_1")
    vh = arrays.get("VH") or arrays.get("Sigma0_VH") or arrays.get("band_2")

    arr_stacked, band_names = normalize_scene_to_array(vv, vh, add_ratio_band=True)
    save_normalized_geotiff(arr_stacked, band_names, output_path, transform, crs)

    if TORCH_AVAILABLE:
        tensor = normalize_scene_to_tensor(vv, vh, verbose=True)
        patches = extract_patches(tensor)
        print(f"\nExtracted {len(patches)} patches (512×512, stride=256)")
