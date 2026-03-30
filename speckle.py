"""
sar_pipeline/speckle.py
------------------------
Step 5: Adaptive speckle filtering for SAR imagery.

Implements two filters derived from the multiplicative speckle noise model:

  I(x,y) = R(x,y) · u(x,y)

where I = observed intensity, R = true backscatter, u = speckle (mean=1,
var=1/L, L = number of looks).

Both filters estimate R̂ as:
  R̂ = mean_local + W · (I − mean_local)

and differ only in how W is computed:

  Lee filter      W = (Cv² − Cv_s²) / Cv²
  Gamma-MAP       W = (alpha − L − 1) / (alpha − 1)
                  alpha = (1 + Cv_s²) / (Cv² − Cv_s²)

where Cv_s = 1/√L  (theoretical speckle coefficient of variation)
      Cv   = std(window) / mean(window)   (local measured Cv)

Key implementation choices:
  - All arithmetic in LINEAR power domain (not dB). Speckle is
    multiplicative in linear; filters derived for that domain.
  - Block-wise processing with overlap to handle large scenes without
    loading two copies of the full image in memory.
  - scipy.ndimage.uniform_filter / generic_filter for the moment
    statistics, which are considerably faster than explicit loops.
  - A refined Lee filter with edge-aligned 8-direction sub-windows
    (Lee-Refined or Lee-Sigma) is offered as an upgrade path.

Dependencies:
    pip install numpy scipy

Usage:
    from speckle import apply_speckle_filter

    # sigma0_linear: np.ndarray shape (H, W), float32, linear power
    filtered = apply_speckle_filter(sigma0_linear, method="lee", n_looks=1)
    filtered = apply_speckle_filter(sigma0_linear, method="gamma_map", n_looks=4)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter


# ─────────────────────────────────────────────────────────────────────────────
# Moment estimation helpers (fast, via uniform_filter)
# ─────────────────────────────────────────────────────────────────────────────

def _local_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute per-pixel local mean over a (window × window) box.
    uniform_filter is a box filter — equivalent to mean over the window,
    faster than a 2D convolution for large arrays.
    """
    return uniform_filter(arr.astype(np.float64), size=window, mode="mirror")


def _local_mean_sq(arr: np.ndarray, window: int) -> np.ndarray:
    """Local mean of squared values — used to derive local variance."""
    return uniform_filter(arr.astype(np.float64) ** 2, size=window, mode="mirror")


def _local_variance(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Local variance via E[X²] − E[X]²  (computational form).
    Clipped to zero to remove floating-point negatives in uniform regions.
    """
    mean = _local_mean(arr, window)
    mean_sq = _local_mean_sq(arr, window)
    return np.maximum(mean_sq - mean ** 2, 0.0)


def _local_cv(arr: np.ndarray, window: int, eps: float = 1e-10) -> np.ndarray:
    """
    Local coefficient of variation: Cv = std / mean.
    Protected against division by zero in masked / zero-valued regions.
    """
    mean = _local_mean(arr, window)
    var = _local_variance(arr, window)
    std = np.sqrt(var)
    return std / np.maximum(np.abs(mean), eps)


# ─────────────────────────────────────────────────────────────────────────────
# Lee filter
# ─────────────────────────────────────────────────────────────────────────────

def lee_filter(
    arr: np.ndarray,
    window_size: int = 7,
    n_looks: int = 1,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Lee adaptive speckle filter (Lee 1980).

    Derived as the MMSE linear estimator under the multiplicative model.

    Weight formula:
        Cv_s  = 1 / sqrt(n_looks)           # theoretical speckle Cv
        Cv    = std(window) / mean(window)   # local measured Cv
        W     = clip((Cv² − Cv_s²) / Cv², 0, 1)
        R̂    = mean + W · (I − mean)

    In homogeneous regions (Cv ≈ Cv_s):  W → 0, output → local mean.
    At edges / targets (Cv >> Cv_s):     W → 1, output → original pixel.

    Parameters
    ----------
    arr : np.ndarray  shape (H, W)
        Input image in LINEAR power (σ° or intensity).
        Must be float32 or float64.  Do NOT pass dB values.
    window_size : int
        Side length of the square estimation window.  Must be odd.
        Typical values: 5, 7, 9.  Larger = more smoothing, more blur.
    n_looks : int
        Effective number of looks of the SAR product.
        Sentinel-1 IW GRD: typically 4–5 equivalent looks.
        Single-look complex (SLC): 1.
        Check your product spec or estimate from homogeneous ocean.
    eps : float
        Small value added to denominators to prevent division by zero.

    Returns
    -------
    filtered : np.ndarray  shape (H, W), float32
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    arr = arr.astype(np.float64)

    # Theoretical speckle Cv
    cv_s = 1.0 / np.sqrt(n_looks)
    cv_s2 = cv_s ** 2

    # Local statistics
    mean = _local_mean(arr, window_size)
    var = _local_variance(arr, window_size)

    # Local Cv² = var / mean²
    mean2 = mean ** 2
    cv2 = var / np.maximum(mean2, eps)

    # Lee weight: W = (Cv² − Cv_s²) / Cv²
    # When Cv² <= Cv_s²  →  W = 0  (homogeneous, smooth fully)
    # When Cv²  > Cv_s²  →  W increases toward 1
    numerator = cv2 - cv_s2
    W = np.where(cv2 > eps, numerator / cv2, 0.0)
    W = np.clip(W, 0.0, 1.0)

    # MMSE estimate
    result = mean + W * (arr - mean)

    # Preserve nodata (zero / negative) pixels from input
    result = np.where(arr <= 0, arr, result)

    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Gamma-MAP filter
# ─────────────────────────────────────────────────────────────────────────────

def gamma_map_filter(
    arr: np.ndarray,
    window_size: int = 7,
    n_looks: int = 1,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Gamma-MAP adaptive speckle filter (Lopes et al. 1990).

    Assumes the true backscatter R follows a Gamma distribution.
    Derives the MAP estimator, which is more aggressive than Lee in
    homogeneous regions while better preserving point targets (ships).

    Weight formula:
        Cv_s  = 1 / sqrt(n_looks)
        alpha = (1 + Cv_s²) / (Cv² − Cv_s²)   # Gamma shape parameter
        W     = (alpha − L − 1) / (alpha − 1)
        R̂    = mean + W · (I − mean)

    When Cv² ≤ Cv_s²:  alpha is undefined (pure speckle) → W = 0.
    When Cv² >> Cv_s²: alpha is small, W approaches 1 (preserve edge).

    Parameters
    ----------
    arr : np.ndarray  shape (H, W)
        Input image in LINEAR power.  Must not be in dB.
    window_size : int
        Must be odd.  Typical: 7.
    n_looks : int
        Effective number of looks.
    eps : float
        Division safety floor.

    Returns
    -------
    filtered : np.ndarray  shape (H, W), float32
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    arr = arr.astype(np.float64)
    L = float(n_looks)

    cv_s = 1.0 / np.sqrt(L)
    cv_s2 = cv_s ** 2

    mean = _local_mean(arr, window_size)
    var = _local_variance(arr, window_size)
    mean2 = np.maximum(mean ** 2, eps)

    cv2 = var / mean2

    # Mask where Cv² > Cv_s² (heterogeneous — edge or target present)
    heterogeneous = cv2 > (cv_s2 + eps)

    # Alpha: Gamma distribution shape parameter
    # Safe computation: set alpha=1 (W=0) in homogeneous pixels
    denom_alpha = np.where(heterogeneous, cv2 - cv_s2, 1.0)
    alpha = (1.0 + cv_s2) / denom_alpha

    # Weight W = (alpha - L - 1) / (alpha - 1)
    # When alpha <= L+1: clamp W to 0 (strong smoothing)
    # When alpha is large (very homogeneous by this formula path): also W→0
    denom_w = np.maximum(alpha - 1.0, eps)
    W = (alpha - L - 1.0) / denom_w
    W = np.where(heterogeneous, W, 0.0)
    W = np.clip(W, 0.0, 1.0)

    result = mean + W * (arr - mean)
    result = np.where(arr <= 0, arr, result)

    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Refined Lee filter (directional sub-windows)
# ─────────────────────────────────────────────────────────────────────────────

def lee_refined_filter(
    arr: np.ndarray,
    window_size: int = 7,
    n_looks: int = 1,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Lee-Refined (directional) speckle filter.

    Improvement over standard Lee: instead of using a single square window
    for both weight computation and mean estimation, it selects the most
    homogeneous of 8 directional sub-windows to compute the local mean.
    This further reduces blurring across edges.

    The 8 sub-windows for a 7×7 window are:
        horizontal, vertical, diagonal (×2), and 4 rotated rectangles
    Each covers roughly half the full window area.

    The weight W is still computed from the full window (for stability).
    The mean used in the final estimator comes from the most homogeneous
    sub-window (lowest variance).

    Parameters
    ----------
    arr : np.ndarray  shape (H, W), float32/float64, linear power
    window_size : int  must be odd, ≥ 5
    n_looks : int
    eps : float

    Returns
    -------
    filtered : np.ndarray  shape (H, W), float32
    """
    if window_size % 2 == 0 or window_size < 5:
        raise ValueError(f"window_size must be odd and ≥ 5, got {window_size}")

    arr64 = arr.astype(np.float64)
    H, W = arr.shape

    # ── Compute weight W from the full window (same as standard Lee) ──────────
    mean_full = _local_mean(arr64, window_size)
    var_full = _local_variance(arr64, window_size)
    cv_s = 1.0 / np.sqrt(n_looks)
    cv_s2 = cv_s ** 2
    cv2_full = var_full / np.maximum(mean_full ** 2, eps)
    W = np.clip((cv2_full - cv_s2) / np.maximum(cv2_full, eps), 0.0, 1.0)

    # ── Define 8 directional sub-window kernels ───────────────────────────────
    # Each kernel is a boolean mask of shape (window_size, window_size).
    # The mask selects pixels in that direction from the centre.
    half = window_size // 2
    sz = window_size

    def _make_kernels(sz: int, half: int) -> list[np.ndarray]:
        """Return 8 directional boolean masks."""
        kernels = []
        # 1. Full top half
        k = np.zeros((sz, sz), dtype=bool)
        k[:half + 1, :] = True
        kernels.append(k)
        # 2. Full bottom half
        k = np.zeros((sz, sz), dtype=bool)
        k[half:, :] = True
        kernels.append(k)
        # 3. Full left half
        k = np.zeros((sz, sz), dtype=bool)
        k[:, :half + 1] = True
        kernels.append(k)
        # 4. Full right half
        k = np.zeros((sz, sz), dtype=bool)
        k[:, half:] = True
        kernels.append(k)
        # 5. Upper-left triangle
        k = np.zeros((sz, sz), dtype=bool)
        for r in range(sz):
            for c in range(sz):
                if r + c <= sz - 1:
                    k[r, c] = True
        kernels.append(k)
        # 6. Upper-right triangle
        k = np.zeros((sz, sz), dtype=bool)
        for r in range(sz):
            for c in range(sz):
                if c - r >= 0:
                    k[r, c] = True
        kernels.append(k)
        # 7. Lower-left triangle
        k = np.zeros((sz, sz), dtype=bool)
        for r in range(sz):
            for c in range(sz):
                if r - c >= 0:
                    k[r, c] = True
        kernels.append(k)
        # 8. Lower-right triangle
        k = np.zeros((sz, sz), dtype=bool)
        for r in range(sz):
            for c in range(sz):
                if r + c >= sz - 1:
                    k[r, c] = True
        kernels.append(k)
        return kernels

    kernels = _make_kernels(sz, half)

    # ── For each pixel, find sub-window with minimum local variance ───────────
    # We use generic_filter per sub-window kernel; pick the kernel that
    # gives the lowest variance (most homogeneous direction).
    best_mean = mean_full.copy()
    best_var = np.full_like(mean_full, np.inf)

    for kernel in kernels:
        flat = kernel.flatten().astype(np.float64)
        n_k = flat.sum()
        if n_k < 2:
            continue

        def _mean_fn(x, _flat=flat, _n=n_k):
            return np.dot(x, _flat) / _n

        def _var_fn(x, _flat=flat, _n=n_k):
            m = np.dot(x, _flat) / _n
            return np.dot(_flat, (x - m) ** 2) / (_n - 1)

        sub_mean = generic_filter(arr64, _mean_fn, size=sz, mode="mirror")
        sub_var = generic_filter(arr64, _var_fn, size=sz, mode="mirror")

        # Update best where this sub-window is more homogeneous
        mask = sub_var < best_var
        best_mean = np.where(mask, sub_mean, best_mean)
        best_var = np.where(mask, sub_var, best_var)

    # ── Apply estimator using directional mean ────────────────────────────────
    result = best_mean + W * (arr64 - best_mean)
    result = np.where(arr64 <= 0, arr64, result)

    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Block-wise processing for large scenes
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filter_blockwise(
    arr: np.ndarray,
    filter_fn,
    block_size: int = 1024,
    overlap: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Apply a filter function to a large array in overlapping blocks,
    then stitch results together.

    The overlap (in pixels) prevents border artefacts where local statistics
    at block edges are computed from a truncated window.  The overlap width
    should be at least window_size // 2.

    Parameters
    ----------
    arr : np.ndarray  shape (H, W)
    filter_fn : callable  signature filter_fn(block, **kwargs) → block
    block_size : int  pixels per block (rows and cols)
    overlap : int  overlap in pixels on each side
    **kwargs : passed through to filter_fn

    Returns
    -------
    out : np.ndarray  shape (H, W), same dtype as filter_fn output
    """
    H, W = arr.shape
    out = np.empty_like(arr, dtype=np.float32)

    row = 0
    while row < H:
        row_end = min(row + block_size, H)
        # Add overlap rows (clamped to image bounds)
        r0 = max(row - overlap, 0)
        r1 = min(row_end + overlap, H)

        col = 0
        while col < W:
            col_end = min(col + block_size, W)
            c0 = max(col - overlap, 0)
            c1 = min(col_end + overlap, W)

            block = arr[r0:r1, c0:c1]
            filtered_block = filter_fn(block, **kwargs)

            # Write only the non-overlapping centre region
            out_r0 = row - r0
            out_r1 = out_r0 + (row_end - row)
            out_c0 = col - c0
            out_c1 = out_c0 + (col_end - col)

            out[row:row_end, col:col_end] = filtered_block[out_r0:out_r1, out_c0:out_c1]
            col = col_end
        row = row_end

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

FILTER_REGISTRY = {
    "lee":          lee_filter,
    "gamma_map":    gamma_map_filter,
    "gamma-map":    gamma_map_filter,
    "lee_refined":  lee_refined_filter,
    "lee-refined":  lee_refined_filter,
    "none":         lambda arr, **_: arr.astype(np.float32),
}


def apply_speckle_filter(
    arr: np.ndarray,
    method: str = "lee",
    window_size: int = 7,
    n_looks: int = 1,
    blockwise: bool = True,
    block_size: int = 2048,
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply an adaptive speckle filter to a SAR intensity image.

    IMPORTANT: Input must be in LINEAR POWER (σ° or intensity).
    Do NOT pass dB values — the filter statistics are derived for
    the multiplicative noise model, which is linear.

    Parameters
    ----------
    arr : np.ndarray  shape (H, W), float32
        Calibrated (and optionally denoised) σ° array.
    method : str
        One of: "lee", "gamma_map", "lee_refined", "none".
        "lee"          — standard Lee MMSE filter. Fast, good general use.
        "gamma_map"    — Gamma-MAP. More aggressive smoothing of ocean,
                         better point target (ship) preservation.
        "lee_refined"  — Lee with directional sub-windows. Sharpest edges
                         but ~8× slower than standard Lee.
        "none"         — pass-through (no filtering).
    window_size : int
        Square estimation window side length.  Must be odd.
        7×7 is the standard for Sentinel-1 IW GRD ship detection.
        Larger windows → more smoothing → fewer speckle artefacts
        but more blurring of ship boundaries.
    n_looks : int
        Effective number of looks of the SAR product.
        Sentinel-1 IW GRD: ~4.9 equivalent looks (use 4 or 5).
        If unsure: estimate from a uniform ocean patch as
        n_looks ≈ mean² / variance.
    blockwise : bool
        Process in blocks to limit peak RAM usage.  Should be True
        for full Sentinel-1 scenes (25000 × 16000 px).
    block_size : int
        Block side length in pixels.  2048 × 2048 uses ~100 MB/block.
    verbose : bool

    Returns
    -------
    filtered : np.ndarray  shape (H, W), float32
    """
    method_key = method.lower().replace(" ", "_")
    if method_key not in FILTER_REGISTRY:
        raise ValueError(
            f"Unknown speckle filter '{method}'. "
            f"Choose from: {list(FILTER_REGISTRY.keys())}"
        )

    if method_key == "none":
        if verbose:
            print("[speckle] No filtering requested — returning input unchanged.")
        return arr.astype(np.float32)

    if verbose:
        print(f"[speckle] Applying {method} filter "
              f"(window={window_size}×{window_size}, n_looks={n_looks}) "
              f"to array shape {arr.shape}...")

    filter_fn = FILTER_REGISTRY[method_key]
    overlap = window_size // 2 + 2   # ensure full window at block edges

    if blockwise:
        result = _apply_filter_blockwise(
            arr,
            filter_fn=filter_fn,
            block_size=block_size,
            overlap=overlap,
            window_size=window_size,
            n_looks=n_looks,
        )
    else:
        result = filter_fn(arr, window_size=window_size, n_looks=n_looks)

    if verbose:
        # Report effective smoothing (compare variance before/after)
        valid_in = arr[arr > 0]
        valid_out = result[result > 0]
        if len(valid_in) and len(valid_out):
            cv_in = np.std(valid_in) / np.mean(valid_in)
            cv_out = np.std(valid_out) / np.mean(valid_out)
            print(f"[speckle] Cv before: {cv_in:.4f}  after: {cv_out:.4f}  "
                  f"(reduction: {(1 - cv_out/cv_in)*100:.1f}%)")

    return result


def estimate_n_looks(arr: np.ndarray, mask_percentile: float = 20.0) -> float:
    """
    Estimate the effective number of looks from a homogeneous region
    of the image (e.g. open ocean away from ships).

    n_looks ≈ mean² / variance   (from the multiplicative noise model)

    Parameters
    ----------
    arr : np.ndarray  shape (H, W), linear power
    mask_percentile : float
        Use only pixels below this percentile of intensity (to select
        ocean pixels and exclude bright targets like ships).

    Returns
    -------
    float : estimated effective number of looks
    """
    flat = arr[arr > 0].ravel()
    threshold = np.percentile(flat, mask_percentile)
    ocean = flat[flat <= threshold]
    if len(ocean) < 100:
        raise ValueError("Too few pixels for reliable n_looks estimation.")
    mean = ocean.mean()
    var = ocean.var()
    n_looks = (mean ** 2) / max(var, 1e-20)
    return float(n_looks)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python speckle.py input.tif [output.tif] [lee|gamma_map|lee_refined]")
        print("\nRunning synthetic test...")

        # Create synthetic speckled image: uniform background + bright target
        rng = np.random.default_rng(42)
        H, W = 512, 512
        true_r = np.ones((H, W), dtype=np.float32) * 0.01   # ocean ~−20 dB
        # Add a bright ship-like target
        true_r[220:240, 240:260] = 0.5                        # ship ~−3 dB
        # Multiply by Gamma-distributed speckle (n_looks=4)
        speckle = rng.gamma(shape=4, scale=1.0/4, size=(H, W)).astype(np.float32)
        observed = true_r * speckle

        print(f"Synthetic image: {H}×{W}, n_looks=4")
        print(f"Ocean Cv (true): {1/np.sqrt(4):.3f}")

        n_est = estimate_n_looks(observed)
        print(f"Estimated n_looks from image: {n_est:.2f}  (should be ~4)")

        for method in ["lee", "gamma_map"]:
            filtered = apply_speckle_filter(
                observed, method=method, window_size=7, n_looks=4, verbose=True
            )
            # Check ship pixel preservation
            ship_response = filtered[220:240, 240:260].mean()
            ocean_response = filtered[50:150, 50:150].mean()
            print(f"  {method}: ship mean={ship_response:.4f}, "
                  f"ocean mean={ocean_response:.4f}, "
                  f"contrast={ship_response/ocean_response:.1f}×")

        sys.exit(0)

    # Real file mode
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".tif", "_speckle.tif")
    method = sys.argv[3] if len(sys.argv) > 3 else "lee"

    import rasterio
    from utils.io import save_geotiff, load_geotiff

    arrays, transform, crs = load_geotiff(input_path)
    filtered_arrays = {}
    for band_name, arr in arrays.items():
        print(f"\nProcessing band: {band_name}")
        n_est = estimate_n_looks(arr)
        print(f"  Estimated n_looks: {n_est:.2f}")
        filtered_arrays[band_name] = apply_speckle_filter(
            arr, method=method, window_size=7,
            n_looks=max(1, round(n_est)), verbose=True
        )

    save_geotiff(output_path, filtered_arrays, transform, crs)
    print(f"\n[speckle] Saved → {output_path}")
