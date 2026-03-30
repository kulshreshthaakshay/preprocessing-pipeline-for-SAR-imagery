"""
sar_pipeline/utils/viz.py
-------------------------
Quick-look image rendering for SAR data at any pipeline stage.

Usage:
    from utils.viz import quicklook, quicklook_scene
    quicklook(vv_arr, title="VV raw DN", save_path="out.png")

Dependencies:
    pip install matplotlib numpy pillow
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless-safe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _percentile_clip(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Clip to percentile range and normalise to [0, 1]."""
    lo = np.nanpercentile(arr, p_low)
    hi = np.nanpercentile(arr, p_high)
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-9)


def quicklook(
    arr: np.ndarray,
    title: str = "SAR band",
    cmap: str = "gray",
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Render a single 2D SAR array as a quick-look image.

    Parameters
    ----------
    arr : np.ndarray
        2D array at any scale (raw DN, linear σ°, dB, or normalised).
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap. "gray" is standard for SAR.
    save_path : str | None
        If given, save to this file instead of displaying.
    figsize : tuple
        Matplotlib figure size in inches.
    """
    display = _percentile_clip(arr)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#111")
    im = ax.imshow(display, cmap=cmap, interpolation="nearest")
    ax.set_title(title, color="white", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="normalised intensity")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[viz] Saved quicklook → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def quicklook_rgb(
    vv: np.ndarray,
    vh: np.ndarray,
    title: str = "VV / VH / VV-VH composite",
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Render a false-colour RGB composite commonly used for SAR ship detection:
        R = VV,  G = VH,  B = VV / VH ratio

    This highlights ships (bright in both channels) vs sea clutter.
    """
    r = _percentile_clip(vv)
    g = _percentile_clip(vh)
    # ratio: avoid log(0) with small epsilon
    ratio = np.log1p(vv + 1e-6) - np.log1p(vh + 1e-6)
    b = _percentile_clip(ratio)

    rgb = np.stack([r, g, b], axis=-1).clip(0, 1)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#111")
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, color="white", fontsize=12)
    ax.axis("off")

    # Legend patches
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=(1, 0, 0), label="R = VV"),
        Patch(facecolor=(0, 1, 0), label="G = VH"),
        Patch(facecolor=(0, 0, 1), label="B = VV/VH"),
    ]
    ax.legend(handles=legend, loc="lower right",
              facecolor="#333", labelcolor="white", fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[viz] Saved RGB quicklook → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def quicklook_scene(scene, save_dir: str = ".") -> None:
    """
    Convenience: render all available bands of a SARScene from ingest.py.

    Parameters
    ----------
    scene : SARScene
        Output of ingest_safe().
    save_dir : str
        Directory to write PNG files into.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    if scene.vv is not None:
        quicklook(scene.vv, title="VV — raw DN",
                  save_path=os.path.join(save_dir, "quicklook_vv.png"))
    if scene.vh is not None:
        quicklook(scene.vh, title="VH — raw DN",
                  save_path=os.path.join(save_dir, "quicklook_vh.png"))
    if scene.vv is not None and scene.vh is not None:
        quicklook_rgb(scene.vv, scene.vh,
                      save_path=os.path.join(save_dir, "quicklook_rgb.png"))
