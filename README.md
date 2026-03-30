# SAR Ship Detection — Preprocessing Pipeline

End-to-end Sentinel-1 GRD preprocessing: raw SAFE → model-ready tensors.

```
ingest → calibrate → denoise → terrain → speckle → normalize → export
```

---

## Repository layout

```
sar_pipeline/
├── ingest.py        Step 1  Read SAFE, extract VV/VH DN arrays
├── calibrate.py     Step 2  DN → σ° (sigma-nought) via calibration LUT
├── denoise.py       Step 3  Subtract thermal noise (NESZ) from σ°
├── terrain.py       Step 4  Range-Doppler terrain correction via SNAP
├── speckle.py       Step 5  Lee / Gamma-MAP adaptive speckle filter
├── normalize.py     Step 6  σ° → dB → clip [−30,0] → [0,1] tensor
├── export.py        Step 7  Write GeoTIFF master + PNG patches + .pt tensor
├── pipeline.py             Full orchestrator (runs all steps in sequence)
├── validate.py             Inter-module validation harness (run this first)
└── utils/
    ├── io.py               GeoTIFF save/load helpers
    └── viz.py              Quicklook PNG renderer
```

---

## 1. Environment setup

### 1-A  Python dependencies

```bash
pip install numpy scipy rasterio lxml torch torchvision pillow tqdm pyproj
```

### 1-B  ESA SNAP (required for terrain correction only)

Download from:
    https://step.esa.int/main/download/snap-download/

Choose **"SNAP with Sentinel Toolboxes"**. After install, configure snappy:

```bash
# Linux / macOS
/path/to/snap/bin/snappy-conf /path/to/python3

# Windows (run as administrator)
C:\snap\bin\snappy-conf.bat C:\Python311\python.exe
```

Add snappy to PYTHONPATH:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PYTHONPATH=$PYTHONPATH:~/.snap/snap-python
```

Verify:

```python
import snappy
print("SNAP version:", snappy.ProductIO)
```

SNAP also requires Java 11+. Make sure JAVA_HOME is set:

```bash
java -version   # must show 11 or higher
```

### 1-C  Verify all Python imports work

```bash
cd sar_pipeline/
python -c "
import numpy, scipy, rasterio, lxml, torch, PIL, tqdm, pyproj
print('All dependencies OK')
"
```

---

## 2. Dataset sources — where to download Sentinel-1 data

You need one or more Sentinel-1 IW GRDH scenes. Three free sources:

### Source A — Copernicus Data Space (recommended, fastest)

URL: https://dataspace.copernicus.eu/

1. Create a free account at dataspace.copernicus.eu
2. Go to "Explore" → select "Sentinel-1"
3. Filter:
   - Product type: GRD
   - Mode: IW (Interferometric Wide Swath)
   - Polarisation: VV+VH (dual-pol)
   - Area: a coastal region with known ship traffic
     (Strait of Malacca, English Channel, Singapore Strait work well)
4. Download the .SAFE.zip file
5. Unzip: `unzip S1A_IW_GRDH_....zip -d ./data/`

Direct API download (after login):

```bash
pip install sentinelsat

python - <<'EOF'
import os
from sentinelsat import SentinelAPI

# Create data directory
os.makedirs("data", exist_ok=True)

api = SentinelAPI(
    "your_username",
    "your_password",
    "https://apihub.copernicus.eu/apihub"
)
products = api.query(
    # This polygon precisely encompasses the Singapore Strait:
    area="POLYGON((103.6 1.1, 104.2 1.1, 104.2 1.5, 103.6 1.5, 103.6 1.1))",
    date=("20230601", "20230630"),
    platformname="Sentinel-1",
    producttype="GRD",
    polarisationmode="VV VH",
)
api.download_all(products, directory_path="data")
EOF
```

### Source B — Alaska Satellite Facility (ASF) Vertex

URL: https://search.asf.alaska.edu/

1. Create free account at vertex.daac.asf.alaska.edu
2. Search by area, date, or orbit number
3. Select "Sentinel-1 GRDH" products
4. Download directly or use the bulk downloader

ASF Python download:

```bash
pip install asf_search

python - <<'EOF'
import os
import asf_search as asf

# Create data directory
os.makedirs("data", exist_ok=True)

results = asf.search(
    platform=asf.PLATFORM.SENTINEL1,
    processingLevel=asf.PRODUCT_TYPE.GRD_HD,
    # This polygon precisely encompasses the Singapore Strait:
    intersectsWith="POLYGON((103.6 1.1,104.2 1.1,104.2 1.5,103.6 1.5,103.6 1.1))",
    maxResults=5,
)
session = asf.ASFSession().auth_with_creds("user", "password")
results.download(path="data", session=session)
EOF
```

### Source C — SAR-Ship / HRSID (pre-annotated ship detection datasets)

For training the detection model you will also need annotated data.
These datasets provide pre-cropped SAR chips with ship annotations:

**SAR-Ship dataset**
- Paper: Wang et al. (2019), "A SAR Dataset of Ship Detection for Deep Learning"
- Download: https://github.com/CAESAR-Radi/SAR-Ship-Dataset
- Format: 256×256 PNG chips + JSON annotations
- Scenes: Sentinel-1 IW and EW GRD, 39,848 ship chips

**HRSID (High-Resolution SAR Images Dataset)**
- Paper: Wei et al. (2020)
- Download: https://github.com/chaozhong2010/HRSID
- Format: COCO JSON format, 5,604 images, 16,951 ship instances
- Scenes: Sentinel-1B IW, SM, EW GRD

**SSDD (SAR Ship Detection Dataset)**
- Download: https://github.com/TianwenZhang0825/Official-SSDD
- Format: Pascal VOC XML + COCO JSON
- 1,160 images, 2,456 ship instances

> Note: SAR-Ship and HRSID are pre-cropped and don't require running
> this preprocessing pipeline. Use them for model training.
> Use this pipeline to preprocess new raw SAFE files for inference.

---

## 3. Sequential run order

Run in this exact order. Each step's output is the next step's input.

```
Step 1  python ingest.py     S1A_....SAFE
Step 2  python calibrate.py  S1A_....SAFE   output/calibrated.tif
Step 3  python denoise.py    S1A_....SAFE   output/denoised.tif
Step 4  python terrain.py    S1A_....SAFE   output/scene_GTC.tif   snappy
Step 5  python speckle.py    output/scene_GTC.tif  output/speckled.tif  lee
Step 6  (normalise is embedded in export.py — no standalone run needed)
Step 7  python export.py     output/scene_GTC.tif  output/  SCENE_ID
```

Or run everything in one command via the orchestrator:

```bash
python pipeline.py /data/S1A_IW_GRDH_....SAFE \
    --output-dir output/ \
    --dem "SRTM 1Sec HGT" \
    --spacing 10 \
    --backend snappy \
    --save-intermediates \
    --verbose
```

---

## 4. Detailed step-by-step preprocessing

### Step 1 — Ingest

Reads the raw SAFE package, extracts VV and VH DN arrays, parses
calibration and noise XML paths.

```bash
python ingest.py /data/S1A_IW_GRDH_1SDV_20230601T054912_048832.SAFE
```

What to check:
- Prints VV and VH shapes (expect ~25000 × 16700 for IW full scene)
- DN min should be 0 (masked ocean edge), max near 50000–65000
- calib_xml and noise_xml should each have 2 entries (VV, VH)
- CRS printed (expect WGS84 or UTM zone)

### Step 2 — Calibrate

Converts raw DN to sigma-nought in linear power using the calibration LUT.

```bash
python calibrate.py /data/S1A_....SAFE output/calibrated.tif
```

What to check:
- σ° median printed — should be −18 to −12 dB for open ocean
- σ° max should be < 5 dB (bright ships/corners)
- Output GeoTIFF written with 2 bands (VV, VH)

Sanity check command:
```bash
python -c "
import rasterio, numpy as np
with rasterio.open('output/calibrated.tif') as src:
    vv = src.read(1)
valid = vv[vv > 0]
db = 10 * np.log10(valid)
print(f'VV median: {np.median(db):.1f} dB  (expect -18 to -12 dB for ocean)')
print(f'VV max:    {db.max():.1f} dB         (expect < 5 dB)')
"
```

### Step 3 — Denoise

Subtracts the NESZ thermal noise from calibrated σ°.

```bash
python denoise.py /data/S1A_....SAFE output/denoised.tif
```

What to check:
- "Range LUT: N azimuth vectors" printed — confirm N > 100
- "Azimuth LUT found" for post-2021 scenes; "range correction only" for older
- Median σ° change should be < 3 dB (noise is small fraction of signal)
- "Pixels at noise floor" fraction should be < 15% for most ocean scenes

### Step 4 — Terrain Correction

Runs SNAP's Range-Doppler terrain correction. This step takes
5–20 minutes depending on scene size and machine speed.
SNAP will automatically download the SRTM DEM if not cached.

```bash
python terrain.py /data/S1A_....SAFE output/scene_GTC.tif snappy
```

What to check:
- Output GeoTIFF has a valid CRS (check with gdalinfo or rasterio)
- Pixel spacing should be ~0.0001° (geographic) or ~10m (UTM)
- The image should be slightly smaller than the raw scene
  (terrain correction crops the distorted edges)

```bash
python -c "
import rasterio
with rasterio.open('output/scene_GTC.tif') as src:
    print('CRS:      ', src.crs)
    print('Shape:    ', src.height, '×', src.width)
    print('Pixel dx: ', abs(src.transform.a), 'CRS units')
    print('Bands:    ', src.count)
    print('Nodata:   ', src.nodata)
"
```

### Step 5 — Speckle filter

Applies Lee adaptive filter to smooth ocean clutter.
Input must be a GeoTIFF of linear σ° (output of terrain.py or denoised.tif).

```bash
python speckle.py output/scene_GTC.tif output/speckled.tif lee
```

What to check:
- Cv (coefficient of variation) should decrease — printed by the script
- Typical: Cv drops from ~0.8–1.0 to ~0.3–0.5 for IW ocean
- Ship pixels should remain bright (check with QGIS or the quicklook)

### Step 6 — Normalize (via export)

Normalisation runs inside export.py automatically.
No separate step needed — it is called by export_scene().

### Step 7 — Export

Writes the three output formats from the terrain-corrected GeoTIFF:

```bash
python export.py output/scene_GTC.tif output/ S1A_048832
```

Outputs written:
```
output/
  S1A_048832_sigma0.tif           float32 GeoTIFF master (archival)
  patches/
    images/
      S1A_048832_r00000_c00000.png   512×512 uint8 PNG patch (training)
      S1A_048832_r00000_c00256.png
      ...
    labels/
      S1A_048832_r00000_c00000.txt   empty YOLO label files (fill with annotations)
    S1A_048832_patches.json          sidecar index (patch → geo coordinates)
  tensors/
    S1A_048832.pt                    float32 PyTorch tensor (C, H, W)
```

---

## 5. Validate all scripts against each other

Run the validation harness before processing real data.
It checks that every module's output satisfies the contract expected
by the next module.

### 5-A  Synthetic validation (no SAFE file required)

```bash
python validate.py --synthetic --output-dir output/validate/
```

This tests steps 2, 3, 5, 6 on synthetic numpy arrays with known values.
All checks should print PASS. If any FAIL, fix the module before continuing.

Expected output:
```
SYNTHETIC VALIDATION (no SAFE file required)
...
Step 2 / calibrate (synthetic)
  PASS  VV: no NaN                               False
  PASS  VV: median in [−30, −5] dB              -17.82
Step 3 / denoise (synthetic)
  PASS  Denoised: no NaN                         False
  PASS  Denoised: all values > 0                 0.00000010
  PASS  Median shift ≤ 3 dB                      0.481
Step 5 / speckle (lee)
  PASS  No NaN                                   False
  PASS  Cv reduced                               0.5123 < 0.9841
  PASS  Ship > ocean (contrast preserved)        0.29101 > 0.01423
Step 5 / speckle (gamma_map)
  PASS  No NaN                                   False
  PASS  Cv reduced                               0.4812 < 0.9841
  PASS  Ship > ocean (contrast preserved)        0.30201 > 0.01201
Step 6 / normalize
  PASS  Output shape is 3-D (C, H, W)            3
  PASS  Channel count ≥ 2                        3
  PASS  VV_norm: min ≥ 0.0                       0.0
  PASS  VV_norm: max ≤ 1.0                       1.0
  PASS  VH_norm: mean in (0.1, 0.9)              0.3421

OVERALL: PASS  N/N checks passed
```

### 5-B  Full validation against a real SAFE file

```bash
python validate.py /data/S1A_....SAFE \
    --output-dir output/validate/ \
    --steps 1,2,3,5,6,7 \
    --verbose
```

Skip step 4 (terrain) on first run if SNAP isn't configured yet:

```bash
python validate.py /data/S1A_....SAFE \
    --output-dir output/validate/ \
    --steps 1,2,3 \
    --verbose
```

Validation report saved to:
    output/validate/validation_report.json

### 5-C  What each check confirms

| Step | Check | Expected |
|------|-------|----------|
| 1 Ingest | VV/VH arrays loaded | shape (H,W), dtype float32 |
| 1 Ingest | DN max in range | ≤ 65535 (uint16 max) |
| 1 Ingest | calib/noise XMLs found on disk | both VV and VH |
| 2 Calibrate | σ° median in dB | −30 to −5 dB (ocean) |
| 2 Calibrate | σ° max < 10 dB | no calibration runaway |
| 2 Calibrate | no NaN | LUT interpolation stable |
| 3 Denoise | median shift < 3 dB | noise << signal |
| 3 Denoise | all values > 0 | clamp applied |
| 3 Denoise | noise_grid stored in meta | for downstream inspection |
| 4 Terrain | CRS defined | SNAP wrote valid geospatial |
| 4 Terrain | pixel spacing plausible | ~10m or ~0.0001° |
| 5 Speckle | Cv reduced | ocean smoothed |
| 5 Speckle | ship > ocean contrast | targets preserved |
| 6 Normalize | values in [0, 1] | no overflow |
| 6 Normalize | 3 channels | VV + VH + ratio band |
| 7 Export | GeoTIFF written | file > 0 bytes |
| 7 Export | PNG patches written | at least 1 chip |
| 7 Export | sidecar JSON has required keys | georef chain intact |

---

## 6. Common errors and fixes

**ImportError: No module named snappy**
→ snappy not on PYTHONPATH. Run:
  `export PYTHONPATH=$PYTHONPATH:~/.snap/snap-python`
  and restart your terminal.

**Java not found / SNAP won't start**
→ Install JDK 11: `sudo apt install openjdk-11-jdk`
  then: `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`

**No calibrationVector elements found**
→ Wrong XML path. Run ingest.py and confirm calib_xml paths exist.
  Some SAFE zips are corrupted — re-download.

**σ° median printed as −40 dB or lower**
→ Calibration LUT not parsed correctly — check that noise XML tag is
  "sigmaNought" not "betaNought" or "dn". See calibrate.py line 75.

**Terrain correction: DEM download fails**
→ SNAP needs internet access to download SRTM. Set proxy if behind firewall:
  In SNAP GUI: Tools → Options → Network → Proxy Settings
  or set environment variable: HTTP_PROXY=http://proxy:port

**PNG patches all discarded (0 kept)**
→ scene.vv pixels are all zero — either nodata mask covers the whole scene
  (scene was acquired over land only) or the SAFE file is corrupted.
  Increase --min-valid-fraction or check the quicklook image.

**validate.py: "median in [-30,-5] dB" FAIL for land scene**
→ Expected — the check is tuned for ocean. Over dense urban areas
  σ° can reach −5 to +5 dB. The check is a sanity warning, not a hard error.

---

## 7. Complete one-liner for a new scene

```bash
# 1. Validate environment (synthetic, no SAFE needed)
python validate.py --synthetic

# 2. Process the scene
python pipeline.py /data/S1A_IW_GRDH_1SDV_20230601T054912_048832.SAFE \
    --output-dir output/ \
    --spacing 10 \
    --backend snappy \
    --save-intermediates

# 3. Export to training formats
python export.py output/S1A_IW_GRDH_1SDV_20230601T054912_048832_04_GTC.tif \
    output/ S1A_048832

# 4. Validate the outputs
python validate.py /data/S1A_IW_GRDH_1SDV_20230601T054912_048832.SAFE \
    --output-dir output/validate/ \
    --steps 1,2,3,5,6,7

# 5. Check the quicklook
python -c "
from utils.viz import quicklook_scene
from ingest import ingest_safe
scene = ingest_safe('/data/S1A_....SAFE')
quicklook_scene(scene, save_dir='output/quicklooks/')
print('Quicklooks written to output/quicklooks/')
"
```
