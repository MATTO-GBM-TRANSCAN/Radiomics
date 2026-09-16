# Radiomic Features Extraction

This workflow computes radiomic features from images and their tumour contours
using PyRadiomics, following the configuration described in the paper
(see [Citation](../README.md#citation)).

## Contents

```
Radiomic_Feature_Extraction/
├── run_extraction.sh                                    --> launches feature extraction
├── input_templates/
│   └── extraction/
│       ├── extraction_parameters.yaml                   --> PyRadiomics configuration used in the paper
│       └── input_template.csv                           --> template for input file
└── code/
    ├── batch_parallel_extraction_pyradiomics.py         --> main script: extracts radiomic features for each image/mask pair
    ├── imageoperations.py                               --> patched PyRadiomics file adding the HEqualized image type 
    └── batch_analyze_discretization.py                  --> optional helper to choose the binWidth setting
    └── split_features_by_type.py                        --> optional helper to improve features .csv visualization
```

## Extracting features

Copy `input_templates/extraction/input_template.csv` and `extraction_parameters.yaml` (described above) into your input folder and edit them for your experiment..

### Folder structure requirements

Predefining this folder structure is recommended to facilitate running the code.

`INPUT_DIR` → `/input`:

```
input/                          <- INPUT_DIR
├── input.csv                      <- copied from input_templates/extraction/input_template.csv (INPUT_CSV_NAME)
├── extraction_parameters.yaml     <- copied from input_templates/extraction/extraction_parameters.yaml
├── images/
│   └── patient01_T1wCE.nii.gz     <- referenced by the `Image` column
└── masks/
    └── patient01_mask.nii.gz      <- referenced by the `Mask` column
```

`OUTPUT_DIR` → `/output`:

```
output/                              <- OUTPUT_DIR (starts empty)
├── radiomic_features_2026-01-15.csv    <- extracted radiomic features
├── log_2026-01-15.txt                  <- log
└── temp/                               <- cached per-case results
```

`INPUT_CSV_NAME` → `input.csv` (default; set it to match your filename if different):

### Input template reference

<details>
<summary><code>input_template.csv</code> column reference</summary>

| Column | Required | Description | Possible values |
|---|---|---|---|
| `Image` | required | Path to the image file (readable by SimpleITK), as seen *inside the container*, i.e. under `/input/...`. | Any file path SimpleITK can read (e.g. `.nii.gz`, `.nrrd`, `.mha`). |
| `Mask` | required | Path to the matching tumour contour (mask) file, as seen *inside the container*, i.e. under `/input/...` | Same as `Image`. |
| `Patient` | optional | Patient identifier. If omitted, one is generated automatically (`Pt <row number>`). | Any text, used only as a label. |
| `Reader` | optional | Identifies who created the segmentation (useful when you have more than one reader/contour per patient). If omitted, defaults to `N-A`. | Any text, used only as a label. |
| `Contour` | optional | Only needed if the same `Patient` appears in more than one row (e.g. one row for the MR contour, one for the PET contour). Used by [`prepare_features_and_endpoints.py`](../Radiomics_Model_Development/README.md#preparing-features-and-endpoints) to combine that patient's rows into one, suffixing each feature with its `Contour` value (e.g. `original_shape_Elongation-MR`, `original_shape_Elongation-PET`), the same `-MR`/`-PET` convention [`MATTO-GBM_Models/`](../MATTO-GBM_Models) uses when combining MR and PET features. | Any text, e.g. `MR`, `PET`. |

</details>

<details>
<summary><code>extraction_parameters.yaml</code> setting reference</summary>

| Setting | Value used in the paper | Description |
|---|---|---|
| `imageType` | `Original`, `HEqualized`, `Wavelet` | Which versions of the image features are computed on: the image as-is, after histogram equalization (the `HEqualized` type added by `imageoperations.py`), and after a wavelet transform. |
| `featureClass` | `shape`, `firstorder`, `glcm`, `glrlm`, `glszm`, `gldm` | Which feature families to compute, with every feature in each family enabled. |
| `normalize` | `true` | Normalizes image intensities before computing features, reduces differences between scanners/vendors (MR signal is relative, not absolute). |
| `normalizeScale` | `100` | Scale factor applied after normalization, so the resulting bin width still makes sense. |
| `interpolator` | `sitkBSpline` | Interpolation method used when resampling the image. |
| `resampledPixelSpacing` | `[0.5, 0.5, 1]` | Resamples every image to this voxel spacing (in mm) before extraction, so features are comparable across scans acquired at different resolutions. |
| `minimumROIDimensions` | `3` | Minimum number of dimensions the tumour contour (ROI) must span to be processed. |
| `minimumROISize` | `50` | Minimum number of voxels the ROI must contain to be processed. |
| `binWidth` | `10` | Width of the bins used to discretize gray levels (yields 40–120 bins for the GTV contours used in the paper). |
| `voxelArrayShift` | `300` | Shift applied to gray values after normalization, so that (almost) all values are positive, needed for some first-order features. |
| `label` | `1` | The label value in the mask file that identifies the tumour contour. |

</details>

<details>
<summary>Choosing a bin width (optional)</summary>

`code/batch_analyze_discretization.py` is how `binWidth: 10` above was chosen: for each image/mask pair in your `input.csv`, it computes how many gray levels result from candidate bin widths 5 to 12. Optional; only needed to re-check the value against your own images, using the same input as the extraction step:

```bash
docker run --rm \
  -v /path/to/your/input:/input \
  -v /path/to/your/output:/output \
  matto-radiomics \
  python -m matto_radiomics.feature_extraction.batch_analyze_discretization \
    --input_csv /input/input.csv
```

It reads the same CSV as the extraction step (`--input_csv` defaults to `/input/input.csv`, same as `batch_parallel_extraction_pyradiomics`) and writes `discretization.csv` to `/output`, with one row per case and a `n_bins_<width>` column for each candidate bin width (5 to 12).

</details>

### Run the code

After setting up the recommended folder structure, please adapt `input.csv` and `extraction_parameters.yaml` to your own experiment. Edit `INPUT_DIR`, `OUTPUT_DIR` and `INPUT_CSV_NAME` at the top of `run_extraction.sh`, then run it from inside this folder:

```bash
cd Radiomic_Feature_Extraction
./run_extraction.sh
```

