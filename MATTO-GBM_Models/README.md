# MATTO-GBM Models

Script-based pipeline to apply the PET|MR radiomics models for the prediction of treatment outcome in recurrent Glioblastoma (GBM).


## Available radiomics models and their inputs

Below is a summary of the available radiomics models and their required inputs. For a detailed description of the imaging protocols and segmentation definitions please refer to the associated paper in [`Citation`](../README.md#citation).

If your dataset does not include tumour segmentations for recurrent GBM, please refer to the segmentation pipeline in [MATTO-GBM-TRANSCAN/Segmentation](https://github.com/MATTO-GBM-TRANSCAN/Segmentation). The resulting segmentations should be reviewed and clinically validated by a qualified medical professional before being used as input for the radiomics models.


**Acute Recurrence (AR)**

| Model | Image(s)/Segmentation(s) | Clinical variables |
|---|---|---|
| `AcuteRecurrence-MR+PET-GTV` | T1wCE / contrast-enhanced tumour GTV(MR) and FET-PET / uptake tumour GTV(PET) | - |
| `AcuteRecurrence-MR-MRandPET+Clinical` | T1wCE / MR and PET intersection contour | Age, Sex, (Re-)current GBM localisation: Frontal |

The model weights can be downloaded from Zenodo: **[10.5281/zenodo.22641702](https://doi.org/10.5281/zenodo.22641702)**.

**Overall Survival (OS)**

| Radiomic Feature | Image / contour | Rule |
|---|---|---|
| `wavelet-HHH_glrlm_RunLengthNonUniformityNormalized` | T1wCE / MR and PET intersection contour | radiomic feature value > threshold -> **Long OS**; radiomic feature value <= threshold -> **Short OS** |

Threshold = 0.405139505.

The Overall Survival rule is already built into the inference code.

For all definitions, the time to event was determined from the end of radiotherapy treatment to the date of the specific event.


## Contents

```text
MATTO-GBM_Models/
├── run_inference_windows.bat             --> launches the inference workflow on Windows
├── run_inference_linux.sh                --> launches the inference workflow on Linux/macOS
├── input_templates/
│   └── inference_config_template.json    --> template containing paths and parameters for one run
└── code/
    ├── run_inference.py                  --> reads the config, extracts features and writes the final predictions
    ├── run_predictions.py                --> applies the radiomics models and outcome rules
    ├── reproduce_paper_results.py        --> applies the published Acute Recurrence models
    ├── stratification.py                 --> combines Acute Recurrence and Overall Survival into the final stratification
    └── model_archives.py                 --> loads local model ZIP files or downloads them from Zenodo
```



## Requirements

### Folder structure

A typical input folder is:

```text
input/                                      <- INPUT_DIR
├── inference_config.json                   <- copied from input_templates/inference_config_template.json
├── Patient01_T1wCE.nii.gz                  <- input T1wCE image
├── Patient01_FETPET.nii.gz                 <- input FET-PET image
├── Patient01_MR_contour.nii.gz             <- MR tumour contour
├── Patient01_PET_contour.nii.gz            <- PET tumour contour
└── Patient01_intersection_contour.nii.gz   <- MR/PET intersection contour
```

If the pretrained model files are already stored locally, a typical models folder is:

```text
models/                                     <- MODELS_DIR
├── AcuteRecurrence-MR+PET-GTV.zip
└── AcuteRecurrence-MR-MRandPET+Clinical.zip
```

The models folder can also contain the already extracted model directories instead of the ZIP files.

### Input files templates

Copy `input_templates/inference_config_template.json` into your input folder, rename it to `inference_config.json` and edit the paths and parameters. Paths in the JSON must be written as they are seen **inside the Docker container**, i.e. below `/input/...`, `/output/...` or `/models/...`.

<details>
<summary><code>inference_config_template.json</code> field reference</summary>

| Field | Required | Description | Possible values |
|---|---|---|---|
| `patient_id` | required | Patient identifier written to the output CSV. | Any text identifier. |
| `experiment_name` | required | Names the results subfolder. | Any text, e.g. `"MATTO_GBM_inference_00"`. |
| `output_dir` | required | Docker output mount point. | Usually `"/output"`. |
| `model_source` | required | Defines how model weights are loaded. | `"zenodo"`, `"local"` or `"mounted"`. |
| `models_path` | required for local/mounted models | Path to a model ZIP file, a folder containing model ZIP files, or a folder containing the extracted model directories. | Usually `"/models"`. |
| `zenodo_reference` | required when `model_source="zenodo"` | Zenodo record or DOI used to download model weights. | `"10.5281/zenodo.22641702"`. |
| `t1wce` | required for the T1wCE+FET-PET model; also used by the intersection workflow when `t1wce_intersection` is empty | Path to the contrast-enhanced T1-weighted MR image. | File path under `/input/...` readable by SimpleITK. |
| `mr_contour` | required for the T1wCE+FET-PET model | Path to the tumour contour associated with the T1wCE image. | File path under `/input/...`. |
| `pet_image` | required for the T1wCE+FET-PET model | Path to the FET-PET image. | File path under `/input/...`. |
| `pet_contour` | required for the T1wCE+FET-PET model | Path to the tumour contour associated with the FET-PET image. | File path under `/input/...`. |
| `t1wce_intersection` | optional | T1wCE image to use with `intersection_contour`. Leave empty to reuse `t1wce`. | Empty, or a file path under `/input/...`. |
| `intersection_contour` | required for Overall Survival and for the intersection+clinical Acute Recurrence model | Path to the MR/PET intersection contour applied to the corresponding T1wCE image. | File path under `/input/...`. |
| `clinical` | required for the intersection+clinical Acute Recurrence model | Clinical variables used by the published model. | Object containing `Age`, `Sex` and `(Re-)current GBM localisation: Frontal`. |

</details>


The config describes one patient per run. You do not need to provide every imaging branch: the code computes whichever predictions are supported by the available inputs.
<br>



## Run the code

`model_source="zenodo"` downloads the model ZIP files automatically from Zenodo record [10.5281/zenodo.22641702](https://doi.org/10.5281/zenodo.22641702).

To use model files already stored locally, set `model_source` to `"local"` or `"mounted"` and set `models_path` to `/models`. Then point `MODELS_DIR` in the launcher to the host folder containing the model ZIP files or extracted model folders.

Edit `INPUT_DIR`, `OUTPUT_DIR`, `MODELS_DIR` and `CONFIG_FILE` at the top of the launcher:

On Windows:

```bat
cd MATTO-GBM_Models
run_inference_windows.bat
```

On Linux/macOS:

```bash
cd MATTO-GBM_Models
./run_inference_linux.sh
```


### Output

Results are written to:

```text
<OUTPUT_DIR>/results/<experiment_name>/<timestamp>/
├── config_input.json
├── config_run.json
└── predictions.csv
```

<details>
<summary><code>predictions.csv</code></summary>

| Output column | Description |
|---|---|
| `Patient` | Patient identifier copied from the input config. |
| `Model` | Prognostic model used: `Model 1 (OS + RF(T1wCE+PET-GTV))` or `Model 2 (OS + AR: RF(T1wCE-MR and PET))`. |
| `Acute Recurrence Probability` | Estimated probability of acute recurrence. Acute recurrence refers to recurrence within the first 90 days. |
| `Acute Recurrence` | Acute Recurrence classification: `Yes` = acute recurrence predicted; `No` = no acute recurrence predicted. |
| `Predicted OS Group` | Overall Survival group obtained from the intersection radiomic feature rule: `Long OS` or `Short OS`. This is a group classification, not an individual survival-time estimate. |
| `Stratification` | Combined outcome derived from Acute Recurrence and Overall Survival: `Good outcome` or `Poor outcome`. |

</details>
