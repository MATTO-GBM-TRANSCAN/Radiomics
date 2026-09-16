# PET|MR Radiomics Models for Treatment Outcome Prediction in recurrent GBM

This repository provides a script-based pipeline to develop and test PET|MR radiomics models to predict treatment outcomes in recurrent Glioblastoma (GBM). 

## Contents Overview

The repository includes three different sections:

1. **[`Radiomic_Feature_Extraction/`](Radiomic_Feature_Extraction)**: extraction of radiomic features from PET|MR images and tumour contours using [PyRadiomics](https://pyradiomics.readthedocs.io/). 
2. **[`Radiomics_Model_Development/`](Radiomics_Model_Development)**: the complete workflow for developing radiomic models.
3. **[`MATTO-GBM Models/`](MATTO-GBM_Models)**: to apply the PET|MR radiomics models to your own cohort of patients with recurrent GBM. Treatment outcomes include: Acute Recurrence (AR), Overall Survival (OS) and Patient with Poor Treatment Outcome (defined as AR &/ short-OS).


Start here and follow the workflow that matches your purpose:

|Aim | Workflow Steps |
|---|---|
| Radiomic Features Extraction | [`Requirements`](#requirements) → [`Radiomic_Feature_Extraction/README.md`](Radiomic_Feature_Extraction/README.md) |
| Radiomics Model Development | [`Requirements`](#requirements) → [`Radiomic_Feature_Extraction/README.md`](Radiomic_Feature_Extraction/README.md) → [`Radiomics_Model_Development/README.md`](Radiomics_Model_Development/README.md) |
| MATTO-GBM Models| [`Requirements`](#requirements) → [`MATTO-GBM_Models/README.md`](MATTO-GBM_Models/README.md) |

## Requirements

Before proceeding with any of the workflows above, make sure the following prerequisites are available on your machine: **Git** and **Docker**

- **Git**, to clone this repository.

Clone the repository and navigate to its root directory. Unless stated otherwise, all commands throughout the repository’s README files are expected to be run from the Radiomics/ directory.

```bash
git clone https://github.com/MATTO-GBM-TRANSCAN/Radiomics.git
cd Radiomics
```

- **Docker**: to run the workflows in the containerized environment provided by the `Dockerfile` at the root of this repository.

All dependencies and software required to run the three workflows are included in a single Docker image. Build the image once from the repository root; it can then be used to run all three workflows.

```bash
docker build -t matto-radiomics .
```

Each workflow directory includes ready-made run_*.sh scripts to launch the corresponding workflow (e.g. Radiomics_Model_Development/run_train_validation_event.sh). Open the appropriate script, update the local paths defined at the top, and run it. The docker run command is already configured and does not need to be modified. Each directory’s README.md specifies which script to use and explains the purpose of each path.

Windows users can adapt the corresponding run_*.sh script into a run_*.ps1 script.



## Repository structure

```
Radiomics/
├── 0_Common/                       helpers
├── 0_Packaging/                    integrates the three workflows described above into a single installable package
├── Radiomic_Feature_Extraction/        workflow 1
├── Radiomics_Model_Development/        workflow 2
├── MATTO-GBM_Models/                   workflow 3
├── Dockerfile                      defines the Docker environment and dependencies shared across all three workflows.
├── CITATION.cff
├── LICENSE  
└── README.md                  
```                  


## Citation

Please cite the following paper if you use this work for your research:

Carles M, Mora-Rubio A, Fechter T, Perez-Herrero S, Fernández-Patón M, Baltas D, Mix M, Meyer P, Popp I, Martí-Bonmatí L, Grosu AL. PET/MR open source radiomics models for treatment outcome prediction in recurrent GBM. Cancers.18(18):3008; https://doi.org/10.3390/cancers18183008.

or

```bibtex
@article{carlesPETMROpenSource2026,
  title = {{PET}/{MR} Open Source Radiomics Models for Treatment Outcome Prediction in Recurrent {GBM}},
  author = {Carles, M. and Mora-Rubio, A. and Fechter, T. and Perez-Herrero, S. and Fernández-Patón, M. and Baltas, D. and Mix, M. and Meyer, P. and Popp, I. and Martí-Bonmatí, L. and Grosu, A. L.},
  journaltitle = {Cancers},
  year = {2026},
  volume = {18},
  number = {18},
  article-number = {3008},
  note = {(https://doi.org/10.3390/cancers18183008)}
}
```


## Funding

This work was supported by the MATTO-GBM Project under the European TRANSCAN-3 ERA-NET 2022 Program, funded by ISCIII (AC23_1/00012), FAECC (TRANSCAN2022-784-104), and the European Union through the Next Generation EU Funds.

## History

- Version 1.0: September 2026

## License

Released under the [MIT License](LICENSE).

