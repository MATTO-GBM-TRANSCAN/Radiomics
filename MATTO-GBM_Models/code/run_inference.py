"""Run MATTO-GBM pretrained model inference from a JSON config."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from matto_radiomics.utils import set_logger

from model_archives import (
    download_zenodo_zip_files,
    ensure_model_directory,
    install_archives_for_models,
)
import run_predictions


ZENODO_MODELS_REFERENCE = "10.5281/zenodo.22641702"
MRPET_MODEL_NAME = "AcuteRecurrence-MR+PET-GTV"
INTERSECTION_MODEL_NAME = "AcuteRecurrence-MR-MRandPET+Clinical"
MODEL_NAMES = (MRPET_MODEL_NAME, INTERSECTION_MODEL_NAME)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTION_PARAMETERS = (
    REPO_ROOT
    / "Radiomic_Feature_Extraction"
    / "input_templates"
    / "extraction"
    / "extraction_parameters.yaml"
)
EXTRACTION_INPUT = Path("/input")
EXTRACTION_OUTPUT = Path("/output")

PREDICTION_COLUMNS = [
    "Patient",
    "Model",
    "Acute Recurrence Probability",
    "Acute Recurrence",
    "Predicted OS Group",
    "Stratification",
]

MODEL_LABELS = {
    "ar_mrpet": "Model 1 (OS + RF(T1wCE+PET-GTV))",
    "ar_intersection": "Model 2 (OS + AR: RF(T1wCE-MR∩PET))",
}

STRATIFICATION_KEYS = {
    "ar_mrpet": "stratification_mrpet",
    "ar_intersection": "stratification_intersection",
}


def _run_python(args: list[str], logger: logging.Logger) -> None:
    logger.info("$ python %s", " ".join(args))
    process = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    if process.stdout:
        logger.info(process.stdout.rstrip())
    if process.stderr:
        logger.info(process.stderr.rstrip())
    if process.returncode != 0:
        raise RuntimeError(f"python {args[0]} failed with exit code {process.returncode}")


def _optional_path(config: dict, key: str) -> Path | None:
    value = config.get(key)
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{key} not found: {path}")
    return path


def _clean_extraction_output() -> None:
    EXTRACTION_OUTPUT.mkdir(parents=True, exist_ok=True)
    for pattern in ("radiomic_features_*.csv", "features.csv", "log_*.txt"):
        for path in EXTRACTION_OUTPUT.glob(pattern):
            if path.is_file():
                path.unlink()
    temp_dir = EXTRACTION_OUTPUT / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def _ensure_extraction_parameters(logger: logging.Logger) -> None:
    target = EXTRACTION_INPUT / "extraction_parameters.yaml"
    if target.is_file():
        return
    EXTRACTION_INPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXTRACTION_PARAMETERS, target)
    logger.info("Copied extraction parameters to %s", target)


def _extract_features(
    rows: list[dict[str, str]],
    run_name: str,
    data_dir: Path,
    logger: logging.Logger,
) -> Path:
    _clean_extraction_output()
    _ensure_extraction_parameters(logger)

    input_csv = data_dir / f"extraction_{run_name}.csv"
    fieldnames = ["Image", "Mask", "Patient"] + (["Contour"] if len(rows) > 1 else [])
    with input_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    logger.info("--- Extracting radiomic features (%s) ---", run_name)
    _run_python(
        [
            "-m",
            "matto_radiomics.feature_extraction.batch_parallel_extraction_pyradiomics",
            "--input_csv",
            str(input_csv),
        ],
        logger,
    )

    extracted = sorted(
        EXTRACTION_OUTPUT.glob("radiomic_features_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not extracted:
        raise RuntimeError(f"Feature extraction for {run_name} did not produce output.")

    logger.info("--- Preparing features (%s) ---", run_name)
    _run_python(
        [
            "-m",
            "matto_radiomics.data_preparation.prepare_features_and_endpoints",
            "--features_path",
            str(extracted[-1]),
        ],
        logger,
    )

    prepared = EXTRACTION_OUTPUT / "features.csv"
    if not prepared.is_file():
        raise RuntimeError(f"Feature preparation for {run_name} did not produce features.csv.")

    destination = data_dir / f"features_{run_name}.csv"
    shutil.copy2(prepared, destination)
    _clean_extraction_output()
    return destination


def _prepare_models(config: dict, work_dir: Path, logger: logging.Logger) -> Path:
    source = str(config.get("model_source", "zenodo")).lower()
    models_path = Path(config.get("models_path", "/models"))

    if source in {"mounted", "local"}:
        if not models_path.exists():
            raise FileNotFoundError(f"models_path not found: {models_path}")

        if all((models_path / model_name).is_dir() for model_name in MODEL_NAMES):
            logger.info("Using model directory: %s", models_path)
            return models_path

        archives = (
            [models_path]
            if models_path.is_file() and models_path.suffix.lower() == ".zip"
            else sorted(models_path.glob("*.zip"))
        )
        if not archives:
            raise ValueError(
                "models_path must be either a directory containing the model "
                "folders, a directory containing model ZIP files, or a model ZIP file."
            )

        models_root = work_dir / "models"
        installed = install_archives_for_models(archives, models_root, MODEL_NAMES)
        if not installed:
            raise RuntimeError("No compatible MATTO-GBM model could be prepared.")
        for model_name in installed:
            logger.info("Prepared model: %s", model_name)
        return models_root

    models_root = work_dir / "models"
    if source == "zenodo":
        logs: list[str] = []
        archives = download_zenodo_zip_files(
            config.get("zenodo_reference", ZENODO_MODELS_REFERENCE),
            work_dir / "zenodo_models",
            logs,
        )
        for line in logs:
            logger.info(line)
    else:
        raise ValueError("model_source must be 'zenodo', 'local', or 'mounted'.")

    installed = install_archives_for_models(archives, models_root, MODEL_NAMES)
    if not installed:
        raise RuntimeError("No compatible MATTO-GBM model could be prepared.")
    for model_name in installed:
        logger.info("Prepared model: %s", model_name)
    return models_root


def _clinical_values(config: dict) -> dict | None:
    clinical = config.get("clinical")
    if clinical in (None, False):
        return None

    required = ("Age", "Sex", "(Re-)current GBM localisation: Frontal")
    missing = [field for field in required if field not in clinical]
    if missing:
        raise ValueError("Missing clinical field(s): " + ", ".join(missing))

    age = int(clinical["Age"])
    sex = int(clinical["Sex"])
    frontal = int(clinical["(Re-)current GBM localisation: Frontal"])
    if not 0 <= age <= 120:
        raise ValueError("Age must be between 0 and 120.")
    if sex not in (0, 1) or frontal not in (0, 1):
        raise ValueError("Sex and frontal localisation must be 0 or 1.")
    return {
        "Age": age,
        "Sex": sex,
        "(Re-)current GBM localisation: Frontal": frontal,
    }


def _write_clinical_csv(values: dict, patient_id: str, destination: Path) -> None:
    row = {"Patient": patient_id, **values}
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _make_outputs_writable(path: Path) -> None:
    """Keep Docker-created output files easy to edit/delete from the host."""
    if not path.exists():
        return
    for current_root, dirs, files in os.walk(path):
        current_path = Path(current_root)
        try:
            current_path.chmod(0o777)
        except OSError:
            pass
        for dirname in dirs:
            try:
                (current_path / dirname).chmod(0o777)
            except OSError:
                pass
        for filename in files:
            try:
                (current_path / filename).chmod(0o666)
            except OSError:
                pass


def _format_prediction_rows(patient_id: str, results: dict) -> list[dict]:
    rows = []
    for ar_key in ("ar_mrpet", "ar_intersection"):
        if ar_key not in results:
            continue

        ar_predictions = results[ar_key]
        ar_row = (
            ar_predictions.loc[patient_id]
            if patient_id in ar_predictions.index
            else ar_predictions.iloc[0]
        )
        probability = float(ar_row["Probability"])
        acute_recurrence = "Yes" if int(float(ar_row["Predicted Label"])) == 1 else "No"

        os_group = ""
        stratification = ""
        strat_key = STRATIFICATION_KEYS[ar_key]
        if strat_key in results and not results[strat_key].empty:
            strat_predictions = results[strat_key]
            strat_row = (
                strat_predictions.loc[patient_id]
                if patient_id in strat_predictions.index
                else strat_predictions.iloc[0]
            )
            os_group = strat_row.get("Predicted OS Group", "")
            stratification = strat_row.get("Stratification", "")
            stratification = {
                "Good responder": "Good outcome",
                "Poor responder": "Poor outcome",
            }.get(stratification, stratification)

        rows.append(
            {
                "Patient": patient_id,
                "Model": MODEL_LABELS[ar_key],
                "Acute Recurrence Probability": f"{probability:.6f}",
                "Acute Recurrence": acute_recurrence,
                "Predicted OS Group": os_group,
                "Stratification": stratification,
            }
        )
    return rows


def run(config: dict, logger: logging.Logger, results_dir: Path) -> dict:
    patient_id = str(config.get("patient_id", "")).strip()
    if not patient_id:
        raise ValueError("patient_id is required.")

    results_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="matto_pro_cli_") as temp:
        work_dir = Path(temp)
        data_dir = work_dir / "data"
        data_dir.mkdir()

        t1wce = _optional_path(config, "t1wce")
        mr_contour = _optional_path(config, "mr_contour")
        pet_image = _optional_path(config, "pet_image")
        pet_contour = _optional_path(config, "pet_contour")
        t1wce_intersection = _optional_path(config, "t1wce_intersection") or t1wce
        intersection_contour = _optional_path(config, "intersection_contour")
        clinical = _clinical_values(config)

        do_mrpet = all((t1wce, mr_contour, pet_image, pet_contour))
        do_intersection = bool(t1wce_intersection and intersection_contour)
        if not do_mrpet and not do_intersection:
            raise ValueError(
                "Provide either t1wce + mr_contour + pet_image + pet_contour, "
                "or t1wce/t1wce_intersection + intersection_contour."
            )

        models_root = _prepare_models(config, work_dir, logger)
        prediction_config = {
            "id_column": "Patient",
            "root_dir": str(data_dir),
            "models_root": str(models_root),
            "mrpet_ar_model_dir": MRPET_MODEL_NAME,
            "intersection_ar_model_dir": INTERSECTION_MODEL_NAME,
            "mrpet_features": None,
            "intersection_features": None,
            "clinical_features": None,
            "save": False,
        }

        if do_mrpet:
            ensure_model_directory(models_root / MRPET_MODEL_NAME)
            features_path = _extract_features(
                [
                    {
                        "Image": str(t1wce),
                        "Mask": str(mr_contour),
                        "Patient": patient_id,
                        "Contour": "MR",
                    },
                    {
                        "Image": str(pet_image),
                        "Mask": str(pet_contour),
                        "Patient": patient_id,
                        "Contour": "PET",
                    },
                ],
                "mrpet",
                data_dir,
                logger,
            )
            prediction_config["mrpet_features"] = features_path.name

        if do_intersection:
            features_path = _extract_features(
                [
                    {
                        "Image": str(t1wce_intersection),
                        "Mask": str(intersection_contour),
                        "Patient": patient_id,
                    }
                ],
                "intersection",
                data_dir,
                logger,
            )
            prediction_config["intersection_features"] = features_path.name

        if clinical:
            clinical_path = data_dir / "clinical.csv"
            _write_clinical_csv(clinical, patient_id, clinical_path)
            prediction_config["clinical_features"] = clinical_path.name

        (results_dir / "config_input.json").write_text(
            json.dumps(config, indent=4),
            encoding="utf-8",
        )
        (results_dir / "config_run.json").write_text(
            json.dumps(prediction_config, indent=4),
            encoding="utf-8",
        )

        run_predictions._logger = logger
        results = run_predictions.run(prediction_config)

        computed = sorted(results.keys())
        prediction_rows = _format_prediction_rows(patient_id, results)
        with (results_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            writer.writerows(prediction_rows)

        logger.info("Computed result keys: %s", computed)
        logger.info("predictions.csv:\n%s", prediction_rows)
        _make_outputs_writable(results_dir.parents[1])
        logger.info("Results saved to %s", results_dir)
        return {
            "results_dir": str(results_dir),
            "computed": computed,
            "predictions": str(results_dir / "predictions.csv"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MATTO-GBM pretrained radiomics inference from a JSON config."
    )
    parser.add_argument("config_file", help="Path to inference_config.json.")
    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    output_dir = Path(config.get("output_dir", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = config.get("experiment_name", "MATTO_GBM_inference")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = output_dir / "results" / experiment_name / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = set_logger(
        name="matto_radiomics.pro_cli",
        level=logging.DEBUG,
        outpath=None,
    )
    try:
        print(json.dumps(run(config, logger, results_dir), indent=2))
    finally:
        _clean_extraction_output()
        _make_outputs_writable(output_dir / "results")


if __name__ == "__main__":
    main()
