from model_archives import ensure_model_directory

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from matto_radiomics.utils import *
from reproduce_paper_results import (
    classify_overall_survival,
    load_features,
    predict_acute_recurrence,
    DEFAULT_OS_FEATURE,
)
from stratification import stratify_responders

# Applies both published Zenodo Acute Recurrence (AR) models to whatever inputs
# the user actually provided, instead of requiring the full set the paper's own
# reproduce_paper_results.py expects. Each result is computed as soon as the
# inputs it needs are available, independently of the others:
#   - AR from `AcuteRecurrence-MR+PET-GTV`     needs the MR+PET GTV features.
#   - Overall Survival (OS)                    needs only the intersection
#                                               features (the threshold rule
#                                               reads a single feature from
#                                               them, see reproduce_paper_results.py).
#   - AR from `AcuteRecurrence-MR-MRandPET+Clinical` needs the intersection
#                                               features AND the clinical
#                                               features (the model was trained
#                                               on both together).
#   - Good/Poor responder is reported as up to two separate combinations, one
#     per AR result available, each paired with OS -- the paper's headline
#     combination (OS + AR from the MR+PET GTV model) and the secondary one
#     (OS + AR from the intersection+clinical model, see stratification.py for
#     why these are not interchangeable).
# See MATTO-GBM_Models/README.md for the full explanation.


def run(config: dict) -> dict:
    """
    Compute whichever of Acute Recurrence, Overall Survival and Good/Poor
    responder are possible from the inputs listed in `config`, skipping
    anything that needs a file not provided.

    :param config: Parsed run configuration, see MATTO-GBM_Models/README.md
        for the field reference. Any of `mrpet_features`, `intersection_features`
        and `clinical_features` may be `None`.
    :type config: dict
    :return: One DataFrame per result that could be computed, keyed by
        "ar_mrpet", "os", "ar_intersection", "stratification_mrpet" and
        "stratification_intersection". Keys for results that could not be
        computed are omitted.
    :rtype: dict
    """
    id_column = config["id_column"]
    root_dir = Path(config["root_dir"])
    models_root = Path(config["models_root"])

    results = {}

    mrpet_features_path = (
        root_dir / config["mrpet_features"] if config.get("mrpet_features") else None
    )
    intersection_features_path = (
        root_dir / config["intersection_features"]
        if config.get("intersection_features")
        else None
    )
    clinical_features_path = (
        root_dir / config["clinical_features"]
        if config.get("clinical_features")
        else None
    )

    ar_mrpet = None
    mrpet_model_dir = models_root / config["mrpet_ar_model_dir"]
    if mrpet_features_path is not None:
        ensure_model_directory(mrpet_model_dir)
    if mrpet_features_path is not None and not mrpet_model_dir.is_dir():
        _logger.info(
            "%s not found under %s; skipping that Acute Recurrence",
            config["mrpet_ar_model_dir"],
            models_root,
        )
    elif mrpet_features_path is not None:
        _logger.info("MR+PET GTV features found; predicting Acute Recurrence")
        mrpet_features = load_features(mrpet_features_path, None, id_column)
        ar_mrpet = predict_acute_recurrence(mrpet_model_dir, mrpet_features)
        results["ar_mrpet"] = ar_mrpet
    else:
        _logger.info("No MR+PET GTV features given; skipping that Acute Recurrence")

    os_predictions = None
    intersection_features = None
    if intersection_features_path is not None:
        intersection_features = load_features(
            intersection_features_path, None, id_column
        )
        if DEFAULT_OS_FEATURE in intersection_features.columns:
            _logger.info("Intersection features found; classifying Overall Survival")
            os_predictions = classify_overall_survival(intersection_features)
            results["os"] = os_predictions
        else:
            _logger.info(
                "Overall Survival feature (%s) not found in the intersection "
                "features; skipping Overall Survival.",
                DEFAULT_OS_FEATURE,
            )
    else:
        _logger.info("No intersection features given; skipping Overall Survival")

    ar_intersection = None
    intersection_model_dir = models_root / config["intersection_ar_model_dir"]
    has_intersection_inputs = (
        intersection_features is not None and clinical_features_path is not None
    )
    if has_intersection_inputs:
        ensure_model_directory(intersection_model_dir)
    if has_intersection_inputs and not intersection_model_dir.is_dir():
        _logger.info(
            "%s not found under %s; skipping Acute Recurrence from the "
            "intersection model",
            config["intersection_ar_model_dir"],
            models_root,
        )
    elif has_intersection_inputs:
        _logger.info(
            "Intersection features and clinical features found; predicting "
            "Acute Recurrence"
        )
        clinical_features = pd.read_csv(clinical_features_path).set_index(id_column)
        joined_features = intersection_features.join(clinical_features, how="inner")
        ar_intersection = predict_acute_recurrence(intersection_model_dir, joined_features)
        results["ar_intersection"] = ar_intersection
    elif intersection_features is not None:
        _logger.info(
            "No clinical features given; skipping Acute Recurrence from the "
            "intersection model"
        )

    if os_predictions is not None and ar_mrpet is not None:
        _logger.info(
            "Combining Overall Survival with Acute Recurrence (MR+PET GTV model)"
        )
        results["stratification_mrpet"] = stratify_responders(
            os_predictions, ar_mrpet
        )
    if os_predictions is not None and ar_intersection is not None:
        _logger.info(
            "Combining Overall Survival with Acute Recurrence (intersection+"
            "clinical model)"
        )
        results["stratification_intersection"] = stratify_responders(
            os_predictions, ar_intersection
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Apply both published Zenodo Acute Recurrence models, Overall "
            "Survival and Good/Poor responder stratification to whichever "
            "inputs are available."
        )
    )
    parser.add_argument("config_file", help="Path to the JSON configuration file.")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    experiment_name = config.get("experiment_name", "PRO_results")
    root_dir = Path(config["root_dir"])
    save_dir = root_dir / "results"
    save = config.get("save", True)

    log_path = None
    if save:
        run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir / experiment_name / run_start_time
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "log.log"
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
    _logger = set_logger(
        name="matto_radiomics.pro",
        level=logging.DEBUG,
        outpath=log_path,
    )

    results = run(config)

    filenames = {
        "ar_mrpet": "acute_recurrence_mrpet_predictions.csv",
        "os": "overall_survival_predictions.csv",
        "ar_intersection": "acute_recurrence_intersection_predictions.csv",
        "stratification_mrpet": "stratification_mrpet.csv",
        "stratification_intersection": "stratification_intersection.csv",
    }
    for key, dataframe in results.items():
        _logger.info("%s:\n%s", filenames[key], dataframe)
        if save:
            dataframe.to_csv(save_dir / filenames[key])

    if save:
        with open(save_dir / "computed.json", "w") as f:
            json.dump(sorted(results.keys()), f, indent=4)
        _logger.info("Results saved to %s", save_dir)

