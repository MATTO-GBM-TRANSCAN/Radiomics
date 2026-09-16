import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn import set_config

from matto_radiomics.utils import *
from stratification import stratify_responders

set_config(transform_output="pandas")

# Applies one of the paper's two published Zenodo Acute Recurrence (AR) models to
# new patient data. Which results come out depends on which model you load:
# `AcuteRecurrence-MR+PET-GTV` (MR+PET GTV contours) only predicts AR, since its
# radiomic features don't include the Overall Survival feature below.
# `AcuteRecurrence-MR-MRandPET+Clinical` (MR-and-PET intersection contour +
# clinical variables) also gets Overall Survival for free, since its radiomic
# features already include the one feature the OS threshold rule needs, and from
# there also the combined Good/Poor responder stratification -- computed
# automatically, no separate configuration needed. See MATTO-GBM_Models/README.md
# for the full explanation, including which feature files each model needs and how
# to get them.
#
# The AR side does not need a "features" list in the config: it reads the exact
# feature names directly from the model itself (feature_names_in_), the same list
# example_inference_config.json would otherwise need to repeat by hand.

# Feature and threshold reproducing the paper's published Overall Survival (OS)
# result. Source: paper section 3.2 ("Regarding OS, the feature showing the best
# prognostic performance was glrlm_RunLengthNonUniformityNormalized, extracted from
# the MR intersection PET on Wavelet High-pass filtered in all directions T1wCE"),
# and ModelsExplanation.docx.
#
# This is NOT a trained model: it is a single-feature threshold rule, where the
# threshold is the median value of the feature in the training set. The paper kept
# this univariate rule instead of a multivariate signature because it outperformed
# every multivariate OS signature tried (C-Index = 0.64 in 5-fold cross-validation,
# 0.65 in the external test set), even though statistically significant multivariate
# OS signatures also exist (see paper Table 3). That is why there is no model.pkl for
# OS: applying it is just comparing one feature value -- already present in the
# loaded radiomic features, when the loaded model's contour includes it -- against
# one number.
DEFAULT_OS_FEATURE = "wavelet-HHH_glrlm_RunLengthNonUniformityNormalized"
DEFAULT_OS_THRESHOLD = 0.405139505


def load_features(
    radiomic_features_path: Path,
    clinical_features_path: Optional[Path],
    id_column: str,
) -> pd.DataFrame:
    """
    Load the radiomic features for the loaded Zenodo model, joined with clinical
    features when the model needs them. Unlike `load_data`, this makes no
    assumption about which clinical variables are present (e.g. no "Methylated"
    column): the Zenodo models use their own, smaller clinical variable set.

    :param radiomic_features_path: Path to the radiomic features CSV matching the
        loaded model's contour(s) (MR+PET GTV, or MR-and-PET intersection).
    :type radiomic_features_path: pathlib.Path
    :param clinical_features_path: Path to the clinical features CSV, or None if
        the loaded model does not use clinical variables.
    :type clinical_features_path: pathlib.Path or None
    :param id_column: Name of the column identifying each patient.
    :type id_column: str
    :return: DataFrame indexed by patient id, radiomic and (if given) clinical
        features side by side, one row per patient present in every file given.
    :rtype: pandas.DataFrame
    """
    (features, _, _, _, _) = load_data(radiomic_features_path, id_column)
    if clinical_features_path is not None:
        clinical_features = pd.read_csv(clinical_features_path).set_index(id_column)
        features = features.join(clinical_features, how="inner")
    return features


def predict_acute_recurrence(model_dir: Path, features: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a Zenodo Acute Recurrence model (model.pkl + preprocessor.pkl) to new
    patients, without needing to know its feature list in advance.

    :param model_dir: Folder containing the downloaded `model.pkl` and
        `preprocessor.pkl` for this model.
    :type model_dir: pathlib.Path
    :param features: DataFrame of radiomic (and, if used, clinical) features, one
        row per patient, indexed by patient id.
    :type features: pandas.DataFrame
    :return: DataFrame indexed like `features`, with the predicted probability and
        label (1 = Acute Recurrence, 0 = no Acute Recurrence).
    :rtype: pandas.DataFrame
    """
    with open(model_dir / "model.pkl", "rb") as f:
        model_ = pickle.load(f)
    with open(model_dir / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    model_features = model_.estimator_.feature_names_in_
    features_transformed = preprocessor.transform(features[model_features])
    features_final = features_transformed[model_features]

    probabilities = model_.predict_proba(features_final)[:, 1]
    labels = (probabilities >= model_.best_threshold_).astype(int)

    return pd.DataFrame(
        {"Probability": probabilities, "Predicted Label": labels},
        index=features.index,
    )


def classify_overall_survival(
    features: pd.DataFrame,
    feature_name: str = DEFAULT_OS_FEATURE,
    threshold: float = DEFAULT_OS_THRESHOLD,
) -> pd.DataFrame:
    """
    Classify patients into "Long OS" / "Short OS" using the single-feature threshold
    rule published in the paper, instead of a trained model.

    :param features: DataFrame of radiomic features, indexed by patient id. Must
        contain `feature_name`, extracted from the T1wCE image using the MR
        intersection PET (MR-and-PET) contour, per the paper.
    :type features: pandas.DataFrame
    :param feature_name: Name of the radiomic feature to threshold.
    :type feature_name: str
    :param threshold: Value the feature is compared against (median of the training
        set in the paper).
    :type threshold: float
    :return: DataFrame indexed like `features`, with the raw feature value and the
        predicted OS group ("Long OS" if the feature is above the threshold, "Short
        OS" otherwise).
    :rtype: pandas.DataFrame
    """
    feature_values = features[feature_name]
    predicted_group = feature_values.apply(
        lambda value: "Long OS" if value > threshold else "Short OS"
    )

    return pd.DataFrame(
        {
            feature_name: feature_values,
            "Predicted OS Group": predicted_group,
        },
        index=features.index,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Apply one of the paper's published Zenodo Acute Recurrence models to "
            "new patients. Also reports Overall Survival and the combined "
            "Good/Poor responder stratification when the loaded model's radiomic "
            "features include the Overall Survival feature."
        )
    )
    parser.add_argument("config_file", help="Path to the JSON configuration file.")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    id_column = config["id_column"]
    experiment_name = config.get("experiment_name", "Paper_results")
    root_dir = Path(config["root_dir"])
    save_dir = root_dir / "results"
    save = config.get("save", True)

    ar_model_dir = root_dir / config["ar_model_dir"]
    radiomic_features_path = root_dir / config["radiomic_features"]
    clinical_features_path = (
        root_dir / clinical_features_filename
        if (clinical_features_filename := config.get("clinical_features"))
        else None
    )

    log_path = None
    if save:
        run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir / experiment_name / run_start_time
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "log.log"
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
    _logger = set_logger(
        name="matto_radiomics.reproduce_paper_results",
        level=logging.DEBUG,
        outpath=log_path,
    )

    features = load_features(radiomic_features_path, clinical_features_path, id_column)

    _logger.info("Predicting Acute Recurrence with %s", config["ar_model_dir"])
    ar_predictions = predict_acute_recurrence(ar_model_dir, features)
    _logger.info("Acute Recurrence predictions:\n%s", ar_predictions)
    if save:
        ar_predictions.to_csv(save_dir / "acute_recurrence_predictions.csv")

    do_os = DEFAULT_OS_FEATURE in features.columns
    if do_os:
        _logger.info(
            "Overall Survival feature found in the loaded radiomic features; "
            "classifying with the paper's published threshold"
        )
        os_predictions = classify_overall_survival(features)
        _logger.info("Overall Survival predictions:\n%s", os_predictions)
        if save:
            os_predictions.to_csv(save_dir / "overall_survival_predictions.csv")

        _logger.info("Combining both into the Good/Poor responder stratification")
        combined_predictions = stratify_responders(os_predictions, ar_predictions)
        _logger.info("Good/Poor responder stratification:\n%s", combined_predictions)
        if save:
            combined_predictions.to_csv(save_dir / "predictions.csv")
    else:
        _logger.info(
            "Overall Survival feature (%s) not found in the loaded radiomic "
            "features; skipping Overall Survival and the Good/Poor responder "
            "stratification.",
            DEFAULT_OS_FEATURE,
        )

    if save:
        _logger.info("Predictions saved to %s", save_dir)

