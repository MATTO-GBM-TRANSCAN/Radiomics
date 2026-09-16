import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_val_score,
)

from matto_radiomics.feature_selection import get_feature_preprocessing_pipeline
from matto_radiomics.models import CoxPH, KaplanMeier
from matto_radiomics.utils import *

set_config(transform_output="pandas")
_logger = logging.getLogger(__name__)


def event_cross_validation(
    model: LogisticRegression,
    scoring_method: str,
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    splitter: BaseCrossValidator,
):
    """
    Perform cross-validation for event analysis.

    :param model: Logistic Regression model.
    :type model: sklearn.linear_model.LogisticRegression
    :param scoring_method: Scoring method to evaluate the model.
    :type scoring_method: str
    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param splitter: Cross-validator for splitting the data.
    :type splitter: sklearn.model_selection.BaseCrossValidator
    :return: Dictionary containing scores, mean, and standard deviation.
    :rtype: dict
    """
    _logger.info(
        "Evaluating %s on feature(s): %s", scoring_method, list(features.columns)
    )
    if scoring_method == "f1":
        scorer = make_scorer(f1_score, pos_label=1)
    elif scoring_method == "roc_auc":
        scorer = make_scorer(roc_auc_score, response_method="predict_proba")
    elif scoring_method == "balanced_accuracy":
        scorer = make_scorer(balanced_accuracy_score)
    else:
        _logger.error("Scoring method %s is not supported.", scoring_method)
        return {"scores": [], "mean": np.inf, "std": np.inf}
    scores = cross_val_score(
        model,
        features,
        endpoints,
        cv=splitter,
        scoring=scorer,
    )
    return {
        "scores": scores.tolist(),
        "mean": np.mean(scores),
        "std": np.std(scores),
    }


def time_to_event_cross_validation(
    model: CoxPH,
    scoring_method: str,
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    splitter: BaseCrossValidator,
):
    """
    Perform cross-validation for time-to-event analysis.

    :param model: Cox proportional hazards model.
    :type model: CoxPH
    :param scoring_method: Scoring method to evaluate the model.
    :type scoring_method: str
    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param splitter: Cross-validator for splitting the data.
    :type splitter: sklearn.model_selection.BaseCrossValidator
    :return: Dictionary containing scores, mean, and standard deviation.
    :rtype: dict
    """
    _logger.info(
        "Evaluating %s on feature(s): %s", scoring_method, list(features.columns)
    )
    model.scoring_method = scoring_method
    scores = cross_val_score(
        model,
        features,
        endpoints,
        cv=splitter,
    )
    return {
        "scores": scores.tolist(),
        "mean": np.mean(scores),
        "std": np.std(scores),
    }


def event_train_test_validation(
    model: CoxPH,
    scoring_method: Union[str, Tuple[str, ...]],
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    test_features: pd.DataFrame,
    test_endpoints: pd.DataFrame,
    save_dir: Path,
    save: bool,
):
    """
    Perform train-test validation for event analysis.

    :param model: Logistic regression model.
    :type model: sklearn.linear_model.LogisticRegression
    :param scoring_method: Scoring method to evaluate the model.
    :type scoring_method: str
    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param test_features: DataFrame containing test feature data.
    :type test_features: pandas.DataFrame
    :param test_endpoints: DataFrame containing test endpoint data.
    :type test_endpoints: pandas.DataFrame
    :param save_dir: Directory to save the results.
    :type save_dir: pathlib.Path
    :param save: Flag to save the results or not.
    :type save: bool
    :return: Dictionary containing validation results.
    :rtype: dict
    """
    model_ = TunedThresholdClassifierCV(model, scoring=scoring_method[0])
    _logger.info(f"Feature subset: {list(features.columns)}")
    model_.fit(features, endpoints)

    if save:
        with open(save_dir / "model.pkl", "wb") as f:
            pickle.dump(model_, f, protocol=pickle.HIGHEST_PROTOCOL)

    results = {}
    test_predictions_proba = model_.predict_proba(test_features)[:, 1]
    test_predictions = (test_predictions_proba >= model_.best_threshold_).astype(int)

    if save:
        predictions_df = pd.DataFrame(
            {
                "Probability": test_predictions_proba,
                "Predicted Label": test_predictions,
            },
            index=test_features.index,
        )
        predictions_df.to_csv(save_dir / "test_predictions.csv")

    results["best_threshold"] = model_.best_threshold_
    for sm in scoring_method:
        if sm == "f1":
            score = f1_score(test_endpoints, test_predictions, pos_label=1)
        elif sm == "roc_auc":
            score = roc_auc_score(test_endpoints, test_predictions_proba)
        elif sm == "balanced_accuracy":
            score = balanced_accuracy_score(test_endpoints, test_predictions)
        else:
            _logger.error("Scoring method %s is not supported.", sm)
            continue
        results[sm] = score
    train_predictions_proba = model_.predict_proba(features)[:, 1]
    train_predictions = (train_predictions_proba >= model_.best_threshold_).astype(int)

    if save:
        predictions_df = pd.DataFrame(
            {
                "Probability": train_predictions_proba,
                "Predicted Label": train_predictions,
            },
            index=features.index,
        )
        predictions_df.to_csv(save_dir / "train_predictions.csv")

    plot_confusion_matrix(
        endpoints,
        train_predictions,
        endpoints.name,
        save_dir,
        "train_confusion_matrix",
        save,
    )
    plot_confusion_matrix(
        test_endpoints,
        test_predictions,
        endpoints.name,
        save_dir,
        "test_confusion_matrix",
        save,
    )
    plot_predictions_over_sigmoid(
        model_.estimator_,
        features,
        endpoints,
        model_.best_threshold_,
        save_dir,
        "train",
        save,
    )
    plot_predictions_over_sigmoid(
        model_.estimator_,
        test_features,
        test_endpoints,
        model_.best_threshold_,
        save_dir,
        "test",
        save,
    )
    plot_lr_coefficients(model_.estimator_, save_dir, save)
    return results


def time_to_event_train_test_validation(
    model: CoxPH,
    scoring_method: Union[str, Tuple[str, ...]],
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    test_features: pd.DataFrame,
    test_endpoints: pd.DataFrame,
    duration_column: str,
    event_column: str,
    save_dir: Path,
    save: bool,
):
    """
    Perform train-test validation for time-to-event analysis.

    :param model: Cox proportional hazards model.
    :type model: CoxPH
    :param scoring_method: Scoring method to evaluate the model.
    :type scoring_method: str
    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param test_features: DataFrame containing test feature data.
    :type test_features: pandas.DataFrame
    :param test_endpoints: DataFrame containing test endpoint data.
    :type test_endpoints: pandas.DataFrame
    :param duration_column: Column name for duration.
    :type duration_column: str
    :param event_column: Column name for event.
    :type event_column: str
    :param save_dir: Directory to save the results.
    :type save_dir: pathlib.Path
    :param save: Flag to save the results or not.
    :type save: bool
    :return: Dictionary containing validation results.
    :rtype: dict
    """
    _logger.info(f"Feature subset: {list(features.columns)}")
    model.fit(features, endpoints)

    if save:
        with open(save_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    results = {}
    for sm in scoring_method:
        model.scoring_method = sm
        results[sm] = model.score(test_features, test_endpoints)
    plot_coxph_coefficients(model.estimator, save_dir, save)
    train_log_partial_hazard = model.estimator.predict_log_partial_hazard(features)
    train_log_partial_hazard.name = "log Partial Hazard"
    train_survival_times = model.predict(features)


    if save:
        predictions_df = pd.DataFrame(
            {
                "log Partial Hazard": train_log_partial_hazard,
                "Survival Times [days]": train_survival_times,
            }
        )
        predictions_df.to_csv(save_dir / "train_predictions.csv", index_label="Patient")

    KaplanMeier(
        train_log_partial_hazard,
        endpoints,
        duration_column,
        event_column,
        None,
        save_dir,
        "train_log_partial_hazard",
        save,
    )
    test_log_partial_hazard = model.estimator.predict_log_partial_hazard(test_features)
    test_log_partial_hazard.name = "log Partial Hazard"
    test_survival_times = model.predict(test_features)

    if save:
        predictions_df = pd.DataFrame(
            {
                "log Partial Hazard": test_log_partial_hazard,
                "Survival Times [days]": test_survival_times,
            }
        )
        predictions_df.to_csv(save_dir / "test_predictions.csv", index_label="Patient")

    KaplanMeier(
        test_log_partial_hazard,
        test_endpoints,
        duration_column,
        event_column,
        None,
        save_dir,
        "test_log_partial_hazard",
        save,
    )
    KaplanMeier(
        test_log_partial_hazard,
        test_endpoints,
        duration_column,
        event_column,
        train_log_partial_hazard.median(),
        save_dir,
        "test_log_partial_hazard_w_train_median",
        save,
    )
    return results


def main(
    root_dir: Path,
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    id_column: str,
    feature_set: Union[str, List[str]],
    binary_feature_names: pd.Index,
    duration_column: str,
    event_column: str,
    evaluation_type: str,
    k_folds: int,
    scoring_method: tuple[str],
    save_dir: Path,
    save: bool,
    random_state: int = 42,
    config: dict = {},
):
    """
    Main function to execute the evaluation pipeline.

    :param root_dir: Root directory for the project.
    :type root_dir: pathlib.Path
    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param id_column: Column name for the ID.
    :type id_column: str
    :param feature_set: Set of features to use for evaluation.
    :type feature_set: str | list[str]
    :param binary_feature_names: Index of binary feature names.
    :type binary_feature_names: pandas.Index
    :param duration_column: Column name for duration.
    :type duration_column: str
    :param event_column: Column name for event.
    :type event_column: str
    :param evaluation_type: Type of evaluation to perform.
    :type evaluation_type: str
    :param k_folds: Number of folds for cross-validation.
    :type k_folds: int
    :param scoring_method: Tuple of scoring methods to evaluate the model.
    :type scoring_method: tuple[str]
    :param save_dir: Directory to save the results.
    :type save_dir: pathlib.Path
    :param save: Flag to save the results or not.
    :type save: bool
    :param random_state: Random state for reproducibility.
    :type random_state: int
    :param config: Additional configuration options.
    :type config: dict
    :return: None
    """

    if len(feature_set) < 1:
        _logger.error(
            "Not enough features for evaluation, at least 1 feature is required."
        )
        return {}

    features = features[feature_set]
    preprocessor = get_feature_preprocessing_pipeline(
        features.columns, binary_feature_names.intersection(features.columns)
    )
    features = preprocessor.fit_transform(features)
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)
    if evaluation_type == "Cross Validation":
        _logger.info("Running cross-validation")
        if duration_column == "":
            model = LogisticRegression()
            splitter = StratifiedKFold(
                n_splits=k_folds, shuffle=True, random_state=random_state
            )
            results = {}
            results[scoring_method[0]] = event_cross_validation(
                model, scoring_method[0], features, endpoints, splitter
            )
            for s_method in scoring_method[1:]:
                results[s_method] = event_cross_validation(
                    model,
                    s_method,
                    features,
                    endpoints[event_column],
                    splitter,
                )
        else:
            model = CoxPH(
                duration_column=duration_column,
                event_column=event_column,
                scoring_method=scoring_method[0],
            )
            splitter = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
            results = {}
            results[scoring_method[0]] = time_to_event_cross_validation(
                model, scoring_method[0], features, endpoints, splitter
            )
            for s_method in scoring_method[1:]:
                results[s_method] = time_to_event_cross_validation(
                    model,
                    s_method,
                    features,
                    endpoints,
                    splitter,
                )
    elif evaluation_type == "Train-Test":
        _logger.info("Running Train-Test evaluation")
        test_radiomic_features_path = root_dir / config["test_radiomic_features"]
        test_clinical_features_path = (
            root_dir / test_clinical_features_filename
            if (test_clinical_features_filename := config.get("test_clinical_features"))
            else None
        )
        test_endpoints_path = root_dir / config["test_endpoints"]
        (test_features, test_endpoints, _, _, _) = load_data(
            test_radiomic_features_path,
            id_column,
            test_clinical_features_path,
            test_endpoints_path,
        )
        test_features = test_features[feature_set]
        test_features = preprocessor.transform(test_features)
        if duration_column == "":
            test_endpoints = test_endpoints.loc[:, event_column]
            test_endpoints = test_endpoints.dropna()
            test_features = test_features.loc[
                test_features.index.intersection(test_endpoints.index, sort=None)
            ]
            _logger.info("Test %s shape: %s", event_column, test_endpoints.shape)
            _logger.info("Test Features shape: %s", test_features.shape)
            model = LogisticRegression()
            results = event_train_test_validation(
                model,
                scoring_method,
                features,
                (
                    endpoints[event_column]
                    if isinstance(endpoints, pd.DataFrame)
                    else endpoints
                ),
                test_features,
                test_endpoints,
                save_dir=save_dir,
                save=save,
            )
        else:
            model = CoxPH(
                duration_column=duration_column,
                event_column=event_column,
                scoring_method=scoring_method[0],
            )
            results = time_to_event_train_test_validation(
                model,
                scoring_method,
                features,
                endpoints,
                test_features,
                test_endpoints,
                duration_column,
                event_column,
                save_dir=save_dir,
                save=save,
            )
    if save:
        with open(save_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=4)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation of a specific feature subset."
    )
    parser.add_argument(
        "config_file",
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()

    # Set up
    with open(args.config_file, "r") as f:
        config = json.load(f)
    id_column = config["id_column"]
    duration_column = config.get("duration_column", "")
    event_column = config["event_column"]
    default_experiment_name = (
        duration_column if (duration_column != "") else event_column
    )
    experiment_name = config.get("experiment_name", default_experiment_name)
    k_folds = config.get("k_folds", 5)
    scoring_method = config["scoring_method"]
    root_dir = Path(config["root_dir"])
    radiomic_features_path = root_dir / config["radiomic_features"]
    clinical_features_path = (
        root_dir / clinical_features_filename
        if (clinical_features_filename := config.get("clinical_features"))
        else None
    )
    feature_set = config["features"]
    evaluation_type = config["evaluation_type"]
    endpoints_path = root_dir / config["endpoints"]
    save_dir = root_dir / "results"
    save = config.get("save", True)
    random_state = config.get("random_state", 42)

    log_path = None
    if save:
        run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir / experiment_name / run_start_time
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "log.log"
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
    _logger = set_logger(
        name="matto_radiomics.evaluate", level=logging.DEBUG, outpath=log_path
    )
    _logger.info("Starting %s analysis", default_experiment_name)

    # Load data
    (features, endpoints, _, _, binary_feature_names) = load_data(
        radiomic_features_path,
        id_column,
        clinical_features_path,
        endpoints_path,
    )
    _ = main(
        root_dir,
        features,
        endpoints,
        id_column,
        feature_set,
        binary_feature_names,
        duration_column,
        event_column,
        evaluation_type,
        k_folds,
        scoring_method,
        save_dir,
        save,
        random_state,
        config,
    )
