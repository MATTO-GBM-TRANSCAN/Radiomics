import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from typing import Tuple, List, Dict, Union

import pandas as pd
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .evaluate_feature_subset import event_cross_validation
from .evaluate_feature_subset import main as test_set_evaluation
from .explore_feature_subsets import parallel_evaluate_feature_subsets
from matto_radiomics.feature_selection import get_full_feature_pipeline
from matto_radiomics.feature_selection.statistics import (
    compute_mannwhitney,
    compute_pearson,
)
from matto_radiomics.utils import *
from multiprocessing import cpu_count

set_config(transform_output="pandas")


def multivariate_modelling(
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    scoring_method: str,
    scoring_method_ascending: bool,
    additional_metrics: tuple[str] = ("roc_auc",),
    k_folds: int = 5,
    n_processes: int = 12,
    random_state: int = 5,
    max_subset_size: int = 8,
) -> Tuple[List[str], Dict[str, Union[List[str], float]]]:
    """
    Perform multivariate modeling for event analysis.

    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param scoring_method: Primary scoring method for evaluation.
    :type scoring_method: str
    :param scoring_method_ascending: Whether higher scores are better.
    :type scoring_method_ascending: bool
    :param additional_metrics: Additional metrics for evaluation.
    :type additional_metrics: tuple[str]
    :param k_folds: Number of cross-validation folds.
    :type k_folds: int
    :param n_processes: Number of processes for parallel evaluation.
    :type n_processes: int
    :param random_state: Random state for reproducibility.
    :type random_state: int
    :return: Tuple containing the best feature subset and results.
    :rtype: tuple[list[str], dict[str, list[str] | float]]
    """
    _multivariate_logger = _logger.getChild("multivariate_modelling")

    if len(features.columns) < 2:
        _multivariate_logger.error(
            "Not enough features for multivariate modelling, at least 2 features are required. Skipping this stage."
        )
        return {}

    model = LogisticRegression()
    splitter = StratifiedKFold(
        n_splits=k_folds, shuffle=True, random_state=random_state
    )
    best_feature_subset, exploration_results = parallel_evaluate_feature_subsets(
        model,
        features,
        endpoints,
        splitter,
        scoring_method,
        scoring_method_ascending,
        subset_size_range=(
            2,
            (
                len(features.columns) + 1
                if len(features.columns) < max_subset_size
                else max_subset_size + 1
            ),
        ),  # Evaluate subsets of size 2 to max(7, len(features.columns)-1)
        n_processes=n_processes,
    )
    if save:
        (save_dir / "validation_multivariate").mkdir()
        filename = save_dir / "validation_multivariate" / "validation_multivariate.json"
        with open(filename, "w") as f:
            json.dump(exploration_results, f, indent=4)

    results = {}
    results["feature_subset"] = best_feature_subset
    results[scoring_method] = event_cross_validation(
        model, scoring_method, features[best_feature_subset], endpoints, splitter
    )
    for s_method in additional_metrics:
        results[s_method] = event_cross_validation(
            model,
            s_method,
            features[best_feature_subset],
            endpoints,
            splitter,
        )
    return results


def univariate_modelling(
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    scoring_method: str,
    scoring_method_ascending: bool,
    additional_metrics: tuple[str] = ("roc_auc",),
    k_folds: int = 5,
    n_processes: int = 12,
    random_state: int = 5,
) -> Tuple[List[str], Dict[str, Union[List[str], float]]]:
    """
    Perform univariate modeling for event analysis.

    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param scoring_method: Primary scoring method for evaluation.
    :type scoring_method: str
    :param scoring_method_ascending: Whether higher scores are better.
    :type scoring_method_ascending: bool
    :param additional_metrics: Additional metrics for evaluation.
    :type additional_metrics: tuple[str]
    :param k_folds: Number of cross-validation folds.
    :type k_folds: int
    :param n_processes: Number of processes for parallel evaluation.
    :type n_processes: int
    :param random_state: Random state for reproducibility.
    :type random_state: int
    :return: Tuple containing the best feature and results.
    :rtype: tuple[list[str], dict[str, list[str] | float]]
    """
    _univariate_logger = _logger.getChild("univariate_modelling")

    if len(features.columns) < 1:
        _univariate_logger.error(
            "Not enough features for univariate modelling, at least 1 feature is required. Skipping this stage."
        )
        return {}

    model = LogisticRegression()
    splitter = StratifiedKFold(
        n_splits=k_folds, shuffle=True, random_state=random_state
    )
    best_feature, _ = parallel_evaluate_feature_subsets(
        model,
        features,
        endpoints,
        splitter,
        scoring_method,
        scoring_method_ascending,
        subset_size_range=(1, 2),  # Evaluate each feature individually
        n_processes=n_processes,
    )

    results = {}
    results["feature"] = best_feature
    results[scoring_method] = event_cross_validation(
        model, scoring_method, features[best_feature], endpoints, splitter
    )
    for s_method in additional_metrics:
        results[s_method] = event_cross_validation(
            model,
            s_method,
            features[best_feature],
            endpoints,
            splitter,
        )
    return results


def run_event_analysis(config: dict):
    """Main entry point for external programs to run event analysis.

    :param config: Configuration dictionary with same structure as JSON config file
    """
    global _logger, save, save_dir

    # Extract parameters from config
    id_column = config["id_column"]
    event_column = config["event_column"]
    experiment_name = config.get("experiment_name", event_column)
    k_folds = config["k_folds"]
    scoring_method = config["scoring_method"]
    scoring_method_ascending = config["scoring_method_ascending"]
    root_dir = Path(config["root_dir"])
    radiomic_features_path = root_dir / config["radiomic_features"]
    clinical_features_path = (
        root_dir / clinical_features_filename
        if (clinical_features_filename := config.get("clinical_features"))
        else None
    )
    endpoints_path = root_dir / config["endpoints"]
    save_dir = root_dir / "results"
    save = config.get("save", True)
    if "n_processes" in config.keys():
        n_processes = config["n_processes"]
    else:
        n_processes = cpu_count()
        n_processes = n_processes * 2 + 1
    if "max_subset_size" in config.keys():
        max_subset_size = config["max_subset_size"]
    else:
        max_subset_size = 8

    log_path = None
    if save:
        run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir / experiment_name / run_start_time
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "log.log"
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
    _logger = set_logger(
        name="matto_radiomics.event", level=logging.DEBUG, outpath=log_path
    )
    _logger.info("Starting %s analysis", event_column)

    # Load data
    (
        features,
        endpoints,
        radiomic_feature_names,
        clinical_feature_names,
        binary_feature_names,
    ) = load_data(
        radiomic_features_path,
        id_column,
        clinical_features_path,
        endpoints_path,
    )
    endpoints = endpoints.loc[:, event_column]
    endpoints = endpoints.dropna()
    features = features.loc[features.index.intersection(endpoints.index, sort=None)]
    _logger.info("%s shape: %s", event_column, endpoints.shape)
    _logger.info("Features shape: %s", features.shape)

    # Compute statistics
    feature_correlation_matrix = compute_pearson(features)
    mw_with_target = compute_mannwhitney(features, endpoints)
    if save:
        (save_dir / "clinical_impact").mkdir()
        filename = save_dir / "clinical_impact" / "Mann-WhitneyU.xlsx"
        mode = "w" if not filename.exists() else "a"
        with pd.ExcelWriter(filename, mode=mode) as writer:
            mw_with_target.to_excel(writer, sheet_name="Mann-Whitney U")

    # Preprocess features
    pipeline = get_full_feature_pipeline(
        feature_correlation_matrix,
        mw_with_target,
        features.columns,
        radiomic_feature_names,
        clinical_feature_names,
        binary_feature_names,
        importance_to_target_method="lower statistic",
    )
    transformed_features = pipeline.fit_transform(
        features,
        endpoints,
    )

    # Initialize results storage
    results = {}

    # Univariate modelling
    _logger.info("Starting univariate modelling")
    univariate_results = univariate_modelling(
        transformed_features,
        endpoints,
        scoring_method=scoring_method[0],
        scoring_method_ascending=scoring_method_ascending,
        additional_metrics=scoring_method[1:],
        k_folds=k_folds,
        n_processes=n_processes,
    )
    results["univariate_results"] = univariate_results

    # Multivariate modelling
    _logger.info("Starting multivariate modelling")
    multivariate_results = multivariate_modelling(
        transformed_features,
        endpoints,
        scoring_method=scoring_method[0],
        scoring_method_ascending=scoring_method_ascending,
        additional_metrics=scoring_method[1:],
        k_folds=k_folds,
        n_processes=n_processes,
        max_subset_size=max_subset_size,
    )
    results["multivariate_results"] = multivariate_results

    # Test set evaluation
    if (config.get("test_radiomic_features") is not None) and (
        config.get("test_endpoints") is not None
    ):
        _logger.info("Starting univariate test set evaluation")
        univariate_test_results = test_set_evaluation(
            root_dir,
            features,
            endpoints,
            id_column,
            univariate_results.get("feature", []),
            binary_feature_names,
            "",
            event_column,
            "Train-Test",
            k_folds,
            scoring_method,
            save_dir / "test_univariate",
            save,
            config=config,
        )
        results["univariate_test_results"] = univariate_test_results

        _logger.info("Starting multivariate test set evaluation")
        multivariate_test_results = test_set_evaluation(
            root_dir,
            features,
            endpoints,
            id_column,
            multivariate_results.get("feature_subset", []),
            binary_feature_names,
            "",
            event_column,
            "Train-Test",
            k_folds,
            scoring_method,
            save_dir / "test_multivariate",
            save,
            config=config,
        )
        results["multivariate_test_results"] = multivariate_test_results

    if save:
        _logger.info("Saved results to %s", save_dir / "results.json")
        with open(save_dir / "results.json", "w") as f:
            json.dump(results, f, indent=4)

    _logger.info("Ended %s analysis", event_column)
    return results


def main():
    """Command-line entry point maintaining original functionality"""
    parser = argparse.ArgumentParser(description="Event Analysis Cross Validation")
    parser.add_argument("config_file", help="Path to the JSON configuration file.")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    run_event_analysis(config)


if __name__ == "__main__":
    main()
