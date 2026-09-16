import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

from typing import Tuple, List, Dict, Union

import pandas as pd
from sklearn import set_config
from sklearn.model_selection import KFold

from .evaluate_feature_subset import main as test_set_evaluation
from .evaluate_feature_subset import time_to_event_cross_validation
from .explore_feature_subsets import parallel_evaluate_feature_subsets
from matto_radiomics.feature_selection import (
    get_feature_preprocessing_pipeline,
    get_feature_selection_pipeline,
)
from matto_radiomics.feature_selection.statistics import (
    compute_logrank,
    compute_pearson,
    compute_spearman,
    multipletest_correction,
)
from matto_radiomics.models import CoxPH, KaplanMeier
from matto_radiomics.utils import *

set_config(transform_output="pandas")


def multivariate_modelling(
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    duration_column: str,
    event_column: str,
    radiomic_feature_names: pd.Index,
    clinical_feature_names: pd.Index,
    feature_correlation_matrix: pd.DataFrame,
    spearman_with_target: pd.DataFrame,
    scoring_method: str,
    scoring_method_ascending: bool,
    additional_metrics: tuple[str] = ("concordance_index",),
    k_folds: int = 5,
    n_processes: int = 12,
    random_state: int = 5,
) -> Tuple[List[str], Dict[str, Union[List[str], float]]]:
    """
    Perform multivariate modeling for time-to-event analysis.

    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param duration_column: Column name for duration data.
    :type duration_column: str
    :param event_column: Column name for event data.
    :type event_column: str
    :param radiomic_feature_names: Index of radiomic feature names.
    :type radiomic_feature_names: pandas.Index
    :param clinical_feature_names: Index of clinical feature names.
    :type clinical_feature_names: pandas.Index
    :param feature_correlation_matrix: Correlation matrix of features.
    :type feature_correlation_matrix: pandas.DataFrame
    :param spearman_with_target: Spearman correlation with target.
    :type spearman_with_target: pandas.DataFrame
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

    pipeline = get_feature_selection_pipeline(
        feature_correlation_matrix,
        spearman_with_target,
        radiomic_feature_names,
        clinical_feature_names,
    )

    features = pipeline.fit_transform(
        features,
        endpoints.loc[:, duration_column],
    )

    model = CoxPH(
        duration_column=duration_column,
        event_column=event_column,
        scoring_method=scoring_method,
    )
    splitter = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    best_feature_subset, exploration_results = parallel_evaluate_feature_subsets(
        model,
        features,
        endpoints,
        splitter,
        scoring_method,
        scoring_method_ascending,
        subset_size_range=(
            1,
            (len(features.columns) + 1 if len(features.columns) < 9 else 9),
        ),  # Evaluate subsets of size 1 up to 8
        n_processes=n_processes,
    )
    if save:
        (save_dir / "validation_multivariate").mkdir()
        filename = save_dir / "validation_multivariate" / "validation_multivariate.json"
        with open(filename, "w") as f:
            json.dump(exploration_results, f, indent=4)

    results = {}
    results["feature_subset"] = best_feature_subset
    results[scoring_method] = time_to_event_cross_validation(
        model, scoring_method, features[list(best_feature_subset)], endpoints, splitter
    )
    for s_method in additional_metrics:
        results[s_method] = time_to_event_cross_validation(
            model,
            s_method,
            features[list(best_feature_subset)],
            endpoints,
            splitter,
        )
    return results


def univariate_modelling(
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    duration_column: str,
    event_column: str,
    radiomic_feature_names: pd.Index,
    clinical_feature_names: pd.Index,
    feature_correlation_matrix: pd.DataFrame,
    logrank: pd.DataFrame,
    k_folds: int = 5,
    random_state: int = 5,
) -> Dict[str, Union[List[str], float]]:
    """
    Perform univariate modeling for time-to-event analysis.

    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param duration_column: Column name for duration data.
    :type duration_column: str
    :param event_column: Column name for event data.
    :type event_column: str
    :param radiomic_feature_names: Index of radiomic feature names.
    :type radiomic_feature_names: pandas.Index
    :param clinical_feature_names: Index of clinical feature names.
    :type clinical_feature_names: pandas.Index
    :param feature_correlation_matrix: Correlation matrix of features.
    :type feature_correlation_matrix: pandas.DataFrame
    :param logrank: LogRank statistic for each feature.
    :type logrank: pandas.DataFrame
    :param k_folds: Number of cross-validation folds.
    :type k_folds: int
    :param random_state: Random state for reproducibility.
    :type random_state: int
    :return: Dictionary containing results.
    :rtype: dict[str, list[str] | float]
    """
    _univariate_logger = _logger.getChild("univariate_modelling")

    if len(features.columns) < 1:
        _univariate_logger.error(
            "Not enough features for univariate modelling, at least 1 feature is required. Skipping this stage."
        )
        return []

    pipeline = get_feature_selection_pipeline(
        feature_correlation_matrix,
        logrank,
        radiomic_feature_names,
        clinical_feature_names,
    )

    features = pipeline.fit_transform(
        features,
        endpoints.loc[:, duration_column],
    )

    significant_features = Counter()
    splitter = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(splitter.split(features, endpoints)):
        train_endpoints = endpoints.iloc[train_index]
        train_features = features.iloc[train_index]
        test_endpoints = endpoints.iloc[test_index]
        test_features = features.iloc[test_index]

        train_logrank = compute_logrank(
            train_features, train_endpoints, duration_column, event_column
        )
        test_logrank = compute_logrank(
            test_features, test_endpoints, duration_column, event_column
        )
        if save:
            fold_logrank = train_logrank.join(
                test_logrank, lsuffix="-train", rsuffix="-test"
            )
            with pd.ExcelWriter(
                save_dir / "clinical_impact" / "LogRank.xlsx", mode="a"
            ) as writer:
                fold_logrank.to_excel(writer, sheet_name=f"Fold {i}")

        fold_significant_features = []
        for col in train_features.columns:
            _univariate_logger.debug(
                "Feature %s logrank p-value: train = %.3f, test = %.3f",
                col,
                logrank.loc[col, "p_value"],
                train_logrank.loc[col, "p_value"],
            )
            if train_logrank.loc[col, "p_value"] < 0.05:
                fold_significant_features.append(col)
        _univariate_logger.info(
            "Fold %d univariate significant features: %s",
            i,
            fold_significant_features,
        )

        significant_features.update(fold_significant_features)

    threshold = k_folds
    significant_features_ = [
        feature for feature, count in significant_features.items() if count >= threshold
    ]
    _univariate_logger.info(
        "Significant features in %d out of %d folds before multiple test correction: %d",
        threshold,
        k_folds,
        len(significant_features_),
    )
    corrected_p_values = multipletest_correction(
        logrank.loc[significant_features_], method="fdr_bh"
    )
    significant_features = corrected_p_values.index[
        corrected_p_values["reject"]
    ].to_list()
    for feature in significant_features:
        KaplanMeier(
            features[feature],
            endpoints,
            duration_column,
            event_column,
            save_dir=save_dir / "validation_univariate",
            save=save,
        )
    _univariate_logger.info(
        "Significant features in %d out of %d folds: %d",
        threshold,
        k_folds,
        len(significant_features),
    )
    return significant_features


def test_set_univariate(
    features: pd.DataFrame,
    binary_feature_names: pd.Index,
    significant_features: list[str],
    root_dir: Path,
    config: dict,
    id_column: str,
    duration_column: str,
    event_column: str,
    save_dir: Path,
    save: bool,
) -> list[str]:
    """
    Perform univariate analysis on the test set.

    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param binary_feature_names: Index of binary feature names.
    :type binary_feature_names: pandas.Index
    :param significant_features: List of significant features.
    :type significant_features: list[str]
    :param root_dir: Root directory for data.
    :type root_dir: pathlib.Path
    :param config: Configuration dictionary.
    :type config: dict
    :param id_column: Column name for unique IDs.
    :type id_column: str
    :param duration_column: Column name for duration data.
    :type duration_column: str
    :param event_column: Column name for event data.
    :type event_column: str
    :param save_dir: Directory to save results.
    :type save_dir: pathlib.Path
    :param save: Whether to save results.
    :type save: bool
    :return: List of significant features in test set.
    :rtype: list[str]
    """
    if len(significant_features) == 0:
        _logger.error(
            "Not enough features for evaluation, at least 1 feature is required."
        )
        return {}

    features = features[significant_features]
    preprocessor = get_feature_preprocessing_pipeline(
        features.columns, binary_feature_names.intersection(features.columns)
    )
    features = preprocessor.fit_transform(features)
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
    test_features = test_features[significant_features]
    test_features = preprocessor.transform(test_features)
    test_significant_features = []
    for feature in significant_features:
        p1 = KaplanMeier(
            test_features[feature],
            test_endpoints,
            duration_column,
            event_column,
            save_dir=save_dir / "test_median",
            save=save,
        )
        p2 = KaplanMeier(
            test_features[feature],
            test_endpoints,
            duration_column,
            event_column,
            features[feature].median(),
            save_dir=save_dir / "train_median",
            save=save,
        )
        if (p1 < 0.05) or (p2 < 0.05):
            test_significant_features.append(feature)
    return test_significant_features


def run_time_to_event_analysis(config: dict):
    """Main entry point for external programs to run time-to-event analysis.

    :param config: Configuration dictionary with same structure as JSON config file
    :return: Dictionary containing analysis results
    """
    global _logger, save, save_dir

    # Extract parameters from config
    id_column = config["id_column"]
    duration_column = config["duration_column"]
    event_column = config["event_column"]
    experiment_name = config.get("experiment_name", duration_column)
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

    log_path = None
    if save:
        run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir / experiment_name / run_start_time
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "log.log"
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
    _logger = set_logger(
        name="matto_radiomics.time_to_event", level=logging.DEBUG, outpath=log_path
    )
    _logger.info("Starting %s analysis", duration_column)

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

    # Handle missing values in endpoints
    endpoints = endpoints[[duration_column, event_column]].dropna()
    features = features.loc[features.index.intersection(endpoints.index, sort=None)]
    _logger.info(
        "Data after NaN removal - Events shape: %s, Features shape: %s",
        endpoints.shape,
        features.shape,
    )

    # Compute statistics
    feature_correlation_matrix = compute_pearson(features)
    spearman_with_target = compute_spearman(features, endpoints, duration_column)
    logrank = compute_logrank(features, endpoints, duration_column, event_column)
    if save:
        (save_dir / "clinical_impact").mkdir()
        with pd.ExcelWriter(
            save_dir / "clinical_impact" / "LogRank.xlsx", mode="w"
        ) as writer:
            logrank.to_excel(writer, sheet_name="All")
        with pd.ExcelWriter(
            save_dir / "clinical_impact" / "Spearman.xlsx", mode="w"
        ) as writer:
            spearman_with_target.to_excel(writer, sheet_name="Spearman")

    # Preprocess features
    pipeline = get_feature_preprocessing_pipeline(
        features.columns, binary_feature_names
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
        duration_column,
        event_column,
        radiomic_feature_names,
        clinical_feature_names,
        feature_correlation_matrix,
        logrank,
        k_folds=k_folds,
    )
    results["univariate_results"] = univariate_results

    # Multivariate modelling
    _logger.info("Starting multivariate modelling")
    if len(features) < 2:
        _logger.error(
            "Not enough samples (%d) for multivariate modelling. Requires at least 2.",
            len(features),
        )
        multivariate_results = {}
    else:
        multivariate_results = multivariate_modelling(
            transformed_features,
            endpoints,
            duration_column,
            event_column,
            radiomic_feature_names,
            clinical_feature_names,
            feature_correlation_matrix,
            spearman_with_target,
            scoring_method=scoring_method[0],
            scoring_method_ascending=scoring_method_ascending,
            additional_metrics=scoring_method[1:],
            k_folds=k_folds,
        )
    results["multivariate_results"] = multivariate_results

    # Test set evaluation
    if (config.get("test_radiomic_features") is not None) and (
        config.get("test_endpoints") is not None
    ):
        _logger.info("Starting univariate test set evaluation")
        univariate_test_results = test_set_univariate(
            features,
            binary_feature_names,
            univariate_results,
            root_dir,
            config,
            id_column,
            duration_column,
            event_column,
            save_dir / "test_univariate",
            save,
        )
        results["univariate_test_results"] = univariate_test_results

        _logger.info("Starting multivariate test set evaluation")
        multivariate_test_results = test_set_evaluation(
            root_dir,
            features,
            endpoints,
            id_column,
            multivariate_results["feature_subset"],
            binary_feature_names,
            duration_column,
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
    parser = argparse.ArgumentParser(
        description="Time to Event Analysis Cross Validation"
    )
    parser.add_argument("config_file", help="Path to the JSON configuration file.")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    run_time_to_event_analysis(config)


if __name__ == "__main__":
    main()
