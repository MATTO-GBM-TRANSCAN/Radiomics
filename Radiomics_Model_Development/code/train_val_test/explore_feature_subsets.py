import itertools
import logging

from typing import Union

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator, cross_val_score

from matto_radiomics.models import CoxPH
from matto_radiomics.utils import *


from multiprocessing import Pool

set_config(transform_output="pandas")
_logger = logging.getLogger(__name__)


def init_pool(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model: BaseEstimator,
    cv: BaseCrossValidator,
    scoring_method: str,
):
    """
    Initialize worker processes for parallel evaluation.

    :param X: Feature data.
    :type X: pandas.DataFrame
    :param y: Target data.
    :type y: pandas.Series
    :param model: Model to evaluate.
    :type model: sklearn.base.BaseEstimator
    :param cv: Cross-validator.
    :type cv: sklearn.model_selection.BaseCrossValidator
    :param scoring_method: Scoring method for evaluation.
    :type scoring_method: str
    """
    global X_data, y_data, model_glob, cv_glob, scorer
    X_data = X
    y_data = y
    model_glob = model
    cv_glob = cv
    if scoring_method == "f1":
        scorer = make_scorer(f1_score, pos_label=1)
    elif scoring_method == "roc_auc":
        scorer = make_scorer(roc_auc_score, response_method="predict_proba")
    elif scoring_method == "balanced_accuracy":
        scorer = make_scorer(balanced_accuracy_score)
    else:
        scorer = None


def evaluate_subset(candidate_features: list[str]) -> tuple[list[str], float]:
    """
    Evaluate a subset of features.

    :param candidate_features: List of feature names.
    :type candidate_features: list[str]
    :return: Tuple containing the feature subset and mean score.
    :rtype: tuple[list[str], float]
    """
    scores = cross_val_score(
        model_glob,
        X_data[list(candidate_features)],
        y_data,
        cv=cv_glob,
        scoring=scorer,
        n_jobs=1,  # keep CV single‐threaded to avoid oversubscription
    )
    return candidate_features, np.mean(scores)


def combinations(iterable: list[str], r: int, required_in_combination: list[str] = []):
    """
    Generate combinations of features.

    :param iterable: Iterable of feature names.
    :type iterable: list[str]
    :param r: Number of features in each combination.
    :type r: int
    :param required_in_combination: Features that must be included.
    :type required_in_combination: list[str]
    :return: Generator of feature combinations.
    :rtype: generator
    """
    for combination in itertools.combinations(iterable, r):
        if not all(item in combination for item in required_in_combination):
            continue
        yield combination


def parallel_evaluate_feature_subsets(
    model: Union[CoxPH, LogisticRegression],
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    splitter: BaseCrossValidator,
    scoring_method: str,
    scoring_method_ascending: bool = True,
    subset_size_range: tuple[int, int] = (1, 9),
    n_processes: int = 12,
):
    """
    Perform parallel evaluation of feature subsets.

    :param model: Model to evaluate.
    :type model: sklearn.base.BaseEstimator
    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param splitter: Cross-validator for splitting the data.
    :type splitter: sklearn.model_selection.BaseCrossValidator
    :param scoring_method: Scoring method for evaluation.
    :type scoring_method: str
    :param scoring_method_ascending: Whether higher scores are better.
    :type scoring_method_ascending: bool
    :param subset_size_range: Range of subset sizes to evaluate.
    :type subset_size_range: tuple[int, int]
    :param n_processes: Number of processes for parallel evaluation.
    :type n_processes: int
    :return: Dictionary containing results.
    :rtype: dict
    """
    results = {}
    # Set initial best to -inf or +inf depending on metric direction
    best_model_performance = -np.inf if scoring_method_ascending else np.inf
    best_feature_subset = []

    # Create the pool once, passing globals
    with Pool(
        processes=n_processes,
        initializer=init_pool,
        initargs=(features, endpoints, model, splitter, scoring_method),
    ) as pool:
        for subset_size in range(*subset_size_range):
            _logger.info(f"Subset size: {subset_size}")

            # Feature subset generator
            if (len(features.columns) < 15) or (subset_size < 3):
                # If there are fewer than 15 features, use all combinations
                # or if subset size is small, use all combinations
                subset_iter = combinations(features.columns, subset_size)
            else:  # If there are more than 15 features, sequentially add features
                subset_iter = combinations(
                    features.columns, subset_size, best_feature_subset
                )
            # Track the best for this subset size
            best_subset_performance = -np.inf if scoring_method_ascending else np.inf
            selected_features = []
            for candidate_features, score in pool.imap_unordered(
                evaluate_subset, subset_iter, chunksize=10
            ):
                if (scoring_method_ascending and score > best_subset_performance) or (
                    not scoring_method_ascending and score < best_subset_performance
                ):
                    best_subset_performance = score
                    selected_features = candidate_features
            results[subset_size] = {
                scoring_method: best_subset_performance,
                "features": selected_features,
            }
            # update overall best
            is_better = (
                scoring_method_ascending
                and best_subset_performance > best_model_performance
            ) or (
                not scoring_method_ascending
                and best_subset_performance < best_model_performance
            )
            if is_better:
                best_model_performance = best_subset_performance
                best_feature_subset = selected_features
            _logger.info(
                f"Best model performance: {best_model_performance}, "
                f"Best feature subset: {best_feature_subset}"
            )
    return list(best_feature_subset), results
