from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .RepresentativeClusterFeatures import (
    RepresentativeClusterFeaturesWithPrecomputedCorrelations,
)
from .VarianceThreshold import VarianceThreshold


def get_feature_preprocessing_pipeline(
    feature_names: pd.Index, binary_feature_names: pd.Index
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for features:
     1. Scale to [0,1].
     2. Discard features with variance < 0.01.
     3. Z-Score normalization.

    :param feature_names: Index of feature names.
    :type feature_names: pandas.Index
    :param binary_feature_names: Index of binary feature names.
    :type binary_feature_names: pandas.Index

    :return: Preprocessing pipeline.
    :rtype: sklearn.compose.ColumnTransformer
    """
    numerical_features = make_pipeline(
        MinMaxScaler(),
        VarianceThreshold(threshold=0.01),
        StandardScaler(),
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_features,
                feature_names.difference(binary_feature_names),
            ),
            ("cat", "passthrough", binary_feature_names),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_selection_pipeline(
    feature_correlation_matrix: pd.DataFrame,
    importance_to_target: pd.DataFrame,
    radiomic_feature_names: pd.Index,
    clinical_feature_names: pd.Index,
    importance_to_target_method: Literal[
        "higher statistic",
        "lower statistic",
        "lower p_value",
    ] = "higher statistic",
) -> ColumnTransformer:
    """
    Create a feature selection pipeline. Radiomic features are clustered
    based on their correlation, and one representative from each cluster
    is kept. Clinical features are not affected.

    :param feature_correlation_matrix: Correlation matrix of features
    :type feature_correlation_matrix: pandas.DataFrame
    :param importance_to_target: Importance scores for features
    :type importance_to_target: pandas.DataFrame
    :param radiomic_feature_names: Index of radiomic feature names
    :type radiomic_feature_names: pandas.Index
    :param clinical_feature_names: Index of clinical feature names
    :type clinical_feature_names: pandas.Index
    :param importance_to_target_method: Method to determine importance to target,
        defaults to "higher statistic"
    :type importance_to_target_method: str
    :return: Feature selection pipeline
    :rtype: sklearn.compose.ColumnTransformer
    """
    selector = ColumnTransformer(
        transformers=[
            (
                "radiomics",
                RepresentativeClusterFeaturesWithPrecomputedCorrelations(
                    feature_correlation_matrix,
                    importance_to_target,
                    filter_by_p_value=True,
                    importance_to_target_method=importance_to_target_method,
                ),
                lambda df: df.columns.isin(radiomic_feature_names),
            ),
            (
                "clinical",
                "passthrough",
                lambda df: df.columns.isin(clinical_feature_names),
            ),
        ],
        verbose_feature_names_out=False,
    )
    return selector


def get_full_feature_pipeline(
    feature_correlation_matrix: pd.DataFrame,
    importance_to_target: pd.DataFrame,
    feature_names: pd.Index,
    radiomic_feature_names: pd.Index,
    clinical_feature_names: pd.Index,
    binary_feature_names: pd.Index,
    importance_to_target_method: Literal[
        "higher statistic",
        "lower statistic",
        "lower p_value",
    ] = "higher statistic",
) -> Pipeline:
    """
    Create a full feature pipeline including preprocessing and selection.

    :param feature_correlation_matrix: Correlation matrix of features
    :type feature_correlation_matrix: pandas.DataFrame
    :param importance_to_target: Importance scores for features
    :type importance_to_target: pandas.DataFrame
    :param feature_names: Index of feature names
    :type feature_names: pandas.Index
    :param radiomic_feature_names: Index of radiomic feature names.
    :type radiomic_feature_names: pandas.Index
    :param clinical_feature_names: Index of clinical feature names
    :type clinical_feature_names: pandas.Index
    :param binary_feature_names: Index of binary feature names
    :type binary_feature_names: pandas.Index
    :param importance_to_target_method: Method to determine importance to target,
        defaults to "higher statistic"
    :type importance_to_target_method: str
    :return: Full feature pipeline.
    :rtype: sklearn.pipeline.Pipeline
    """
    preprocessor = get_feature_preprocessing_pipeline(
        feature_names, binary_feature_names
    )
    selector = get_feature_selection_pipeline(
        feature_correlation_matrix,
        importance_to_target,
        radiomic_feature_names,
        clinical_feature_names,
        importance_to_target_method=importance_to_target_method,
    )
    return make_pipeline(preprocessor, selector)
