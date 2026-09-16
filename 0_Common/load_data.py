import logging
from functools import reduce
from typing import Optional

import pandas as pd

_logger = logging.getLogger(__name__)


def load_data(
    radiomic_features_path: str,
    id_column: str,
    clinical_features_path: Optional[str] = None,
    endpoints_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index, pd.Index, pd.Index]:
    """
    This function reads radiomic features, clinical features, and endpoints from their respective CSV files,
    processes them, and returns a tuple containing the combined features, endpoints, and metadata about the features.

    :param radiomic_features_path: Path to the CSV file containing radiomic features.
    :type radiomic_features_path: str
    :param id_column: Name of the column to use as the index for all dataframes.
    :type id_column: str
    :param clinical_features_path: Path to the CSV file containing clinical features.
    :type clinical_features_path: str, optional
    :param endpoints_path: Path to the CSV file containing endpoints.
    :type endpoints_path: str, optional
    :return: A tuple containing:

        - features (pandas.DataFrame): DataFrame with combined radiomic and clinical features.
        - endpoints (pandas.DataFrame): DataFrame with endpoints.
        - radiomic_features_names (pandas.Index): Index of radiomic feature names.
        - clinical_features_names (pandas.Index): Index of clinical feature names.
        - binary_features (pandas.Index): Index of binary features.

    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.Index, pd.Index, pd.Index]
    """

    _logger.debug("Loading radiomic features from %s", radiomic_features_path)
    radiomic_features = pd.read_csv(radiomic_features_path)
    # Pre-filter columns: keep id_column and any column that contains "_"
    pre_filter = radiomic_features.columns.to_series().apply(
        lambda x: (x == id_column) or ("_" in x)
    )

    # Filter radiomic_features columns using pre_filter
    radiomic_features = radiomic_features.loc[:, pre_filter]
    radiomic_features = radiomic_features.set_index(id_column)
    feature_filter_criteria = [
        ~radiomic_features.columns.str.startswith("diagnostics"),
        radiomic_features.columns.str.contains("Mask-original_VolumeNum"),
        radiomic_features.columns.str.contains("Mask-original_VoxelNum"),
    ]
    feature_filter = reduce(lambda x, y: x | y, feature_filter_criteria)
    radiomic_features = radiomic_features.loc[:, feature_filter]
    radiomic_feature_names = radiomic_features.columns

    clinical_features = pd.DataFrame()
    if clinical_features_path:
        _logger.debug("Loading clinical features from %s", clinical_features_path)
        clinical_features = pd.read_csv(clinical_features_path)
        clinical_features = clinical_features.set_index(id_column)
        clinical_features = clinical_features.loc[
            clinical_features.index.intersection(
                radiomic_features.index, sort=None  # type: ignore
            )
        ]
        clinical_features["Methylated"] = clinical_features["Methylated"].fillna(0.5)
    clinical_features = clinical_features.dropna(axis=1, how="any")
    clinical_feature_names = clinical_features.columns

    features = pd.concat([radiomic_features, clinical_features], axis=1)
    features = features.dropna(
        axis=0, how="any"
    )  # Drop patients with NaN values, e.g. patients with no clinical data
    binary_feature_names = features.columns[features.nunique(axis=0) == 2]

    _logger.info("Features shape: %s", features.shape)
    _logger.info(
        "Radiomic features: %d, Clinical features: %d (%d of them binary)",
        len(radiomic_feature_names),
        len(clinical_feature_names),
        len(binary_feature_names),
    )

    endpoints = pd.DataFrame()
    if endpoints_path:
        _logger.debug("Loading endpoints from %s", endpoints_path)
        endpoints = pd.read_csv(endpoints_path)
        endpoints = endpoints.set_index(id_column)
        endpoints = endpoints.loc[
            endpoints.index.intersection(features.index, sort=None)  # type: ignore
        ]

        _logger.info("Endpoints shape: %s", endpoints.shape)

    return (
        features,
        endpoints,
        radiomic_feature_names,
        clinical_feature_names,
        binary_feature_names,
    )
