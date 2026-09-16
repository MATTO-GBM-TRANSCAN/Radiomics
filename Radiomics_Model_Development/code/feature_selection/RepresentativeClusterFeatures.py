import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin

from .statistics import multipletest_correction

_logger = logging.getLogger(__name__)


class RepresentativeClusterFeatures(TransformerMixin, BaseEstimator):
    """
    Transformer for reducing features based on clustering correlated features
    and selecting a representative from each cluster.

    :param correlation_clustering_method: The method used to compute the correlation matrix for clustering,
        defaults to "pearson".
    :type correlation_clustering_method: Literal["pearson", "kendall", "spearman"]
    :param correlation_selection_method: The method used to compute the correlation for feature selection
        within clusters, defaults to "spearman".
    :type correlation_selection_method: Literal["pearson", "kendall", "spearman"]
    :param correlation_threshold: The threshold above which features will be clustered together,
        defaults to 0.7.
    :type correlation_threshold: float
    :param clustering_method: The method used for hierarchical clustering of features, defaults to "average".
    :type clustering_method: str
    :param filter_by_p_value: Whether to keep only features with p-value < 0.05, defaults to False.
    :type filter_by_p_value: bool
    """

    def __init__(
        self,
        correlation_clustering_method: Literal[
            "pearson", "kendall", "spearman"
        ] = "pearson",
        correlation_selection_method: Literal[
            "pearson", "kendall", "spearman"
        ] = "spearman",
        correlation_threshold: float = 0.7,
        clustering_method: str = "average",
        filter_by_p_value: bool = False,
    ):
        self.correlation_clustering_method = correlation_clustering_method
        self.correlation_selection_method = correlation_selection_method
        self.correlation_threshold = correlation_threshold
        self.clustering_method = clustering_method
        self.filter_by_p_value = filter_by_p_value

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Fit the transformer to the data.

        :param X: Feature data.
        :type X: pandas.DataFrame
        :param y: Target data.
        :type y: pandas.DataFrame
        :return: Fitted transformer.
        :rtype: RepresentativeClusterFeatures
        """

        correlation_matrix = X.corr(method=self.correlation_clustering_method).abs()  # type: ignore
        distance_matrix = 1.0 - correlation_matrix
        distance_matrix = distance_matrix.fillna(1.0)
        distance_matrix = distance_matrix.clip(lower=0.0, upper=1.0)
        distance_matrix = squareform(distance_matrix)
        feature_linkage = linkage(distance_matrix, method=self.clustering_method)

        clusters = cut_tree(feature_linkage, height=1.0 - self.correlation_threshold)
        clusters = np.concatenate(clusters)
        n_clusters = clusters.max() + 1
        _logger.debug("Number of clusters: %d", n_clusters)

        importance_to_target = X.corrwith(
            y, method=self.correlation_selection_method  # type: ignore
        ).abs()
        self.to_keep_ = []
        non_significant_features = 0
        for cluster in range(n_clusters):
            cluster_features = X.columns[clusters == cluster]
            candidate_features = importance_to_target.loc[cluster_features]
            if self.filter_by_p_value:
                candidate_features = candidate_features[
                    candidate_features["p_value"] < 0.05
                ]
            if len(candidate_features) == 0:
                non_significant_features += 1
                continue
            to_keep = candidate_features["statistic"].idxmax()
            self.to_keep_.append(to_keep)

        _logger.warning(
            "No significant feature in %d of %d clusters",
            non_significant_features,
            n_clusters,
        )
        _logger.info(
            "Features after clustering %d/%d", len(self.to_keep_), X.columns.size
        )

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """
        Transform the data, keeping only the representative features from each cluster.

        :param X: Feature data.
        :type X: pandas.DataFrame
        :param y: Target data (not used, kept for compatibility with sklearn), defaults to None.
        :type y: pandas.DataFrame, optional
        :return: Transformed feature data.
        :rtype: pandas.DataFrame
        """

        X_ = X.copy()
        X_ = X_.loc[:, self.to_keep_]
        return X_


class RepresentativeClusterFeaturesWithPrecomputedCorrelations(
    RepresentativeClusterFeatures
):
    """
    Transformer for reducing features based on clustering correlated features
    and selecting a representative from each cluster. It uses **precomputed** correlations.

    :param correlation_matrix: Feature correlation matrix.
    :type correlation_matrix: pandas.Dataframe
    :param importance_to_target: Features to target correlation matrix.
    :type importance_to_target: pandas.Dataframe
    :param correlation_threshold: The threshold above which features will be clustered together,
        defaults to 0.7.
    :type correlation_threshold: float
    :param clustering_method: The method used for hierarchical clustering of features, defaults to "average".
    :type clustering_method: str
    :param importance_to_target_method: How to choose the representative from each cluster.
        One of ["higher statistic", "lower statistic", "lower p_value"], defaults to "higher statistic".
    :type importance_to_target_method: str
    :param filter_by_p_value: Whether to keep only features with p-value < 0.05, defaults to False.
    :type filter_by_p_value: bool
    """

    def __init__(
        self,
        correlation_matrix: pd.DataFrame,
        importance_to_target: pd.DataFrame,
        correlation_threshold: float = 0.7,
        clustering_method: str = "average",
        importance_to_target_method: Literal[
            "higher statistic",
            "lower statistic",
            "lower p_value",
        ] = "higher statistic",
        filter_by_p_value: bool = False,
    ):
        self.correlation_matrix = correlation_matrix
        self.importance_to_target = importance_to_target
        self.correlation_threshold = correlation_threshold
        self.clustering_method = clustering_method
        self.importance_to_target_method = importance_to_target_method
        self.filter_by_p_value = filter_by_p_value

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """
        Fit the transformer to the data.

        :param X: Feature data.
        :type X: pandas.DataFrame
        :param y: Target data (not used, kept for compatibility with sklearn), defaults to None.
        :type y: pandas.DataFrame, optional
        :return: Fitted transformer.
        :rtype: RepresentativeClusterFeaturesWithPrecomputedCorrelations
        """

        correlation_matrix = self.correlation_matrix.loc[X.columns, X.columns].copy()  # type: ignore
        distance_matrix = 1.0 - correlation_matrix
        distance_matrix = distance_matrix.fillna(1.0)
        distance_matrix = distance_matrix.clip(lower=0.0, upper=1.0)
        distance_matrix = squareform(distance_matrix)
        feature_linkage = linkage(distance_matrix, method=self.clustering_method)

        clusters = cut_tree(feature_linkage, height=1.0 - self.correlation_threshold)
        clusters = np.concatenate(clusters)
        n_clusters = clusters.max() + 1
        _logger.debug("Number of clusters: %d", n_clusters)

        to_keep_ = []
        non_significant_features = 0
        for cluster in range(n_clusters):
            cluster_features = correlation_matrix.columns[clusters == cluster]
            candidate_features = self.importance_to_target.loc[cluster_features]
            if self.filter_by_p_value:
                candidate_features = candidate_features[
                    candidate_features["p_value"] < 0.05
                ]
            if len(candidate_features) == 0:
                non_significant_features += 1
                continue
            if self.importance_to_target_method == "higher statistic":
                to_keep = candidate_features["statistic"].idxmax()
            elif self.importance_to_target_method == "lower statistic":
                to_keep = candidate_features["statistic"].idxmin()
            elif self.importance_to_target_method == "lower p_value":
                to_keep = candidate_features["p_value"].idxmin()
            to_keep_.append(to_keep)  # type: ignore

        self.to_keep_ = to_keep_

        # corrected_p_values = multipletest_correction(
        #     self.importance_to_target.loc[to_keep_], method="fdr_bh"
        # )
        # self.to_keep_ = corrected_p_values.index[corrected_p_values["reject"]].to_list()

        _logger.warning(
            "No significant feature in %d of %d clusters",
            non_significant_features,
            n_clusters,
        )
        _logger.info(
            "Features after clustering %d/%d", len(self.to_keep_), X.columns.size
        )

        return self
