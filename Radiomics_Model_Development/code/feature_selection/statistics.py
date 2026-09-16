from typing import Optional

import pandas as pd
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests


def compute_logrank(
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    duration_column: str,
    event_column: str,
    feature_medians: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute logrank test for each feature, stratifying by its median value (continuous)
    or by its value (binary). This determines which features can separate the data in two
    groups with different survival curves.

    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param duration_column: Column name for duration data.
    :type duration_column: str
    :param event_column: Column name for event data.
    :type event_column: str
    :param feature_medians: Series containing median values for features.
    :type feature_medians: pandas.Series
    :return: DataFrame containing log-rank test statistics and p-values.
    :rtype: pandas.DataFrame
    """
    results = {}
    if feature_medians is None:
        feature_medians = features.median()
    for col in features.columns:
        binary = features[col].unique().size == 2
        if not binary:
            group = (features[col] > feature_medians[col]).values
        else:
            group = (features[col] == 1).values
        results[col] = logrank_test(
            endpoints.loc[group, duration_column],  # type: ignore
            endpoints.loc[~group, duration_column],  # type: ignore
            event_observed_A=endpoints.loc[group, event_column],  # type: ignore
            event_observed_B=endpoints.loc[~group, event_column],  # type: ignore
        )

    # Build a DataFrame from the results
    logrank_df = pd.DataFrame(
        {
            "raw statistic": {col: res.test_statistic for col, res in results.items()},
            "statistic": {col: abs(res.test_statistic) for col, res in results.items()},
            "p_value": {col: res.p_value for col, res in results.items()},
        }
    )
    logrank_df = logrank_df.sort_index()
    return multipletest_correction(logrank_df)


def compute_mannwhitney(
    features: pd.DataFrame, endpoints: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute Mann-Whitney U test statistics for each feature with regards to the binary
    endpoint. This determines wich features have different mean values when separating
    patients by their corresponding endpoint.

    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :return: DataFrame containing Mann-Whitney U test statistics and p-values.
    :rtype: pandas.DataFrame
    """
    results = {}
    for col in features.columns:
        u1, p = mannwhitneyu(
            features.loc[endpoints == 0, col],  # type: ignore
            features.loc[endpoints == 1, col],  # type: ignore
        )
        u2 = endpoints.value_counts().prod() - u1  # Statistic for the second group
        results[col] = {
            "statistic": min(u1, u2),
            "p_value": p,
        }  # Use the smaller U statistic

    # Build a DataFrame from the results
    mannwhitney_df = pd.DataFrame(
        {
            "statistic": {col: res["statistic"] for col, res in results.items()},
            "p_value": {col: res["p_value"] for col, res in results.items()},
        }
    )
    mannwhitney_df = mannwhitney_df.sort_index()
    return multipletest_correction(mannwhitney_df)


def compute_pearson(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation coefficients between each feature pair.

    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :return: DataFrame containing **absolute** Pearson correlation coefficients.
    :rtype: pandas.DataFrame
    """
    return features.corr(method="pearson").abs()


def compute_spearman(
    features: pd.DataFrame, endpoints: pd.DataFrame, duration_column: str
) -> pd.DataFrame:
    """
    Compute Spearman correlation coefficients between individual features and continous endpoint.

    :param features: DataFrame containing feature data.
    :type features: pandas.DataFrame
    :param endpoints: DataFrame containing endpoint data.
    :type endpoints: pandas.DataFrame
    :param duration_column: Column name for duration data.
    :type duration_column: str
    :return: DataFrame containing Spearman correlation coefficients and p-values.
    :rtype: pandas.DataFrame
    """
    results = {}
    for col in features.columns:
        results[col] = spearmanr(features[col], endpoints.loc[:, duration_column])

    # Build a DataFrame from the results
    spearman_df = pd.DataFrame(
        {
            "raw statistic": {col: res.correlation for col, res in results.items()},
            "statistic": {col: abs(res.correlation) for col, res in results.items()},
            "p_value": {col: res.pvalue for col, res in results.items()},
        }
    )
    spearman_df = spearman_df.sort_index()
    return multipletest_correction(spearman_df)


def multipletest_correction(
    test_results: pd.DataFrame, alpha=0.05, method="fdr_bh"
) -> pd.DataFrame:
    if not test_results["p_value"].notna().any():
        return test_results
    test_results_corrected = test_results.copy()
    ok = test_results_corrected["p_value"].notna()
    reject, q_vals, _, _ = multipletests(
        test_results_corrected.loc[ok, "p_value"].values, alpha=alpha, method=method
    )
    test_results_corrected.loc[ok, "q_value"] = q_vals
    test_results_corrected.loc[ok, "reject"] = reject
    return test_results_corrected
