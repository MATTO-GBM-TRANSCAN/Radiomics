from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

from matto_radiomics.utils import save_figures


def KaplanMeier(
    variable: pd.Series,
    endpoints: pd.DataFrame,
    duration_column: str,
    event_column: str,
    median: Optional[float] = None,
    save_dir: Path = Path("."),
    filename: Optional[str] = None,
    save: bool = True,
):
    """
    Plot Kaplan-Meier curves splitting the data in two groups.

    If the variable is continuous, the median is used. If the variable is binary,
    its value is used to split the data. The statistical difference of the survival
    of the groups is assessed using the LogRank test.

    :param variable: The variable used to split the data into two groups. Can be binary or continuous.
    :type variable: pandas.Series
    :param endpoints: DataFrame containing the duration and event columns for survival analysis.
    :type endpoints: pandas.DataFrame
    :param duration_column: Name of the column in `endpoints` representing the duration/time-to-event.
    :type duration_column: str
    :param event_column: Name of the column in `endpoints` representing the event indicator (1=event, 0=censored).
    :type event_column: str
    :param median: Median value to split the variable if it is continuous. If None, computed from `variable`.
    :type median: float, optional
    :param save_dir: Directory to save the plot, defaults to current directory.
    :type save_dir: Path
    :param filename: Filename for saving the plot. If None, uses the variable name.
    :type filename: str, optional
    :param save: Whether to save the plot, defaults to True.
    :type save: bool
    :returns: p-value from the log-rank test comparing the two groups.
    :rtype: float
    """
    if median is None:
        median = variable.median()
    binary = variable.unique().size == 2
    if not binary:
        group = (variable > median).values
    else:
        group = (variable == 1).values

    if (group.sum() == 0) or (group.sum() == len(endpoints)):  # type: ignore
        return

    fig, ax = plt.subplots()

    durations_a = endpoints.loc[group, duration_column]  # type: ignore
    events_a = endpoints.loc[group, event_column]  # type: ignore
    kmf_a = KaplanMeierFitter()
    kmf_a.fit(
        durations_a,
        events_a,
        label=f"{'' if binary else '>'}{'1' if binary else ' Median'}",
    )
    ax = kmf_a.plot_survival_function(
        ax=ax,
        label=f"{'' if binary else '>'}{'1' if binary else ' Median'} ({kmf_a.median_survival_time_/30:.1f} months)",
    )

    durations_b = endpoints.loc[~group, duration_column]  # type: ignore
    events_b = endpoints.loc[~group, event_column]  # type: ignore
    kmf_b = KaplanMeierFitter()
    kmf_b.fit(
        durations_b,
        events_b,
        label=f"{'' if binary else '<='}{'0' if binary else ' Median'}",
    )
    ax = kmf_b.plot_survival_function(
        ax=ax,
        label=f"{'' if binary else '<='}{'0' if binary else ' Median'} ({kmf_b.median_survival_time_/30:.1f} months)",
    )

    result = logrank_test(
        durations_a,
        durations_b,
        event_observed_A=events_a,
        event_observed_B=events_b,
    )

    x_max = endpoints[duration_column].max()
    step = (
        3 if (x_max // 30 + 1 <= 24) else 6
    )  # If there are more than 24 months, use 6 months step for x-axis ticks, if not, use 3 months step
    ax.set_xticks(  # type: ignore
        range(0, x_max, 30 * step),
        labels=range(0, x_max // 30 + 1, step),  # type: ignore
    )
    ax.set_xlabel("Months")  # type: ignore
    ax.legend(  # type: ignore
        title=f"{variable.name} (p = {result.p_value:.3f})",
        ncols=2,
        loc="upper right",
        fontsize="small",
    )

    ## Save a version of the Kaplan-Meier plot without at-risk counts
    # if save:
    #     filename_ = (
    #         filename + "_no_counts"
    #         if filename != None
    #         else f"{variable.name}_no_counts"
    #     )
    #     save_figures(fig, save_dir, filename_)

    add_at_risk_counts(kmf_a, kmf_b, xticks=range(0, x_max, 30 * step), ax=ax)
    plt.tight_layout()

    if save:
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        filename_ = filename if filename != None else variable.name
        save_figures(fig, save_dir, f"km_{filename_}")

    plt.close()

    return result.p_value
