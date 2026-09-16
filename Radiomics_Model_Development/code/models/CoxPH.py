from typing import Literal

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator


# Reference: https://github.com/ManuelaS/sklearn-lifelines/blob/master/sklearn_lifelines/estimators_wrappers.py
class CoxPH(BaseEstimator):
    """
    Scikit-learn wrapper for lifelines' Cox proportional hazards model.

    :param duration_column: Column name for duration data.
    :type duration_column: str
    :param event_column: Column name for event data.
    :type event_column: str
    :param scoring_method: Scoring method for evaluation, defaults to "concordance_index".
    :type scoring_method: str
    """

    def __init__(
        self,
        *,
        duration_column: str,
        event_column: str,
        scoring_method: Literal[
            "concordance_index", "log_likelihood"
        ] = "concordance_index"
    ):
        self.duration_column = duration_column
        self.event_column = event_column
        self.scoring_method = scoring_method

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Fit the Cox proportional hazards model.

        :param X: Feature data.
        :type X: pandas.DataFrame
        :param y: Target data.
        :type y: pandas.DataFrame
        :return: Fitted model.
        :rtype: CoxPH
        """

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, index=y.index)
        X_ = X.assign(
            **{
                self.duration_column: y[self.duration_column],
                self.event_column: y[self.event_column],
            }
        )
        est = CoxPHFitter()
        est.fit(
            X_,
            duration_col=self.duration_column,
            event_col=self.event_column,
        )
        self.estimator = est
        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict expected survival times.

        :param X: Feature data.
        :type X: pandas.DataFrame
        :return: Predicted survival times.
        :rtype: pandas.Series
        """
        return self.estimator.predict_expectation(X)

    def score(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Score the model using the specified scoring method at model initialization.

        :param X: Feature data.
        :type X: pandas.DataFrame
        :param y: Target data.
        :type y: pandas.DataFrame
        :return: Model score.
        :rtype: float
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, index=y.index)
        X_ = X.assign(
            **{
                self.duration_column: y[self.duration_column],
                self.event_column: y[self.event_column],
            }
        )
        return self.estimator.score(X_, scoring_method=self.scoring_method)
