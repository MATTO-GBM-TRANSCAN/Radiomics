import logging

from sklearn.feature_selection import VarianceThreshold as _VarianceThreshold

_logger = logging.getLogger(__name__)


class VarianceThreshold(_VarianceThreshold):
    """
    Transformer for removing features with low variance.

    :param threshold: Variance threshold for feature removal.
    :type threshold: float
    """

    def __init__(self, threshold: float):
        super().__init__(threshold=threshold)
        self.logged = False

    def fit(self, X, y=None):
        self.logged = False
        return super().fit(X, y)

    def transform(self, X):
        X_ = super().transform(X)
        if not self.logged:
            _logger.info(
                "Features after variance threshold %d/%d",
                X_.columns.size,  # type: ignore # X and X_ are pandas.DataFrame
                X.columns.size,  # type: ignore
            )
            self.logged = True
        return X_
