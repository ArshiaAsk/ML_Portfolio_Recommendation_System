"""Train-only scaler abstraction for portfolio ML pipelines.

The core problem this module solves is **scaling leakage**: if a scaler (e.g.
``StandardScaler``) is fit on the full dataset including test rows, the model
receives implicit knowledge about the future distribution of features.  This
inflates in-sample performance and degrades real-world results.

Design
------
``TrainOnlyScaler`` wraps any scikit-learn–compatible scaler and enforces the
correct fit/transform protocol:

    1. ``fit(X_train)``  – statistics are learned *only* from training data.
    2. ``transform(X)``  – can be called on both train and test splits using
       the statistics learned in step 1.
    3. ``fit_transform(X_train)``  – convenience shorthand for step 1 + 2.
    4. ``inverse_transform(X)``  – delegates to the underlying scaler.

Calling ``transform`` before ``fit`` raises ``RuntimeError``.

Example
-------
>>> from sklearn.preprocessing import StandardScaler, MinMaxScaler
>>> from portfolio_ml.modeling.scalers import TrainOnlyScaler
>>> import numpy as np
>>> X_train = np.random.randn(200, 10)
>>> X_test  = np.random.randn(50, 10)
>>> scaler = TrainOnlyScaler(StandardScaler())
>>> X_train_scaled = scaler.fit_transform(X_train)
>>> X_test_scaled  = scaler.transform(X_test)   # uses train stats only
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


class TrainOnlyScaler:
    """Wrapper that enforces train-only fitting for any sklearn-compatible scaler.

    Args:
        scaler: An instantiated scikit-learn scaler such as ``StandardScaler()``
            or ``RobustScaler()``.  It must implement ``fit``, ``transform``,
            and ``inverse_transform``.

    Attributes:
        is_fitted_: ``True`` once :meth:`fit` or :meth:`fit_transform` has
            been called.
        feature_names_in_: List of column names (only set when input is a
            ``pd.DataFrame``).
    """

    def __init__(self, scaler: Any) -> None:
        self._scaler = scaler
        self.is_fitted_: bool = False
        self.feature_names_in_: list[str] | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray | pd.DataFrame) -> "TrainOnlyScaler":
        """Learn scaling statistics from training data only.

        Args:
            X_train: 2-D array or DataFrame of training features.
                Shape: ``(n_samples, n_features)``.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If ``X_train`` is 1-D.
        """
        X_arr = self._to_array(X_train, record_columns=True)
        if X_arr.ndim == 1:
            raise ValueError("TrainOnlyScaler expects a 2-D input (n_samples, n_features).")

        self._scaler.fit(X_arr)
        self.is_fitted_ = True
        logger.debug(
            "TrainOnlyScaler.fit: fitted %s on %d training samples, %d features.",
            type(self._scaler).__name__,
            X_arr.shape[0],
            X_arr.shape[1],
        )
        return self

    def transform(
        self, X: np.ndarray | pd.DataFrame, preserve_df: bool = True
    ) -> np.ndarray | pd.DataFrame:
        """Scale *X* using statistics learned during :meth:`fit`.

        Args:
            X: 2-D array or DataFrame to transform.
            preserve_df: If ``True`` and *X* is a ``pd.DataFrame``, the
                result is returned as a ``pd.DataFrame`` with the same
                index and columns.

        Returns:
            Scaled array or DataFrame.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._check_is_fitted()
        is_df = isinstance(X, pd.DataFrame)
        index = X.index if is_df else None
        columns = X.columns.tolist() if is_df else None

        X_arr = self._to_array(X)
        X_scaled = self._scaler.transform(X_arr)

        if preserve_df and is_df:
            return pd.DataFrame(X_scaled, index=index, columns=columns)
        return X_scaled

    def fit_transform(
        self, X_train: np.ndarray | pd.DataFrame, preserve_df: bool = True
    ) -> np.ndarray | pd.DataFrame:
        """Fit on *X_train* and return its scaled version.

        This is equivalent to calling :meth:`fit` then :meth:`transform` on
        the same data.  Never call this on the full dataset – only on training
        data.

        Args:
            X_train: Training features.
            preserve_df: Passed through to :meth:`transform`.

        Returns:
            Scaled training data.
        """
        return self.fit(X_train).transform(X_train, preserve_df=preserve_df)

    def inverse_transform(
        self, X: np.ndarray | pd.DataFrame, preserve_df: bool = True
    ) -> np.ndarray | pd.DataFrame:
        """Reverse the scaling transformation.

        Args:
            X: Scaled data to invert.
            preserve_df: Preserve DataFrame structure if input is a DataFrame.

        Returns:
            Data in original scale.
        """
        self._check_is_fitted()
        is_df = isinstance(X, pd.DataFrame)
        index = X.index if is_df else None
        columns = X.columns.tolist() if is_df else None

        X_arr = self._to_array(X)
        X_inv = self._scaler.inverse_transform(X_arr)

        if preserve_df and is_df:
            return pd.DataFrame(X_inv, index=index, columns=columns)
        return X_inv

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def scaler(self) -> Any:
        """The underlying scikit-learn scaler instance (read-only)."""
        return self._scaler

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "TrainOnlyScaler has not been fitted yet.  Call fit(X_train) "
                "or fit_transform(X_train) before transform()."
            )

    def _to_array(
        self, X: np.ndarray | pd.DataFrame, record_columns: bool = False
    ) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if record_columns:
                self.feature_names_in_ = X.columns.tolist()
            return X.values
        return np.asarray(X)

    def __repr__(self) -> str:  # pragma: no cover
        state = "fitted" if self.is_fitted_ else "unfitted"
        return f"TrainOnlyScaler({self._scaler!r}, {state})"


# ---------------------------------------------------------------------------
# Pre-configured factory helpers
# ---------------------------------------------------------------------------


def standard_scaler() -> TrainOnlyScaler:
    """Return a :class:`TrainOnlyScaler` wrapping ``sklearn.StandardScaler``."""
    from sklearn.preprocessing import StandardScaler  # lazy import

    return TrainOnlyScaler(StandardScaler())


def robust_scaler() -> TrainOnlyScaler:
    """Return a :class:`TrainOnlyScaler` wrapping ``sklearn.RobustScaler``."""
    from sklearn.preprocessing import RobustScaler  # lazy import

    return TrainOnlyScaler(RobustScaler())


def minmax_scaler(feature_range: tuple[float, float] = (0.0, 1.0)) -> TrainOnlyScaler:
    """Return a :class:`TrainOnlyScaler` wrapping ``sklearn.MinMaxScaler``.

    Args:
        feature_range: Desired range of scaled values.
    """
    from sklearn.preprocessing import MinMaxScaler  # lazy import

    return TrainOnlyScaler(MinMaxScaler(feature_range=feature_range))
