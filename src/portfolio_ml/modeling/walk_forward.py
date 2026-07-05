"""Walk-forward cross-validation splitter for time-series portfolio data.

Design decisions
----------------
* **No data leakage by construction**: train windows always end strictly before
  the start of the corresponding test window.  There is no overlap.
* **Two window modes**:
  - *expanding* (default): the training window grows with each fold, starting
    from the very first date in the dataset.  This mirrors how a real system
    would be deployed – it always uses all available history.
  - *rolling*: the training window slides forward at each fold with a fixed
    length of ``lookback_window`` periods.  Useful when older data is
    considered stale or structurally different.
* **Rebalance frequency** controls how far forward the test window advances
  at each fold (analogous to the portfolio rebalance period).

Example
-------
>>> import pandas as pd
>>> import numpy as np
>>> from portfolio_ml.modeling.walk_forward import WalkForwardSplitter
>>> dates = pd.date_range("2018-01-01", periods=1000, freq="B")
>>> splitter = WalkForwardSplitter(lookback_window=252, rebalance_freq=21)
>>> splits = splitter.split(dates)
>>> for fold_idx, (train_idx, test_idx) in enumerate(splits):
...     print(f"Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)

IndexPair = Tuple[np.ndarray, np.ndarray]


@dataclass
class WalkForwardSplitter:
    """Generate train/test index pairs using walk-forward methodology.

    Args:
        lookback_window: Minimum number of periods required in the training
            window before the first fold begins.  In *rolling* mode this is
            also the fixed training window size.
        rebalance_freq: Number of periods the test window advances between
            consecutive folds (i.e. how often the portfolio is rebalanced).
        window_type: ``"expanding"`` (default) or ``"rolling"``.
        test_window: Number of periods in each test window.  Defaults to
            ``rebalance_freq`` so test and rebalance periods coincide.

    Attributes:
        n_splits_: Set after calling :meth:`split` – the number of folds.

    Raises:
        ValueError: If parameter combination is invalid.
    """

    lookback_window: int
    rebalance_freq: int
    window_type: str = "expanding"
    test_window: int | None = None
    n_splits_: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lookback_window < 1:
            raise ValueError("lookback_window must be >= 1.")
        if self.rebalance_freq < 1:
            raise ValueError("rebalance_freq must be >= 1.")
        if self.window_type not in {"expanding", "rolling"}:
            raise ValueError("window_type must be 'expanding' or 'rolling'.")
        if self.test_window is None:
            object.__setattr__(self, "test_window", self.rebalance_freq)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def split(
        self,
        dates: pd.DatetimeIndex | pd.Series | pd.Index,
    ) -> List[IndexPair]:
        """Compute all (train_indices, test_indices) pairs.

        Args:
            dates: Ordered sequence of dates representing the full timeline.
                Duplicate dates are *not* expected; pass unique sorted dates.

        Returns:
            List of ``(train_idx, test_idx)`` tuples where each element is a
            1-D integer numpy array of positional indices into *dates*.

        Notes:
            Indices are **positional** (0-based), not label-based.  Use them
            with ``iloc`` on any DataFrame that shares the same row order.

        Example:
            >>> splits = splitter.split(dates)
            >>> for train_idx, test_idx in splits:
            ...     train_df = df.iloc[train_idx]
            ...     test_df  = df.iloc[test_idx]
        """
        dates_arr = np.asarray(dates)
        n = len(dates_arr)
        test_window = self.test_window or self.rebalance_freq

        if n < test_window:
            raise ValueError(
                f"Not enough periods ({n}) for even one test window of size {test_window}."
            )

        effective_lookback = min(self.lookback_window, max(1, n - test_window))
        if effective_lookback < self.lookback_window:
            logger.warning(
                "Requested lookback_window=%d exceeds available history (%d periods); "
                "using effective lookback=%d.",
                self.lookback_window,
                n,
                effective_lookback,
            )

        splits: List[IndexPair] = []
        test_start = effective_lookback  # first index that can be in a test set

        while test_start + test_window <= n:
            test_end = test_start + test_window  # exclusive

            if self.window_type == "expanding":
                train_start = 0
                train_end = test_start  # exclusive
            else:  # rolling
                train_end = test_start  # exclusive
                train_start = max(0, train_end - effective_lookback)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))
            test_start += self.rebalance_freq

        self.n_splits_ = len(splits)
        logger.info(
            "WalkForwardSplitter: %d folds | mode=%s | lookback=%d | rebalance=%d",
            self.n_splits_,
            self.window_type,
            effective_lookback,
            self.rebalance_freq,
        )
        return splits

    def iter_splits(
        self,
        dates: pd.DatetimeIndex | pd.Series | pd.Index,
    ) -> Generator[IndexPair, None, None]:
        """Lazy generator version of :meth:`split`.

        Yields ``(train_idx, test_idx)`` one fold at a time without materialising
        the full list, which is preferable when memory is constrained or the
        number of folds is very large.

        Args:
            dates: Same as :meth:`split`.

        Yields:
            ``(train_idx, test_idx)`` tuples.
        """
        for pair in self.split(dates):
            yield pair

    def get_fold_dates(
        self,
        dates: pd.DatetimeIndex | pd.Series,
    ) -> list[dict]:
        """Return human-readable fold date ranges for inspection / logging.

        Args:
            dates: Same as :meth:`split`.

        Returns:
            List of dicts with keys ``fold``, ``train_start``, ``train_end``,
            ``test_start``, ``test_end``.
        """
        dates_index = pd.DatetimeIndex(dates)
        info = []
        for fold, (train_idx, test_idx) in enumerate(self.split(dates_index)):
            info.append(
                {
                    "fold": fold,
                    "train_start": dates_index[train_idx[0]],
                    "train_end": dates_index[train_idx[-1]],
                    "test_start": dates_index[test_idx[0]],
                    "test_end": dates_index[test_idx[-1]],
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )
        return info
