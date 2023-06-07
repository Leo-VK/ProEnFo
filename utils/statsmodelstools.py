"""
Adopted from development version of statsmodels

Author: Kishan Manani
License: BSD-3 Clause

An implementation of MSTL [1], an algorithm for time series decomposition when
there are multiple seasonal components.

This implementation has the following differences with the original algorithm:
- Missing data must be handled outside of this class.
- The algorithm proposed in the paper handles a case when there is no
seasonality. This implementation assumes that there is at least one seasonal
component.

[1] K. Bandura, R.J. Hyndman, and C. Bergmeir (2021)
MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple
Seasonal Patterns
https://arxiv.org/pdf/2107.13462.pdf
"""
from typing import Dict, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from scipy.stats import boxcox

from statsmodels.tsa._stl import STL

def freq_to_period(freq: Union[str, offsets.DateOffset]) -> int:
    """
    Convert a pandas frequency to a periodicity
    Parameters
    ----------
    freq : str or offset
        Frequency to convert
    Returns
    -------
    int
        Periodicity of freq
    Notes
    -----
    Annual maps to 1, quarterly maps to 4, monthly to 12, weekly to 52.
    """
    if not isinstance(freq, offsets.DateOffset):
        freq = to_offset(freq)  # go ahead and standardize
    assert isinstance(freq, offsets.DateOffset)
    freq = freq.rule_code.upper()

    if freq == "A" or freq.startswith(("A-", "AS-")):
        return 1
    elif freq == "Q" or freq.startswith(("Q-", "QS-")):
        return 4
    elif freq == "M" or freq.startswith(("M-", "MS")):
        return 12
    elif freq == "W" or freq.startswith("W-"):
        return 52
    elif freq == "D":
        return 7
    elif freq == "B":
        return 5
    elif freq == "H":
        return 24
    else:  # pragma : no cover
        raise ValueError(
            "freq {} not understood. Please report if you "
            "think this is in error.".format(freq)
        )


class MSTL:
    """
    MSTL(endog, periods=None, windows=None, lmbda=None, iterate=2,
         stl_kwargs=None)

    Season-Trend decomposition using LOESS for multiple seasonalities.

    Parameters
    ----------
    endog : array_like
        Data to be decomposed. Must be squeezable to 1-d.
    periods : {int, array_like, None}, optional
        Periodicity of the seasonal components. If None and endog is a pandas
        Series or DataFrame, attempts to determine from endog. If endog is a
        ndarray, periods must be provided.
    windows : {int, array_like, None}, optional
        Length of the seasonal smoothers for each corresponding period.
        Must be an odd integer, and should normally be >= 7 (default). If None
        then default values determined using 7 + 4 * np.arange(1, n + 1, 1)
        where n is number of seasonal components.
    lmbda : {float, str, None}, optional
        The lambda parameter for the Box-Cox transform to be applied to `endog`
        prior to decomposition. If None, no transform is applied. If "auto", a
        value will be estimated that maximizes the log-likelihood function.
    iterate : int, optional
        Number of iterations to use to refine the seasonal component.
    stl_kwargs: dict, optional
        Arguments to pass to STL.

    See Also
    --------
    """

    def __init__(
        self,
        endog,
        *,
        periods: Optional[Union[int, Sequence[int]]] = None,
        windows: Optional[Union[int, Sequence[int]]] = None,
        lmbda: Optional[Union[float, str]] = None,
        iterate: int = 2,
        stl_kwargs: Optional[Dict[str, Union[int, bool, None]]] = None,
    ):
        self.endog = endog
        self._y = self._to_1d_array(endog)
        self.nobs = self._y.shape[0]
        self.lmbda = lmbda
        self.periods, self.windows = self._process_periods_and_windows(
            periods, windows
        )
        self.iterate = iterate
        self._stl_kwargs = self._remove_overloaded_stl_kwargs(
            stl_kwargs if stl_kwargs else {}
        )

    def fit(self):
        """
        Estimate a trend component, multiple seasonal components, and a
        residual component.

        Returns
        -------
        DecomposeResult
            Estimation results.
        """
        num_seasons = len(self.periods)
        iterate = 1 if num_seasons == 1 else self.iterate

        # Box Cox
        if self.lmbda == "auto":
            y, lmbda = boxcox(self._y, lmbda=None)
            self.est_lmbda = lmbda
        elif self.lmbda:
            y = boxcox(self._y, lmbda=self.lmbda)
        else:
            y = self._y

        # Get STL fit params
        stl_inner_iter = self._stl_kwargs.pop("inner_iter", None)
        stl_outer_iter = self._stl_kwargs.pop("outer_iter", None)

        # Iterate over each seasonal component to extract seasonalities
        seasonal = np.zeros(shape=(num_seasons, self.nobs))
        deseas = y
        for _ in range(iterate):
            for i in range(num_seasons):
                deseas = deseas + seasonal[i]
                res = STL(
                    endog=deseas,
                    period=self.periods[i],
                    seasonal=self.windows[i],
                    **self._stl_kwargs,
                ).fit(inner_iter=stl_inner_iter, outer_iter=stl_outer_iter)
                seasonal[i] = res.seasonal
                deseas = deseas - seasonal[i]

        seasonal = np.squeeze(seasonal.T)
        trend = res.trend
        rw = res.weights
        resid = deseas - trend

        # Return pandas if endog is pandas
        if isinstance(self.endog, (pd.Series, pd.DataFrame)):
            index = self.endog.index
            y = pd.Series(y, index=index, name="observed")
            trend = pd.Series(trend, index=index, name="trend")
            resid = pd.Series(resid, index=index, name="resid")
            rw = pd.Series(rw, index=index, name="robust_weight")
            cols = [f"seasonal_{period}" for period in self.periods]
            if seasonal.ndim == 1:
                seasonal = pd.Series(seasonal, index=index, name="seasonal")
            else:
                seasonal = pd.DataFrame(seasonal, index=index, columns=cols)

        # Avoid circular imports
        from statsmodels.tsa.seasonal import DecomposeResult

        return DecomposeResult(y, seasonal, trend, resid, rw)


    def __str__(self):
        return (
            "MSTL(endog,"
            f" periods={self.periods},"
            f" windows={self.windows},"
            f" lmbda={self.lmbda},"
            f" iterate={self.iterate})"
        )

    def _process_periods_and_windows(
        self,
        periods: Union[int, Sequence[int], None],
        windows: Union[int, Sequence[int], None],
    ) -> Tuple[Sequence[int], Sequence[int]]:
        periods = self._process_periods(periods)

        if windows:
            windows = self._process_windows(windows, num_seasons=len(periods))
            periods, windows = self._sort_periods_and_windows(periods, windows)
        else:
            windows = self._process_windows(windows, num_seasons=len(periods))
            periods = sorted(periods)

        if len(periods) != len(windows):
            raise ValueError("Periods and windows must have same length")

        # Remove long periods from decomposition
        if any(period >= self.nobs / 2 for period in periods):
            warnings.warn(
                "A period(s) is larger than half the length of time series."
                " Removing these period(s)."
            )
            periods = tuple(
                period for period in periods if period < self.nobs / 2
            )
            windows = windows[: len(periods)]

        return periods, windows

    def _process_periods(
        self, periods: Union[int, Sequence[int], None]
    ) -> Sequence[int]:
        if periods is None:
            periods = (self._infer_period(),)
        elif isinstance(periods, int):
            periods = (periods,)
        else:
            pass
        return periods

    def _process_windows(
        self,
        windows: Union[int, Sequence[int], None],
        num_seasons: int,
    ) -> Sequence[int]:
        if windows is None:
            windows = self._default_seasonal_windows(num_seasons)
        elif isinstance(windows, int):
            windows = (windows,)
        else:
            pass
        return windows

    def _infer_period(self) -> int:
        freq = None
        if isinstance(self.endog, (pd.Series, pd.DataFrame)):
            freq = getattr(self.endog.index, "inferred_freq", None)
        if freq is None:
            raise ValueError("Unable to determine period from endog")
        period = freq_to_period(freq)
        return period

    @staticmethod
    def _sort_periods_and_windows(
        periods, windows
    ) -> Tuple[Sequence[int], Sequence[int]]:
        if len(periods) != len(windows):
            raise ValueError("Periods and windows must have same length")
        periods, windows = zip(*sorted(zip(periods, windows)))
        return periods, windows

    @staticmethod
    def _remove_overloaded_stl_kwargs(stl_kwargs: Dict) -> Dict:
        args = ["endog", "period", "seasonal"]
        for arg in args:
            stl_kwargs.pop(arg, None)
        return stl_kwargs

    @staticmethod
    def _default_seasonal_windows(n: int) -> Sequence[int]:
        return tuple(7 + 4 * i for i in range(1, n + 1))  # See [1]

    @staticmethod
    def _to_1d_array(x):
        y = np.ascontiguousarray(np.squeeze(np.asarray(x)), dtype=np.double)
        if y.ndim != 1:
            raise ValueError("y must be a 1d array")
        return y