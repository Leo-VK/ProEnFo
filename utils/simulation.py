from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import signal
import statsmodels.api as sm


class DataGenerator(ABC):
    @abstractmethod
    def generate_series(self, length: int) -> pd.Series:
        pass

    def generate_dataframe(self, length: int, offset_by_column: dict[str, list[float]]) -> pd.Series:
        data = {}
        for column in offset_by_column:
            data[column] = self.generate_series(length)
        return pd.DataFrame(data) + pd.DataFrame(offset_by_column)


class RandomDataGenerator(DataGenerator):
    def __init__(self,
                 left_bound: int = 0,
                 right_bound: int = 1,
                 offset: float = 0.,
                 seed: int = 0):
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.offset = offset
        self.seed = seed

    def generate_series(self, length: int) -> pd.Series:
        np.random.seed(self.seed)
        return pd.Series(np.random.randint(self.left_bound, self.right_bound, size=length) + self.offset,
                         name=f"sample([{self.left_bound},{self.right_bound}]) + offset")


class RandomWalkGenerator(DataGenerator):
    def __init__(self,
                 start_value: float = 0.,
                 offset: float = 0.,
                 seed: int = 0,
                 variant: Literal["unit", "normal"] = "unit"):
        self.start_value = start_value
        self.offset = offset
        self.seed = seed
        self.variant = variant

    def generate_series(self, length: int) -> pd.Series:
        np.random.seed(self.seed)
        if self.variant == "unit":
            steps = np.random.randint(low=0, high=2, size=length)
        elif self.variant == "normal":
            steps = np.random.normal(loc=0, scale=1, size=length)
        else:
            raise ValueError("Step variant unknown")

        series = pd.Series(steps).replace(0, -1)
        series.iloc[0] = self.start_value
        series.name = f"cumsum(sample([-1,1]))" if self.variant == "unit" else f"cumsum(sample([normal(0,1)]))"
        return series.cumsum() + self.offset


class TrendGenerator(DataGenerator):
    def __init__(self,
                 left_bound: int = 0,
                 right_bound: int = 1,
                 offset: float = 0):
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.offset = offset

    def generate_series(self, length: int) -> pd.Series:
        return pd.Series(np.linspace(self.left_bound, self.right_bound, length) + self.offset,
                         name=f"sample([{self.left_bound},{self.right_bound}]) + {self.offset}")


class TriangleSignalGenerator(DataGenerator):
    def __init__(self,
                 left_bound: int = 0,
                 right_bound: int = 1,
                 amplitude: float = 1.,
                 frequency: int = 5,
                 phase: float = 0.,
                 offset: float = 0.,
                 width: float = 0.5):
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset
        self.width = width

    def generate_series(self, length: int) -> pd.Series:
        time = np.linspace(self.left_bound, self.right_bound, length)
        values = self.amplitude * signal.sawtooth(2 * np.pi * self.frequency * time + self.phase,
                                                  self.width) + self.offset
        return pd.Series(values, index=time,
                         name=f"{self.amplitude}*saw(2*pi*{self.frequency}t+{self.phase}) + {self.offset}")


class SquareSignalGenerator(DataGenerator):
    def __init__(self,
                 left_bound: int = 0,
                 right_bound: int = 1,
                 amplitude: float = 1.,
                 frequency: int = 5,
                 phase: float = 0.,
                 offset: float = 0.,
                 duty: float = 0.5):
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset
        self.duty = duty

    def generate_series(self, length: int) -> pd.Series:
        time = np.linspace(self.left_bound, self.right_bound, length)
        values = self.amplitude * signal.square(2 * np.pi * self.frequency * time + self.phase, self.duty) + self.offset
        return pd.Series(values, index=time,
                         name=f"{self.amplitude}*rect(2*pi*{self.frequency}t+{self.phase}) + {self.offset}")


class HarmonicDataGenerator(DataGenerator):
    def __init__(self,
                 left_bound: int = 0,
                 right_bound: int = 1,
                 amplitude: float = 1.,
                 frequency: int = 5,
                 phase: float = 0.,
                 offset: float = 0.):
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

    def generate_series(self, length: int) -> pd.Series:
        time = np.linspace(self.left_bound, self.right_bound, length)
        values = self.amplitude * np.sin(2 * np.pi * self.frequency * time + self.phase) + self.offset
        return pd.Series(values, index=time,
                         name=f"{self.amplitude}*sin(2*pi*{self.frequency}t+{self.phase}) + {self.offset}")


class AutoRegressiveDataGenerator(DataGenerator):
    def __init__(self,
                 p_coefs: Optional[list[int]] = None,
                 q_coefs: Optional[list[int]] = None):
        self.p_coefs = p_coefs if p_coefs else [1]
        self.q_coefs = q_coefs if q_coefs else [1]

    def generate_series(self, length: int) -> pd.Series:
        # Insert required zero lag
        self.p_coefs.insert(0, 1)
        self.q_coefs.insert(0, 1)

        # Invert p-coefficients due to polynomial representation
        p_coefs = -np.array(self.p_coefs)
        q_coefs = np.array(self.q_coefs)
        return pd.Series(sm.tsa.ArmaProcess(p_coefs, q_coefs).generate_sample(length),
                         name=f"AR({len(p_coefs), len(q_coefs)})")


def add_noise(series: pd.Series, seed: int = 0) -> pd.Series:
    np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=1, size=len(series))
    series.name = f"{series.name} + noise(normal(0,1))"
    return series + noise


def fourier_series_approximation(series: pd.Series, degree: int) -> pd.Series:
    t = series.index.to_numpy() if series.index.is_numeric() else np.arange(len(series))
    T = np.max(t) - np.min(t) if series.index.is_numeric() else len(series)
    fourier_transform = np.fft.fft(series)
    a_0 = fourier_transform.real[0]
    a_k = 2 * fourier_transform.real[1:]
    b_k = -2 * fourier_transform.imag
    fourier_series = a_0
    for n in range(1, degree + 1):
        fourier_series += a_k[n] * np.cos(2 * np.pi * n * t / T) + b_k[n] * np.sin(2 * np.pi * n * t / T)
    return pd.Series(fourier_series / series.size, index=series.index, name="fourier_series")


class SwingingDoorCompression:
    def __init__(self, threshold: float = 1., use_datetime: bool = False):
        self.threshold = threshold
        self.lower_bound_min = np.inf
        self.upper_bound_max = -np.inf
        self.lower = None
        self.upper = None
        self.use_datetime = use_datetime

    def compress(self, series: pd.Series):
        dataframe = series.rename('value').to_frame().reset_index(names='time')
        old_index = dataframe['time']
        if isinstance(series.index, pd.DatetimeIndex):
            if self.use_datetime:
                dataframe['time'] = (dataframe['time'] - dataframe['time'].iloc[0]).dt.total_seconds()
            else:
                dataframe['time'] = np.arange(len(old_index))

        values = {}
        for i, row in dataframe.iterrows():
            values[i] = self._swing_door(row)

        compression = pd.DataFrame.from_dict(values, orient='index')
        compression['time'] = old_index
        return compression.set_index('time').squeeze()

    def _open_door(self, point: pd.Series):
        self.upper = point + [0, self.threshold]
        self.lower = point + [0, -self.threshold]
        if (point.time - self.upper.time) == 0 or (point.time - self.lower.time) == 0:
            self.upper_bound_max = -np.inf
            self.lower_bound_min = np.inf
        else:
            self.upper_bound_max = (point.value - self.upper.value) / (point.time - self.upper.time)
            self.lower_bound_min = (point.value - self.lower.value) / (point.time - self.lower.time)

    def _swing_door(self, point: pd.Series):
        if self.upper is None and self.lower is None:
            self._open_door(point)
            return point

        upper_bound = (point.value - self.upper.value) / (point.time - self.upper.time)
        lower_bound = (point.value - self.lower.value) / (point.time - self.lower.time)
        reset_flag = False
        if upper_bound > self.upper_bound_max:
            self.upper_bound_max = upper_bound
            if self.upper_bound_max >= self.lower_bound_min:
                reset_flag = True
        if lower_bound < self.lower_bound_min:
            self.lower_bound_min = lower_bound
            if self.upper_bound_max >= self.lower_bound_min:
                reset_flag = True
        if reset_flag:
            self._open_door(point)
            return point
        return pd.Series([point.time, np.nan], index=['time', 'value'])