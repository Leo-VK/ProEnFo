from models import benchmark

from typing import List

class Benchmark:
    """Class representing a benchmark model"""

    def __init__(self, model):
        self.model = model
        self.name = self.__class__.__name__


class BP(Benchmark):
    """Persistence model"""

    def __init__(self):
        super().__init__(
            model=benchmark.Persistence()
        )


class BC(Benchmark):
    """Climatology model"""

    def __init__(self):
        super().__init__(
            model=benchmark.Climatology()
        )


class BMA(Benchmark):
    """Moving average model"""

    def __init__(self, window_size: int):
        super().__init__(
            model=benchmark.MovingAverage(window_size=window_size)
        )


class BCEP(Benchmark):
    """Conditional error quantile persistence model"""

    def __init__(self, quantiles: List[float]):
        super().__init__(
            model=benchmark.ConditionalErrorPersistence(quantiles=quantiles)
        )
class BCEARIMA(Benchmark):
    """Conditional error quantile persistence model"""

    def __init__(self, autoregressive_order:int, differencing_order:int, moving_average_order:int, quantiles: List[float]):
        super().__init__(
            model=benchmark.ConditionalErrorARIMA(autoregressive_order = autoregressive_order,
                                                  differencing_order = differencing_order,
                                                  moving_average_order = moving_average_order,
                                                  quantiles=quantiles)
        )


class BEQ(Benchmark):
    """Expanding quantile model"""

    def __init__(self, quantiles: List[float]):
        super().__init__(
            model=benchmark.ExpandingQuantiles(quantiles=quantiles)
        )


class BMQ(Benchmark):
    """Moving quantile model"""

    def __init__(self, window_size: int, quantiles: List[float]):
        super().__init__(
            model=benchmark.MovingQuantiles(window_size=window_size, quantiles=quantiles)
        )

class ARIMA(Benchmark):
    """Conditional error quantile persistence model"""

    def __init__(self, autoregressive_order:int, differencing_order:int, moving_average_order:int):
        super().__init__(
            model=benchmark.ExpandingARIMA(autoregressive_order = autoregressive_order,
                                                  differencing_order = differencing_order,
                                                  moving_average_order = moving_average_order,
                                                  )
        )


