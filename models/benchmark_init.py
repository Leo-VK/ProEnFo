from models import benchmark


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

    def __init__(self, quantiles: list[float]):
        super().__init__(
            model=benchmark.ConditionalErrorPersistence(quantiles=quantiles)
        )


class BEQ(Benchmark):
    """Expanding quantile model"""

    def __init__(self, quantiles: list[float]):
        super().__init__(
            model=benchmark.ExpandingQuantiles(quantiles=quantiles)
        )


class BMQ(Benchmark):
    """Moving quantile model"""

    def __init__(self, window_size: int, quantiles: list[float]):
        super().__init__(
            model=benchmark.MovingQuantiles(window_size=window_size, quantiles=quantiles)
        )
