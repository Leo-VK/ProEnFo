import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    """Timer class for performance timing
    Adopted from (https://realpython.com/python-timer/)"""
    def __init__(self, verbose: int = 0):
        self._start_time = None
        self.verbose = verbose

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        if self.verbose > 0:
            print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return elapsed_time
