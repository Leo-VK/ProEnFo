import numpy as np


class TrainTestSplit:
    """Simple train/test split for cross validation"""
    def __init__(self, test_ratio=0.2):
        self.test_ratio = test_ratio

    def split(self, X, y=None, groups=None) -> tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        n_test = np.ceil(self.test_ratio * n_samples)
        n_train = int(n_samples - n_test)
        indices = np.arange(n_samples)

        yield indices[:n_train], indices[n_train:]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return 1


