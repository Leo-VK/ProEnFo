from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test_set(X: pd.DataFrame, target: str, train_ratio: float) -> tuple[Any, Any, Any, Any]:
    """Split pandas dataset into train/test features and targets"""
    X_train, X_test = train_test_split(X, train_size=train_ratio, shuffle=False)
    Y_train = X_train.pop(target)
    Y_test = X_test.pop(target)
    return X_train, X_test, Y_train, Y_test