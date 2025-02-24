from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from typing import List, Tuple, Dict    
import re

# def reorder_columns(data: pd.DataFrame, target: str) -> pd.DataFrame:
#     def extract_column_info(col_name: str):
#         match = re.match(r"(\w+)(?:_t)?([\+-]?\d+)?", col_name)
#         if match:
#             variable_name, lag = match.groups()
#             variable_name = variable_name.rstrip("_t")
#             if lag:
#                 if '+' in lag:
#                     lag = int(lag.replace('+', ''))
#                 else:
#                     lag = int(lag)
#             else:
#                 lag = 0
#             return variable_name, lag
#         return None

#     def sorting_key(col_name: str):
#         variable_name, lag = extract_column_info(col_name)
#         return (variable_name, lag) if lag <= 0 else (variable_name, lag + 1)

#     sorted_columns = sorted(data.columns, key=sorting_key)
    
#     # 修改部分开始
#     target_columns = [col for col in sorted_columns if target in col]
#     non_target_columns = [col for col in sorted_columns if col not in target_columns]
#     sorted_columns = target_columns + non_target_columns
#     # 修改部分结束

#     reordered_data = data[sorted_columns]

#     return reordered_data

def reorder_columns(data: pd.DataFrame, target: List[str]) -> pd.DataFrame:
    def extract_column_info(col_name: str):
        match = re.match(r"(\w+)(?:_t)?([\+-]?\d+)?", col_name)
        if match:
            variable_name, lag = match.groups()
            variable_name = variable_name.rstrip("_t")
            if lag:
                if '+' in lag:
                    lag = int(lag.replace('+', ''))
                else:
                    lag = int(lag)
            else:
                lag = 0
            return variable_name, lag
        return None

    def sorting_key(col_name: str):
        variable_name, lag = extract_column_info(col_name)
        return (variable_name, lag) if lag <= 0 else (variable_name, lag + 1)

    sorted_columns = sorted(data.columns, key=sorting_key)
    
    # 修改部分开始
    target_columns = [col for col in sorted_columns if any(t in col for t in target)]
    non_target_columns = [col for col in sorted_columns if col not in target_columns]
    sorted_columns = target_columns + non_target_columns
    # 修改部分结束

    reordered_data = data[sorted_columns]

    return reordered_data

# def split_train_test_set(X: pd.DataFrame, target: str, train_ratio: float,pred_len:int) -> Tuple[Any, Any, Any, Any]:
#     """Split pandas dataset into train/test features and targets"""
#     X = reorder_columns(X)
#     X_train, X_test = train_test_split(X, train_size=train_ratio, shuffle=False)
#     X_test = X_test[pred_len:]


#     Y_train = X_train.pop(target)
#     Y_test = X_test.pop(target)


#     return X_train, X_test, Y_train, Y_test

# def split_train_test_set(X: pd.DataFrame, target: str, train_ratio: float, pred_len: int) -> Tuple[Any, Any, Any, Any]:
#     """Split pandas dataset into train/test features and targets"""
#     X = reorder_columns(X,target)
#     X_train, X_test = train_test_split(X, train_size=train_ratio, shuffle=False)
#     X_test = X_test[pred_len:]

#     # 修改部分开始
#     Y_train = pd.DataFrame()
#     Y_test = pd.DataFrame()
#     for i in range(pred_len + 1):
#         current_target = f"{target}_t+{i}" if i > 0 else target
#         Y_train[current_target] = X_train.pop(current_target)
#         Y_test[current_target] = X_test.pop(current_target)
#     # 修改部分结束

#     return X_train, X_test, Y_train, Y_test

def split_train_test_set(X: pd.DataFrame, target: List[str], train_ratio: float, pred_len: int) -> Tuple[Any, Any, Any, Any]:
    def check_chars_in_string(char_list, string):
            return any(char in string for char in char_list)
    # def sort_columns(ex_columns) -> pd.DataFrame:
    
    #     # 提取列名中的前缀和可能存在的后缀
    #     suffixes = [re.findall(r'(.*?)(?:_t([-+]\d+))?$', col)[0] for col in ex_columns]
    #     suffixes = [(x[0], int(x[1]) if x[1] else 0) for x in suffixes]

    #     # 使用后缀和前缀对列名进行排序，确保后缀的排序优先级高于前缀
    #     ex_columns_sorted = sorted(ex_columns, key=lambda x: (suffixes[ex_columns.index(x)][1], suffixes[ex_columns.index(x)][0]))

    #     return ex_columns_sorted
    def sort_columns(ex_columns) -> pd.DataFrame:
        # 提取列名中的前缀和可能存在的后缀
        suffixes = [re.findall(r'(.*?)(?:_t([-+]\d+))?$', col)[0] for col in ex_columns]
        suffixes = [(x[0], int(x[1]) if x[1] else 0) for x in suffixes]

        # 定义优先排序的关键字及其优先级
        priority_keywords = ['Day', 'Hour', 'Month', 'Weekday']
        priority_dict = {keyword: i for i, keyword in enumerate(priority_keywords)}

        # 使用后缀、优先关键字和前缀对列名进行排序，确保后缀的排序优先级最高，优先关键字次之，前缀最低
        ex_columns_sorted = sorted(ex_columns, key=lambda x: (suffixes[ex_columns.index(x)][1], 
                                                            priority_dict.get(suffixes[ex_columns.index(x)][0], len(priority_keywords)), 
                                                            suffixes[ex_columns.index(x)][0]))

        return ex_columns_sorted
    def reorder_X_columns(X, target):
        target_columns = [col for col in X.columns if check_chars_in_string(target, col)]
        ex_columns = [col for col in X.columns if col not in target_columns]
        target_columns = sort_columns(target_columns)
        ex_columns = sort_columns(ex_columns)

        new_order = target_columns + ex_columns
        # 重新排序列
        X = X[new_order]

        return X
    """Split pandas dataset into train/test features and targets"""
    X = reorder_columns(X,target)
    X_train, X_test = train_test_split(X, train_size=train_ratio, shuffle=False)
    X_test = X_test[pred_len:]

    # 修改部分开始
    Y_train = pd.DataFrame()
    Y_test = pd.DataFrame()
    for t in target:
        for i in range(pred_len + 1):
            current_target = f"{t}_t+{i}" if i > 0 else t
            Y_train[current_target] = X_train.pop(current_target)
            Y_test[current_target] = X_test.pop(current_target)
    # 修改部分结束
    X_train = reorder_X_columns(X_train,target)
    X_test = reorder_X_columns(X_test,target)
    Y_train = Y_train[sort_columns(Y_train.columns.tolist())]
    Y_test = Y_test[sort_columns(Y_test.columns.tolist())]
    return X_train, X_test, Y_train, Y_test

