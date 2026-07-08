"""
lgbm_quant_pipeline_explained.py

reconstructed from the user's notebook screenshots.

what this file is:
- a combined, cleaned python script version of the notebook workflow
- chinese comments translated into english
- heavier explanations added throughout
- sections reorganized so the overall pipeline is easier to follow

important honesty:
1. some very long lines in the screenshots were horizontally truncated, especially
   the full feature list. i preserved the visible part and marked the missing part
   with TODO comments.
2. some filesystem paths are environment-specific to the original notebook setup.
   i kept the visible paths and marked them as project-specific.
3. this script is meant to document the workflow clearly. it may need small edits
   before it runs in a different environment.

big picture:
this is a factor-data pipeline for quant research using lightgbm. it:
- loads factor files and label files from parquet
- merges them by timestamp
- builds train / validation splits by date and stock universe
- converts a continuous return-like label into classification-style buckets
- balances the label distribution
- trains a lightgbm model in incremental rounds
- saves model checkpoints, prediction distribution statistics, thresholds, and feature importance
"""

# ---------------------------------------------------------------------
# initialization / imports
# ---------------------------------------------------------------------

import os
import sys
import time
import gc
import json
import random
import pickle
import importlib
import fcntl
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
import lightgbm as lgb

from numba import jit, typed
from tqdm.contrib.concurrent import process_map
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# project-specific internal dependency from the original notebook
# it appeared in the screenshots, but may not exist outside the user's environment
from dw_data.fastpai import getAPI, getlS, getOrig, gkts


# ---------------------------------------------------------------------
# utility helpers
# ---------------------------------------------------------------------

def get_absolute_paths(directory: str):
    """
    recursively walk through a directory and return a list containing the
    absolute paths of all subfolders and files.

    this appeared near the top of the original notebook and is used to collect
    all parquet files under the factor data directory.
    """
    absolute_paths = []

    for root, dirs, files in os.walk(directory):
        for name in dirs:
            absolute_paths.append(os.path.join(root, name))
        for name in files:
            absolute_paths.append(os.path.join(root, name))

    return absolute_paths


def read_parquet_file(file_path, read_columns=None):
    """
    read a parquet file into a pandas dataframe.

    parameters
    ----------
    file_path : str
        path to the parquet file
    read_columns : list[str] | None
        if provided, only these columns are loaded

    returns
    -------
    pd.DataFrame
    """
    table = pq.read_table(file_path, columns=read_columns)
    df = table.to_pandas()
    return df


def read_data(task):
    """
    read one (date, code) task:
    - load the factor parquet
    - load the label parquet
    - merge them by time
    - fill missing label columns with 0
    - drop the raw time column from the label table
    - attach stock code
    - optionally filter out extreme labels

    the original notebook used global variables such as:
    - feature_ls
    - label
    so this function preserves that design.

    expected task format
    --------------------
    task = (date, code)
    """
    date, code = task

    # project-specific paths visible in the screenshots
    factor_path_1 = f"217/data1/base_factor/{date}/{code}.parquet"
    label_path_1 = f"217/data1/base_label/{date}/{code}.parquet"

    # check whether both files exist
    if not os.path.exists(factor_path_1):
        print(f"factor file does not exist: {factor_path_1}")
        return pd.DataFrame()

    if not os.path.exists(label_path_1):
        print(f"label file does not exist: {label_path_1}")
        return pd.DataFrame()

    try:
        # load factor columns plus the timestamp column needed for the merge
        factor_df = read_parquet_file(
            factor_path_1,
            read_columns=feature_ls + ["close_time"],
        )

        # load full label file
        label_df = read_parquet_file(label_path_1)

        # merge factor dataframe and label dataframe by time
        merged_df = pd.merge(
            factor_df,
            label_df,
            how="left",
            left_on="close_time",
            right_on="Time",
        )

        # identify label columns excluding the timestamp column
        label_columns = label_df.columns.difference(["Time"])

        # fill missing label values with 0
        merged_df[label_columns] = merged_df[label_columns].fillna(0)

        # drop the raw label-side time column
        final_df = merged_df.drop(columns=["Time"])

        # fill again after drop, matching the original notebook logic
        final_df[label_columns] = final_df[label_columns].fillna(0)

        # keep the stock code
        final_df["code"] = code

        # filter out extreme labels
        # note: 'label' is a global string set later in the notebook
        final_df = final_df[final_df[label].abs() < 0.2]

        return final_df

    except Exception as e:
        print(f"{code} in {date}: error occurred while processing the task: {e}")
        return pd.DataFrame()


def parallel_read_data(tasks, max_workers=4, batch_size=2000):
    """
    read many (date, code) tasks in parallel and concatenate the results.

    why batching exists:
    --------------------
    loading everything in one giant process_map call may be too memory-heavy.
    batching gives more control over memory pressure.
    """
    results = []

    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        batch_results = process_map(
            read_data,
            batch_tasks,
            max_workers=max_workers,
        )
        results.append(pd.concat(list(batch_results), axis=0))
        del batch_results

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def random_subset(X_train, y_train, frac=0.2):
    """
    sample a random subset of the training set.

    this is used in the notebook to carry forward a fraction of the previous
    round's training data into the next incremental training round.
    """
    np.random.seed(42)

    subset_size = int(X_train.shape[0] * frac)
    subset_indices = np.random.choice(
        X_train.shape[0],
        size=subset_size,
        replace=False,
    )

    old_subset_X = X_train[subset_indices, :]
    old_subset_y = y_train[subset_indices]
    return old_subset_X, old_subset_y


# ---------------------------------------------------------------------
# label construction / balancing
# ---------------------------------------------------------------------

def create_labels(series, thresholds):
    """
    create signed bucket labels based on absolute-value thresholds.

    example idea:
    -------------
    if thresholds = [0.001, 0.002, 0.003]
    then:
    - tiny moves close to 0 stay near class 0
    - larger positive moves become +1, +2, +3, ...
    - larger negative moves become -1, -2, -3, ...

    this converts a continuous target into an ordinal/signed label space.

    parameters
    ----------
    series : pd.Series
        continuous label series
    thresholds : list[float]
        bucket boundaries on absolute value

    returns
    -------
    pd.Series
        signed discrete labels
    """
    thresholds = sorted(thresholds)
    abs_series = series.abs()

    # add lower and upper bounds
    extended_thresholds = [0.0] + thresholds + [np.inf]

    # find bucket indices
    indices = np.searchsorted(
        extended_thresholds,
        abs_series,
        side="right",
    ) - 1

    # assign sign back to the bucket index
    labels = indices * np.sign(series)

    return pd.Series(labels, index=series.index)


def downsample_zero_label(labels, non_zero_count, miltipile, random_state=42):
    """
    downsample class 0 examples based on the specified multiple.

    note:
    -----
    the original notebook spelled this variable as 'miltipile', not 'multiple'.
    i preserved the original spelling to stay close to the screenshots.

    logic:
    ------
    if class 0 is massively overrepresented, keep only a controlled number of
    zero-label samples relative to the total number of non-zero samples.
    """
    np.random.seed(random_state)

    zero_indices = labels[labels == 0].index
    target_zero_count = int(non_zero_count * miltipile)

    if len(zero_indices) > target_zero_count:
        sampled_zero_indices = np.random.choice(
            zero_indices,
            size=target_zero_count,
            replace=False,
        )
    else:
        sampled_zero_indices = zero_indices

    return sampled_zero_indices


def balance_labels(labels, random_state=42):
    """
    balance positive and negative samples for each absolute label level.

    example:
    --------
    if you have +3 and -3, keep the same count from each side.
    if +2 has 1000 and -2 has 600, sample 600 from the positive side too.

    class 0 is excluded here and handled separately by downsampling.
    """
    np.random.seed(random_state)
    sampled_indices = []

    # get all non-zero absolute labels, sorted
    abs_labels = sorted(
        set(abs(label) for label in labels.unique() if label != 0)
    )

    for abs_label in abs_labels:
        positive_label = abs_label
        negative_label = -abs_label

        positive_indices = labels[labels == positive_label].index
        negative_indices = labels[labels == negative_label].index

        positive_count = len(positive_indices)
        negative_count = len(negative_indices)

        min_count = min(positive_count, negative_count)

        if positive_count > min_count:
            sampled_positive_indices = np.random.choice(
                positive_indices,
                size=min_count,
                replace=False,
            )
        else:
            sampled_positive_indices = positive_indices

        if negative_count > min_count:
            sampled_negative_indices = np.random.choice(
                negative_indices,
                size=min_count,
                replace=False,
            )
        else:
            sampled_negative_indices = negative_indices

        sampled_indices.extend(sampled_positive_indices)
        sampled_indices.extend(sampled_negative_indices)

    return sampled_indices


def get_sampled_indices(series, thresholds, miltipile, random_state=42):
    """
    get final sampled indices after:
    1. turning the continuous series into signed bucket labels
    2. downsampling label 0
    3. balancing positive and negative classes at each absolute bucket level
    """
    labels = create_labels(series, thresholds)

    non_zero_count = len(labels[labels != 0])

    sampled_zero_indices = downsample_zero_label(
        labels,
        non_zero_count,
        miltipile,
        random_state,
    )

    balanced_indices = balance_labels(labels, random_state)

    sampled_indices = list(sampled_zero_indices) + balanced_indices
    return sampled_indices


# ---------------------------------------------------------------------
# task discovery / split construction
# ---------------------------------------------------------------------

# example data root shown in the notebook
data_main_path = "shareData/"

# collect all base factor parquet files
directory_to_search = f"/home/jupyterData/{data_main_path}/data1/base_factor/"

result = get_absolute_paths(directory_to_search)
result_ls = [path for path in result if "parquet" in path]

task_ls = []
for file in result_ls:
    file_splite = file.split("/")
    date = file_splite[-2]
    code = file_splite[-1][:6]
    task_ls.append((date, code))

print(len(task_ls), task_ls[0])

# build ordered trading day list
trade_days_list = list(set([task[0] for task in task_ls]))
trade_days_list.sort()
print(trade_days_list[0], trade_days_list[-1])

# load stock universe
# original notebook path (visible in screenshot):
# /home/jupyterData/nfsData/176/data2/zlb/code_ls.json
with open("/home/jupyterData/nfsData/176/data2/zlb/code_ls.json", "r") as file:
    code_ls = json.load(file)

# flatten while excluding key '000688'
code_list = [
    value
    for key, item in code_ls.items()
    for value in item
    if key != "000688"
]
code_set = set(code_list)
print(f"共计{len(code_set)}只股票参与训练和验证")

# train / validation split
train_start = "2024-09-01"
train_end = "2024-12-13"
valid_start = "2024-12-13"
valid_end = "2025-01-14"
remove_date = ["2024-09-30", "2024-10-08"]

train_set = [
    task for task in task_ls
    if train_start <= task[0] <= train_end
    and task[0] not in remove_date
    and task[1] in code_set
]
print(len(train_set))

valid_set = [
    task for task in task_ls
    if valid_start <= task[0] <= valid_end
    and task[1] in code_set
]
print(len(valid_set))


# ---------------------------------------------------------------------
# label processing function used before training
# ---------------------------------------------------------------------

# the original notebook used this value as:
# "control the number of zero-label samples as a multiple of non-zero samples"
miltipile = 1

def process_label(train_data, train_label):
    """
    apply the label sampling pipeline to a training dataframe:
    - use the continuous target column stored in the global variable `label`
    - compute sampled indices
    - convert the sampled rows into discrete labels
    - write them into `train_label`
    - return only sampled rows
    """
    series = train_data[label]
    sampled_indices = get_sampled_indices(series, thresholds, miltipile)
    balanced_labels = create_labels(train_data.loc[sampled_indices, label], thresholds)
    train_data.loc[sampled_indices, train_label] = balanced_labels
    return train_data.loc[sampled_indices, :]


# ---------------------------------------------------------------------
# global training configuration
# ---------------------------------------------------------------------

# factor data save root visible in screenshot
path_data = "nfsData/2176"
# alternative shown in comment in the notebook:
# path_data = "shareData"

# model save paths visible in screenshot, but environment-specific
path_save = [
    f"/home/jupyterData/{path_data}/data2/yr/yr_model",
    "/home/jupyterData/nfsData/217/data1/zlb/vr_model/",
]

global label, feature_ls

# target choice from screenshot
label_name = 30

# IMPORTANT:
# the feature list line in the screenshot was horizontally truncated.
# only the visible leading features are preserved here.
# you will need to add the rest from the original notebook/source file.
feature_ls = [
    "take_time",
    "avgtrp",
    "amt20amt_mean",
    "count20ct_mean",
    "count_c20ct_mean",
    "amt_c20amt_mean",
    "oct_mean2atick",
    "oamt_mean2aamt",
    "BuyCancelPrice_mean",
    "BuyCancelPrice_std",
    # TODO: the screenshot cuts off the remainder of this very long list
]

print(f"共有特征{len(feature_ls)}")

base_model_name = f"lgbm_zl_all_1{label_name}s"
label = f"mid1hlr1#{label_name}"
train_label = "label"

feature_ls_dt = {}
for idx, feature in enumerate(feature_ls):
    feature_ls_dt[feature] = idx

num_train_ts = 6000
num_valid_ts = 6000

# lightgbm parameters from the notebook
params = {
    "boosting_type": "gbdt",      # gradient boosting decision trees
    "objective": "regression",    # regression task
    "metric": "rmse",             # monitor RMSE
    "num_leaves": 128,            # more leaves -> potentially higher fit capacity
    "max_depth": -1,              # no manual depth cap
    "learning_rate": 0.08,        # smaller LR usually needs more rounds
    "feature_fraction": 0.8,      # use 80% of features each boosting round
    "bagging_fraction": 0.8,      # use 80% of rows in bagging rounds
    "bagging_freq": 5,            # perform bagging every 5 rounds
    "verbose": 1,
    "num_threads": 150,
    # "device": "gpu",
    # "gpu_platform_id": 0,
    # "gpu_device_id": 0,
}

total_round = np.ceil(len(train_set) / num_train_ts).astype("int")
print(f"总共要跑的轮数:{total_round}")


# ---------------------------------------------------------------------
# version 1: fixed thresholds
# ---------------------------------------------------------------------

def run_fixed_threshold_training():
    """
    training version based on a fixed manually specified threshold list.
    """
    global thresholds

    thresholds = [0.001, 0.002, 0.003, 0.0045, 0.006, 0.009]

    # quantiles to save from training predictions
    save_quantile = [0.001, 0.01, 0.05, 0.1, 0.2, 0.25]

    # randomly draw validation tasks
    valid_files = random.sample(valid_set, num_valid_ts)
    valid_data = parallel_read_data(
        valid_files,
        max_workers=10,
        batch_size=len(valid_files),
    )

    # create classification-style labels for validation
    valid_data[train_label] = create_labels(valid_data[label], thresholds)

    X_test = valid_data.loc[:, feature_ls].astype(np.float32).values
    y_test = valid_data.loc[:, train_label].values
    del valid_data

    print("测试集:", X_test.shape)

    for rnd in range(0, 5):
        print(f"正在训练第{rnd}轮")

        train_files = random.sample(train_set, len(train_set))
        train_batch = train_files[rnd * num_train_ts:(rnd + 1) * num_train_ts]

        train_data = parallel_read_data(
            train_batch,
            max_workers=10,
            batch_size=num_train_ts,
        )
        train_data = process_label(train_data, train_label=train_label)

        print(f"train_data: {train_data.shape}")
        print("剔除极端标签后的train_data", train_data.shape)

        X_train = train_data.loc[:, feature_ls].values.astype(np.float32, copy=False)
        y_train = train_data.loc[:, train_label].values

        print("训练集:", X_train.shape, "验证集:", X_test.shape)

        del train_data
        gc.collect()

        # shuffle training set
        shuffled_indices = np.random.permutation(len(X_train))
        X_train = X_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        print("数据准备完成")

        # custom callback to print validation IC
        def log_ic_score(env):
            if env.evaluation_result_list:
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    if data_name == "valid" and eval_name == "rmse" and env.iteration % 50 == 0:
                        y_pred_cb = env.model.predict(X_test, num_iteration=env.iteration)
                        ic, _ = pearsonr(y_test, y_pred_cb)
                        print(f"[{env.iteration}] valid's IC: {ic}")

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
            log_ic_score,
        ]

        if rnd == 0:
            lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
            print(f"lgb_train: X_shape {X_train.shape} y_shape {y_train.shape}")

            gbm = lgb.train(
                params,
                lgb_train,
                num_boost_round=500,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=["train", "valid"],
                callbacks=callbacks,
                keep_training_booster=True,
            )
        else:
            # merge in 20% of last round's training data
            X_train_add = np.vstack([old_subset_X, X_train])
            y_train_add = np.concatenate([old_subset_y, y_train])

            lgb_train = lgb.Dataset(X_train_add, y_train_add, free_raw_data=False)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
            print(f"lgb_train: X_shape {X_train_add.shape} y_shape {y_train_add.shape}")

            # slightly decay learning rate, but not below 0.01
            params["learning_rate"] = max(params["learning_rate"] - 0.005, 0.01)

            # load previous round model
            last_gbm = lgb.Booster(
                model_file=os.path.join(
                    path_save[0],
                    f"{base_model_name}_round{rnd-1}",
                    f"{base_model_name}_round{rnd-1}.txt",
                )
            )

            gbm = lgb.train(
                params,
                lgb_train,
                num_boost_round=300,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=["train", "valid"],
                callbacks=callbacks,
                keep_training_booster=True,
                init_model=last_gbm,
            )

            del X_train_add, y_train_add

        # validation prediction
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"The RMSE of prediction is: {rmse}")

        ic, _ = pearsonr(y_test, y_pred)
        print(f"The IC (Information Coefficient) of prediction is: {ic}")

        # training prediction stats
        y_pred_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
        pred_stats = {
            "mean": y_pred_train.mean(),
            "std": y_pred_train.std(),
        }

        if rnd == 0:
            dict_quantile = {f"{q}": np.quantile(y_pred_train, q) for q in save_quantile}
            print("dict_quantile:", {f"{q}": np.quantile(y_pred_train, q) for q in save_quantile})

        model_name = f"{base_model_name}_round{rnd}"

        for single_path in path_save:
            path_model_save = os.path.join(single_path, f"{model_name}")
            if not os.path.exists(path_model_save):
                os.makedirs(path_model_save, exist_ok=True)

            path_info = os.path.join(path_model_save, f"{model_name}.pkl")
            path_model = os.path.join(path_model_save, f"{model_name}.txt")
            path_json = os.path.join(path_model_save, "X_stats.json")

            # save model metadata
            with open(path_info, "wb") as f:
                pickle.dump(
                    {
                        "model": gbm,
                        "feature_ls": feature_ls,
                        "feature_stats": [],
                        "pred_stats": pred_stats,
                        "quantile": dict_quantile,
                    },
                    f,
                )

            # save plain text model
            gbm.save_model(path_model, num_iteration=gbm.best_iteration)

            # save JSON stats
            data_to_save = {model_name: pred_stats}
            with open(path_json, "w") as json_file:
                json.dump(data_to_save, json_file, indent=4)

            print("数据已成功保存为 pkl 文件 和 JSON 文件。")

        # keep 20% of this round's training set for the next round
        old_subset_X, old_subset_y = random_subset(X_train, y_train, frac=0.2)

        del lgb_train, lgb_eval, y_pred_train, y_pred, gbm
        gc.collect()


# ---------------------------------------------------------------------
# version 2: quantile-derived thresholds
# ---------------------------------------------------------------------

def run_quantile_threshold_training():
    """
    training version where thresholds are inferred from label quantiles
    during the first round, then reused later.

    the notebook calls this:
    "分位数阈值版本" (quantile threshold version)
    """
    global thresholds

    # quantiles used to generate threshold buckets
    q_list = [0.01, 0.03, 0.06, 0.12, 0.2, 0.3, 0.7, 0.8, 0.88, 0.94, 0.97, 0.99]

    # quantiles saved for model output distribution tracking
    save_quantile = [0.001, 0.01, 0.05, 0.1, 0.2, 0.25]
    save_quantile = [item for s in save_quantile for item in (1 - s, s)]

    is_read_valid = False
    is_first_train = True

    for rnd in range(0, 5):
        print(f"正在训练第{rnd}轮")

        train_batch = train_files[rnd * num_train_ts:(rnd + 1) * num_train_ts]
        train_data = parallel_read_data(
            train_batch,
            max_workers=30,
            batch_size=len(train_batch),
        )

        print(f"train_data: {train_data.shape}")

        if rnd == 0:
            y_label = train_data[label].values
            quantile_list = np.nanquantile(y_label, q_list)

            thresholds = []
            len_quantile = len(quantile_list)

            # average the symmetric quantiles to make absolute thresholds
            for i in range(int(len_quantile / 2) - 1, -1, -1):
                thresholds.append(
                    (abs(quantile_list[i]) + abs(quantile_list[len_quantile - i - 1])) / 2
                )

            print(f"threshold为{thresholds}")

            model_name = f"{base_model_name}_round{rnd}"
            for single_path in path_save:
                path_model_save = os.path.join(single_path, f"{model_name}")
                if not os.path.exists(path_model_save):
                    os.makedirs(path_model_save, exist_ok=True)

                path_threshold = os.path.join(single_path, model_name, "threshold.pkl")
                with open(path_threshold, "wb") as file:
                    pickle.dump(thresholds, file)
        else:
            # load quantile metadata and thresholds from round 0
            dict_quantile = pd.read_pickle(
                os.path.join(
                    path_save[0],
                    f"{base_model_name}_round0",
                    f"{base_model_name}_round0.pkl",
                )
            )["quantile"]
            thresholds = pd.read_pickle(
                os.path.join(
                    path_save[0],
                    f"{base_model_name}_round0",
                    "threshold.pkl",
                )
            )

        if not is_read_valid:
            valid_files = random.sample(valid_set, num_valid_ts)
            valid_data = parallel_read_data(
                valid_files,
                max_workers=30,
                batch_size=len(valid_files),
            )

            valid_data[train_label] = create_labels(valid_data[label], thresholds)
            X_test = valid_data.loc[:, feature_ls].astype(np.float32).values
            y_test = valid_data.loc[:, train_label].values
            del valid_data

            print("测试集", X_test.shape)
            is_read_valid = True

        train_data = process_label(train_data, train_label=train_label)

        print("构造后的train_data", train_data.shape)

        X_train = train_data.loc[:, feature_ls].values.astype(np.float32, copy=False)
        y_train = train_data.loc[:, train_label].values

        plt.hist(y_train, bins=100, color="green", alpha=0.7)
        plt.title(f"Dist of label#{label_name}_rnd{rnd}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.xlim(-6, 6)
        plt.tight_layout()
        plt.show()

        print("训练集:", X_train.shape, "验证集:", X_test.shape)

        del train_data
        gc.collect()

        shuffled_indices = np.random.permutation(len(X_train))
        X_train = X_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        print("数据准备完成")

        def log_ic_score(env):
            if env.evaluation_result_list:
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    if data_name == "valid" and eval_name == "rmse" and env.iteration % 50 == 0:
                        y_pred_cb = env.model.predict(X_test, num_iteration=env.iteration)
                        ic, _ = pearsonr(y_test, y_pred_cb)
                        print(f"[{env.iteration}] valid's IC: {ic}")

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
            log_ic_score,
        ]

        if rnd == 0:
            lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
            print(f"lgb_train: X_shape {X_train.shape} y_shape {y_train.shape}")

            gbm = lgb.train(
                params,
                lgb_train,
                num_boost_round=500,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=["train", "valid"],
                callbacks=callbacks,
                keep_training_booster=True,
            )
        else:
            if not is_first_train:
                X_train_add = np.vstack([old_subset_X, X_train])
                y_train_add = np.concatenate([old_subset_y, y_train])
            else:
                X_train_add = X_train
                y_train_add = y_train

            lgb_train = lgb.Dataset(X_train_add, y_train_add, free_raw_data=False)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
            print(f"lgb_train: X_shape {X_train_add.shape} y_shape {y_train_add.shape}")

            params["learning_rate"] = max(params["learning_rate"] - 0.005, 0.01)

            last_gbm = lgb.Booster(
                model_file=os.path.join(
                    path_save[0],
                    f"{base_model_name}_round{rnd-1}",
                    f"{base_model_name}_round{rnd-1}.txt",
                )
            )

            gbm = lgb.train(
                params,
                lgb_train,
                num_boost_round=300,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=["train", "valid"],
                callbacks=callbacks,
                keep_training_booster=True,
                init_model=last_gbm,
            )

            del X_train_add, y_train_add

        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"The RMSE of prediction is: {rmse}")

        ic, _ = pearsonr(y_test, y_pred)
        print(f"The IC (Information Coefficient) of prediction is: {ic}")

        y_pred_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
        pred_stats = {
            "mean": y_pred_train.mean(),
            "std": y_pred_train.std(),
        }

        if rnd == 0:
            dict_quantile = {f"{q}": np.quantile(y_pred_train, q) for q in save_quantile}
            print("dict_quantile:", {f"{q}": np.quantile(y_pred_train, q) for q in save_quantile})

        model_name = f"{base_model_name}_round{rnd}"

        for single_path in path_save:
            path_model_save = os.path.join(single_path, f"{model_name}")
            if not os.path.exists(path_model_save):
                os.makedirs(path_model_save, exist_ok=True)

            path_info = os.path.join(path_model_save, f"{model_name}.pkl")
            path_model = os.path.join(path_model_save, f"{model_name}.txt")
            path_json = os.path.join(path_model_save, "X_stats.json")

            with open(path_info, "wb") as f:
                pickle.dump(
                    {
                        "model": gbm,
                        "feature_ls": feature_ls,
                        "feature_stats": [],
                        "pred_stats": pred_stats,
                        "quantile": dict_quantile,
                    },
                    f,
                )

            gbm.save_model(path_model, num_iteration=gbm.best_iteration)

            data_to_save = {model_name: pred_stats}
            with open(path_json, "w") as json_file:
                json.dump(data_to_save, json_file, indent=4)

            print("数据已成功保存为 pkl 文件 和 JSON 文件。")

        del lgb_train, lgb_eval, y_pred_train, y_pred, gbm

        old_subset_X, old_subset_y = random_subset(X_train, y_train, frac=0.2)
        is_first_train = False

        del X_train, y_train
        gc.collect()


# ---------------------------------------------------------------------
# feature importance extraction
# ---------------------------------------------------------------------

def export_feature_importance(
    rnd=4,
    label_name=60,
    importance_gain_csv=None,
    importance_split_csv=None,
):
    """
    load a saved lightgbm model and export feature importance.

    the notebook showed:
    - gain-based importance
    - split-count importance

    note:
    -----
    the screenshot also showed a NameError because `lgb` was not defined in that
    notebook cell. that problem disappears in this script because lgb is imported
    at the top.
    """
    base_model_name = f"lgbm_zl_all_1{label_name}s"

    gbm = lgb.Booster(
        model_file=os.path.join(
            path_save[0],
            f"{base_model_name}_round{rnd}",
            f"{base_model_name}_round{rnd}.txt",
        )
    )

    feature_names = gbm.feature_name()

    # gain-based importance
    importance_gain = gbm.feature_importance(importance_type="gain")
    importance_df_gain = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_gain,
    }).sort_values(by="importance", ascending=False)

    print("按信息增益（Gain）排序的特征重要性：")
    print(importance_df_gain.head(10))

    if importance_gain_csv is not None:
        importance_df_gain.to_csv(importance_gain_csv, index=False)

    # split-based importance
    importance_split = gbm.feature_importance(importance_type="split")
    importance_df_split = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_split,
    }).sort_values(by="importance", ascending=False)

    print("按分裂次数（Split）排序的特征重要性：")
    print(importance_df_split.head(10))

    if importance_split_csv is not None:
        importance_df_split.to_csv(importance_split_csv, index=False)

    return importance_df_gain, importance_df_split


# ---------------------------------------------------------------------
# big-picture explanation
# ---------------------------------------------------------------------
#
# why this exists in a quant workflow:
#
# 1. raw factor data and labels are usually stored per stock per day or per time bucket.
#    this notebook loads them file-by-file and merges them into model-ready tables.
#
# 2. the target is not used directly in raw continuous form.
#    instead, the pipeline converts it into signed magnitude buckets.
#    that lets the model focus on "how strong and in what direction" a move is,
#    rather than predicting every tiny real-valued fluctuation exactly.
#
# 3. financial labels are usually badly imbalanced:
#    most observations are close to zero, and large positive / negative moves
#    are rare. the zero-class downsampling and +/- label balancing try to stop
#    the model from learning the trivial answer "predict almost no move."
#
# 4. lightgbm is used because:
#    - it is fast on wide tabular factor data
#    - it handles non-linear interactions well
#    - it works well with noisy, heterogeneous engineered features
#    - it is easier to train and debug than deep sequence models for many factor setups
#
# 5. the multi-round training logic is a practical engineering choice:
#    - train on batches of tasks instead of loading everything into memory
#    - keep a subset of old data so the model does not fully forget previous rounds
#    - continue training from the previous booster checkpoint
#
# 6. the saved quantiles / prediction statistics are not decoration.
#    they are useful later for:
#    - calibrating signals
#    - setting long-short cutoffs
#    - mapping raw model outputs into portfolio actions
#
# what this does NOT solve automatically:
# - leakage
# - execution cost modeling
# - turnover control
# - portfolio construction
# - regime robustness
# - economic validation of factor logic
#
# bluntly:
# this pipeline is not "the strategy."
# it is the model-training layer inside a larger quant research stack.
#
# the real quality depends on:
# - whether the factor features are meaningful
# - whether the label is economically sensible
# - whether the time split is honest
# - whether out-of-sample IC remains stable
# - whether the signal survives costs and capacity constraints
#

if __name__ == "__main__":
    # this script is intentionally not auto-running training because the original
    # notebook depends on a project-specific filesystem and internal libraries.
    print("combined lgbm quant pipeline loaded.")
    print("edit paths / feature list / environment-specific dependencies before running.")
