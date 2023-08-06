import os
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple
from schema.data_schema import BinaryClassificationSchema
from sklearn.preprocessing import MinMaxScaler
from feature_engine.encoding import OneHotEncoder
from scipy.stats import zscore
from joblib import dump, load
from config import paths
from imblearn.over_sampling import SMOTE


def impute_numeric(input_data: pd.DataFrame, column: Any, value='median', schema: BinaryClassificationSchema = None, target: pd.Series = None) -> pd.DataFrame:
    if column not in input_data.columns:
        return input_data
    if value == 'mean':
        input_data[column].fillna(value=input_data[column].mean(), inplace=True)
    elif value == 'median':
        input_data[column].fillna(value=input_data[column].median(), inplace=True)
    elif value == 'mode':
        input_data[column].fillna(value=input_data[column].mode().iloc[0], inplace=True)
    else:
        input_data[column].fillna(value=value, inplace=True)
    return input_data


def indicate_missing_values(input_data: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = input_data.select_dtypes(exclude=["number"]).columns.tolist()
    return input_data.replace("", np.nan)


def impute_categorical(input_data: pd.DataFrame, column: Any) -> pd.DataFrame:
    if column not in input_data.columns:
        return input_data
    perc = percentage_of_missing_values(input_data)
    if column in perc and perc[column] > 10:
        input_data[column].fillna(value='Missing', inplace=True)
    else:
        input_data[column].fillna(value=input_data[column].mode().iloc[0], inplace=True)
    return input_data


def drop_all_nan_features(input_data: pd.DataFrame) -> pd.DataFrame:
    return input_data.dropna(axis=1, how='all')


def percentage_of_missing_values(input_data: pd.DataFrame) -> Dict:
    columns_with_missing_values = input_data.columns[input_data.isna().any()]
    return (input_data[columns_with_missing_values].isna().mean().sort_values(ascending=False) * 100).to_dict()


def drop_constant_features(input_data: pd.DataFrame) -> pd.DataFrame:
    constant_columns = input_data.columns[input_data.nunique() == 1]
    return input_data.drop(columns=constant_columns)


def drop_duplicate_features(input_data: pd.DataFrame) -> pd.DataFrame:
    return input_data.T.drop_duplicates().T


def encode(input_data: pd.DataFrame, schema: BinaryClassificationSchema, encoder=None) -> pd.DataFrame:
    cat_features = schema.categorical_features
    if not cat_features:
        return input_data
    if encoder is not None:
        encoder = load(paths.ENCODER_FILE)
        input_data = encoder.transform(input_data)
        return input_data

    encoder = OneHotEncoder(top_categories=3)
    encoder.fit(input_data)
    input_data = encoder.transform(input_data)
    dump(encoder, paths.ENCODER_FILE)
    return input_data


def drop_mostly_missing_columns(input_data: pd.DataFrame, thresh=0.6) -> pd.DataFrame:
    threshold = int(thresh * len(input_data))
    return input_data.dropna(axis=1, thresh=threshold)


def cast_data(input_data: pd.DataFrame):
    input_data = input_data.convert_dtypes()
    return input_data


def normalize(input_data: pd.DataFrame, schema: BinaryClassificationSchema, scaler=None) -> pd.DataFrame:
    numeric_features = schema.numeric_features
    if not numeric_features:
        return input_data
    numeric_features = list(set(numeric_features).intersection(input_data.columns))
    if scaler is None:
        scaler = MinMaxScaler()
        input_data[numeric_features] = scaler.fit_transform(input_data[numeric_features])
        dump(scaler, paths.SCALER_FILE)
    else:
        print('hena')
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    return input_data


def remove_outliers_zscore(input_data: pd.DataFrame, column: str, target: pd.Series = None) -> pd.DataFrame:
    if column not in input_data.columns:
        return input_data, target
    input_data[column] = input_data[column].astype(np.float64)
    threshold = 3
    z_scores = np.abs(zscore(input_data[column]))
    condition = z_scores < threshold
    after_removal = input_data[condition]
    if (after_removal.shape[0] / input_data.shape[0]) < 0.1:
        if target is not None:
            return input_data[condition], target[condition]
        else:
            return input_data[condition], None
    else:
        return input_data, target


def remove_outliers_iqr(input_data: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in input_data.columns:
        return input_data
    q1 = input_data[column].quantile(0.25)
    q3 = input_data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    condition = (input_data[column] > upper) | (input_data[column] < lower)
    return input_data[~condition]


def handle_class_imbalance(
    transformed_data: pd.DataFrame, transformed_labels: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE.

    Args:
        transformed_data (pd.DataFrame): The transformed data.
        transformed_labels (pd.Series): The transformed labels.
        random_state (int): The random state seed for reproducibility. Defaults to 0.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the balanced data and
            balanced labels.
    """
    # Adjust k_neighbors parameter for SMOTE
    # set k_neighbors to be the smaller of two values:
    #       1 and,
    #       the number of instances in the minority class minus one
    k_neighbors = min(
        1, sum(transformed_labels == min(transformed_labels.value_counts().index)) - 1
    )
    smote = SMOTE(k_neighbors=k_neighbors, random_state=0)
    balanced_data, balanced_labels = smote.fit_resample(
        transformed_data, transformed_labels
    )
    return balanced_data, balanced_labels

