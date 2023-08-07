import numpy as np
import pandas as pd
from typing import Dict, Tuple
from schema.data_schema import BinaryClassificationSchema
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import OneHotEncoder
from scipy.stats import zscore
from joblib import dump, load
from config import paths
from imblearn.over_sampling import SMOTE


def impute_numeric(input_data: pd.DataFrame, column: str, value='median') -> pd.DataFrame:
    """
    Imputes the missing numeric values in the given dataframe column based on the parameter 'value'.

    Args:
        input_data (pd.DataFrame): The data to be imputed.
        column (str): The name of the column.
        value (str): The value to use when imputing the column. Can only be one of ['mean', 'median', 'mode']

    Returns:
        A dataframe after imputation
    """

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
    """
    Replaces empty strings with NaN in a dataframe.

    Args:
        input_data (ps.DataFrame): The dataframe to be processed.

    Returns:
        A dataframe after replacing empty strings with NaN.
    """
    return input_data.replace("", np.nan)


def impute_categorical(input_data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Imputes the missing categorical values in the given dataframe column.
    If the percentage of missing values in the column is greater than 0.1, imputation is done using the word "Missing".
    Otherwise, the mode is used.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.
        column (str): The name of the column to be imputed.

    Returns:
        A dataframe after imputation
    """
    if column not in input_data.columns:
        return input_data
    perc = percentage_of_missing_values(input_data)
    if column in perc and perc[column] > 10:
        input_data[column].fillna(value='Missing', inplace=True)
    else:
        input_data[column].fillna(value=input_data[column].mode().iloc[0], inplace=True)
    return input_data


def drop_all_nan_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that only contain NaN values.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.

    Returns:
        A dataframe after dropping NaN columns
    """
    return input_data.dropna(axis=1, how='all')


def percentage_of_missing_values(input_data: pd.DataFrame) -> Dict:
    """
    Calculates the percentage of missing values in each column of a given dataframe.

    Args:
        input_data (pd.DataFrame): The dataframe to calculate the percentage of missing values on.

    Returns:
        A dictionary of column names as keys and the percentage of missing values as values.
    """
    columns_with_missing_values = input_data.columns[input_data.isna().any()]
    return (input_data[columns_with_missing_values].isna().mean().sort_values(ascending=False) * 100).to_dict()


def drop_constant_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that contain only one value.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.

    Returns:
        A dataframe after dropping constant columns
    """
    constant_columns = input_data.columns[input_data.nunique() == 1]
    return input_data.drop(columns=constant_columns)


def drop_duplicate_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are exactly the same and keeps only one of them.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.

    Returns:
        A dataframe after dropping duplicated columns
    """
    return input_data.T.drop_duplicates().T


def encode(input_data: pd.DataFrame, schema: BinaryClassificationSchema, encoder=None) -> pd.DataFrame:
    """
    Performs one-hot encoding for the top 3 categories on categorical features of a given dataframe.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.
        schema (BinaryClassificationSchema): The schema of the given data.
        encoder: Indicates if instantiating a new encoder is required or not.

    Returns:
        A dataframe after performing one-hot encoding
    """
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
    """
    Drops columns in which NaN values exceeds a certain threshold.

    Args:
        input_data: (pd.DataFrame): the data to be processed.
        thresh (float): The threshold to use.

    Returns:
        A dataframe after dropping the specified columns.
    """
    threshold = int(thresh * len(input_data))
    return input_data.dropna(axis=1, thresh=threshold)


def normalize(input_data: pd.DataFrame, schema: BinaryClassificationSchema, scaler=None) -> pd.DataFrame:
    """
    Performs z-score normalization on numeric features of a given dataframe.

    Args:
        input_data (pd.DataFrame): The data to be normalized.
        schema (BinaryClassificationSchema): The schema of the given data.
        scaler: Indicated if a new scaler needs to be instantiated.

    Returns:
        A dataframe after z-score normalization
    """

    input_data = input_data.copy()
    numeric_features = schema.numeric_features
    if not numeric_features:
        return input_data
    numeric_features = [f for f in numeric_features if f in input_data.columns]
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(input_data[numeric_features])
        dump(scaler, paths.SCALER_FILE)
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    return input_data


def remove_outliers_zscore(input_data: pd.DataFrame, column: str, target: pd.Series = None)\
        -> (pd.DataFrame, pd.Series):
    """
    Removes rows that have been identified as outliers using the z-score method according to a given column.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.
        column (str): The name of the column.
        target (pd.Series): The targets series associated with the input dataframe.
    """

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


def handle_class_imbalance(
        transformed_data: pd.DataFrame, transformed_labels: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE.

    Args:
        transformed_data (pd.DataFrame): The transformed data.
        transformed_labels (pd.Series): The transformed labels.

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
