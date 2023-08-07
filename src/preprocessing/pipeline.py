from typing import List

from config import paths
from schema.data_schema import BinaryClassificationSchema
from preprocessing.preprocess import *


def create_pipeline(schema: BinaryClassificationSchema) -> List[Any]:
    """
        Creates pipeline of preprocessing steps

        Args:
            schema (BinaryClassificationSchema): BinaryClassificationSchema object carrying data about the schema
        Returns:
            A list of tuples containing the functions to be executed in the pipeline on a certain column
        """
    pipeline = [(drop_constant_features, None),
                (drop_all_nan_features, None),
                (drop_duplicate_features, None),
                (drop_mostly_missing_columns, None),
                (indicate_missing_values, None),
                ]
    numeric_features = schema.numeric_features
    cat_features = schema.categorical_features
    for f in numeric_features:
        pipeline.append((impute_numeric, f))
        # pipeline.append((remove_outliers_zscore, f))
    pipeline.append((normalize, 'schema'))

    for f in cat_features:
        pipeline.append((impute_categorical, f))
    pipeline.append((encode, 'schema'))

    return pipeline


def run_testing_pipeline(data: pd.DataFrame, data_schema: BinaryClassificationSchema, pipeline: List):
    for stage, column in pipeline:
        if column is None:
            data = stage(data)
        elif column == 'schema':
            if stage.__name__ == 'normalize' and os.path.exists(paths.SCALER_FILE):
                scaler = load(paths.SCALER_FILE)
                data = normalize(data, data_schema, scaler)
            elif stage.__name__ == 'encode':
                data = stage(data, data_schema, encoder='predict')
            else:
                data = stage(data, data_schema)
        else:
            if stage.__name__ == 'remove_outliers_zscore':
                continue
            else:
                data = stage(data, column)
    return data


