import numpy as np
import pandas as pd
from typing import List
from config import paths
from utils import read_csv_in_directory, save_dataframe_as_csv
from logger import get_logger
from ANN_Classifier import Classifier, predict_with_model
from preprocessing.pipeline import create_pipeline, run_testing_pipeline
from schema.data_schema import load_saved_schema


logger = get_logger(task_name="predict")


def create_predictions_dataframe(
        predictions_arr: np.ndarray,
        class_names: List[str],
        prediction_field_name: str,
        ids: pd.Series,
        id_field_name: str,
        return_probs: bool = False,
) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Performs the following transformations:
    - converts to pandas dataframe
    - adds class labels as headers for columns containing predicted probabilities
    - inserts the id column

    Args:
        predictions_arr (np.ndarray): Predicted probabilities from predictor model.
        class_names List[str]: List of target classes (labels).
        prediction_field_name (str): Field name to use for predicted class.
        ids: ids as a numpy array for each of the samples in  predictions.
        id_field_name (str): Name to use for the id field.
        return_probs (bool, optional): If True, returns the predicted probabilities
            for each class. If False, returns the final predicted class for each
            data point. Defaults to False.

    Returns:
        Predictions as a pandas dataframe
    """
    if predictions_arr.shape[1] != len(class_names):
        raise ValueError(
            "Length of class names does not match number of prediction columns"
        )
    predictions_df = pd.DataFrame(predictions_arr, columns=class_names)
    if len(predictions_arr) != len(ids):
        raise ValueError("Length of ids does not match number of predictions")
    predictions_df.insert(0, id_field_name, ids)
    if return_probs:
        return predictions_df
    predictions_df[prediction_field_name] = predictions_df[class_names].idxmax(axis=1)
    predictions_df.drop(class_names, axis=1, inplace=True)
    return predictions_df


def run_batch_predictions() -> None:
    test_data = read_csv_in_directory(paths.TEST_DIR)
    data_schema = load_saved_schema(paths.SAVED_SCHEMA_DIR_PATH)
    model = Classifier.load(paths.PREDICTOR_DIR_PATH)
    pipeline = create_pipeline(data_schema)
    features = data_schema.features
    x_test = test_data[features]

    logger.info("Transforming the data...")
    x_test = run_testing_pipeline(x_test, data_schema, pipeline)
    logger.info("Making predictions...")
    predictions_arr = predict_with_model(model, x_test, return_probs=True)
    predictions_df = create_predictions_dataframe(
        predictions_arr,
        data_schema.target_classes,
        'prediction',
        test_data[data_schema.id],
        data_schema.id,
        return_probs=True,
    )
    logger.info("Saving predictions...")
    save_dataframe_as_csv(
        dataframe=predictions_df, file_path=paths.PREDICTIONS_FILE_PATH
    )

    logger.info("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions()