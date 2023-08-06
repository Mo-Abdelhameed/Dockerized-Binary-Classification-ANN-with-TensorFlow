import os
from ANN_Classifier import Classifier, evaluate_predictor_model, save_predictor_model, train_predictor_model
from preprocessing.target_encoder import get_target_encoder, transform_targets
from utils import read_csv_in_directory
from config import paths
from logger import get_logger, log_error
from schema.data_schema import load_json_data_schema, save_schema
from preprocessing.pipeline import create_pipeline
from preprocessing.preprocess import handle_class_imbalance

from utils import set_seeds, read_json_as_dict


logger = get_logger(task_name="train")


def run_training(
        input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
        saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
        train_dir: str = paths.TRAIN_DIR,
        predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
        default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
       ):
    try:
        logger.info("Starting training...")
        set_seeds(seed_value=123)

        logger.info("Loading and saving schema...")
        data_schema = load_json_data_schema(input_schema_dir)
        save_schema(schema=data_schema, save_dir_path=saved_schema_dir_path)

        logger.info("Loading training data...")
        train_data = read_csv_in_directory(train_dir)
        features = data_schema.features
        target = data_schema.target
        x_train = train_data[features]
        y_train = train_data[target]
        pipeline = create_pipeline(data_schema)
        # x_train, y_train = handle_class_imbalance(x_train, y_train)
        for stage, column in pipeline:
            if column is None:
                x_train = stage(x_train)
            elif column == 'schema':
                x_train = stage(x_train, data_schema)
            else:
                if stage.__name__ == 'remove_outliers_zscore':
                    x_train, y_train = stage(x_train, column, target=y_train)
                else:
                    x_train = stage(x_train, column)

        default_hyperparameters = read_json_as_dict(default_hyperparameters_file_path)

        target_encoder = get_target_encoder(data_schema)
        transformed_targets = transform_targets(target_encoder, train_data)

        x_train, transformed_targets = handle_class_imbalance(x_train, transformed_targets)

        model = train_predictor_model(x_train, transformed_targets, default_hyperparameters)
        if not os.path.exists(predictor_dir_path):
            os.makedirs(predictor_dir_path)
        model.save(predictor_dir_path)
        logger.info('Model saved!')

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
