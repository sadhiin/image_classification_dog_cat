import argparse
from src import logger
from tensorflow.keras.models import load_model
from src.utils.getconfig import read_params
import src.utils.preprocess as preprocessing


def evaluate_model(config_path):
    try:
        config = read_params(config_path)
        logger.info(f"Loading the model")

        model = load_model(config['model']['savemodel'])
        logger.info(f"Model loaded")
        logger.info(model.summary())

        test_set = preprocessing.get_test_set(config_path)
        bs = config['base']['batch_size']
        loss, acc = model.evaluate(
            test_set,
            steps=len(test_set)//bs
        )
        logger.info(f"Model evaluated")
        logger.info(f"Loss: {loss}")
        logger.info(f"Accuracy: {acc}")
    except Exception as e:
        logger.error(f"Error at evaluating the model: {e}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")

    parsed_args = args.parse_args()

    evaluate_model(config_path=parsed_args.config)
