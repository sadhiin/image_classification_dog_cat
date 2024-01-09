import argparse
import numpy as np
import tensorflow as tf
from src import logger
from src.utils.getconfig import read_params
# from tensorflow.keras.models import load_model


def predict_model(config_path, image_path):
    pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    args.add_argument("--image_path", default="")

    parsed_args = args.parse_args()

    predict_model(config_path=parsed_args.config,
                  image_path=parsed_args.image_path)
