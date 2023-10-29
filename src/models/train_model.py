import os
import yaml
import logging
import argparse
from cnn import CNN_Model
import src.utils.getconfig as readconfig
import src.utils.preprocess as preprocessing


def compile_model(config_path):
    config = readconfig.read_params(config_path)

    model = CNN_Model()
    model.compile(
        loss=config['modelcompile']['loss'],
        optimizer= config['modelcompile']['optimizer'],
        metrics = [config['modelcompile']['metrics']]
    )

    return model


def train_model(config_path, model, get_training_history: bool):
    config = readconfig.read_params(config_path)
    model = compile_model(config_path)
    train_set = preprocessing.get_train_set()
    val_set= preprocessing.get_validation_set()
    test_set = preprocessing.get_test_set()
    bs = config['base']['batch_size']

    history= model.fit(
        train_set,
        steps_per_epoch= len(train_set)//bs,
        epochs= config['training_config']['epoch'],
        validation_data= val_set,
        # callbacks= CBS
    )

    if get_training_history:
        return history


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")

    parsed_args = args.parse_args()

    train_model(config_path=parsed_args.config)
