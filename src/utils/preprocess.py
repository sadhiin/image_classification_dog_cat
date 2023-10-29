import os
import yaml
import logging
import argparse
from getconfig import read_params
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def tain_generetor(config_path=None):
    return ImageDataGenerator(rescale=1./255)


def tain_generetor(config_path=None):
    if config_path:
        config = read_params(config_path)
        train_datagen = None
        return train_datagen
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=config["split_data"]["validation_size"])  # set validation split
        return train_datagen


def get_train_set(config_path):
    config = read_params(config_path)

    # set as training data
    train_datagen = tain_generetor()
    training_set = train_datagen.flow_from_directory(
        config["data_source"]["train_data_dir"],

        target_size=(config["base"]["width"], config["base"]["height"]),

        batch_size=config["base"]["batch_size"],

        class_mode="binary",

        subset="training"
    )
    return training_set


def get_validation_set(config_path):
    config = read_params(config_path)

    # set as training data
    train_datagen = tain_generetor()
    validation_set = train_datagen.flow_from_directory(
        config["data_source"]["train_data_dir"],
        target_size=(config["base"]["width"], config["base"]["height"]),
        batch_size=config["base"]["batch_size"],
        class_mode="binary",
        subset="validation"
    )  # set as validation data

    return validation_set


def get_test_set(config_path):
    config = read_params(config_path)
    test_datagen = tain_generetor()
    test_set = test_datagen.flow_from_directory(config["data_source"]["test"],
                                                target_size=(config["base"]["width"], config["base"]["height"]))
    return test_set


# if __name__ == "__main__":
#     args = argparse.ArgumentParser()
#     args.add_argument("--config", default="params.yaml")

#     parsed_args = args.parse_args()

#     get_train_set(config_path=parsed_args.config)
#     get_validation_set(config_path=parsed_args.config)
#     get_test_set(config_path=parsed_args.config)
