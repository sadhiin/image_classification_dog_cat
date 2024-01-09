import argparse
from src import logger
from src.models.cnn import CNN_Model
from src.utils.getconfig import read_params
import src.utils.preprocess as preprocessing
from src.utils.keras_cbs import get_callbacks

def compile_model(config_path):
    config = read_params(config_path)
    logger.info(f"Loading the model")

    model = CNN_Model()
    logger.info(f"Model loaded")

    logger.info(model.summary())

    model.compile(
        loss=config['model']['loss'],
        optimizer=config['model']['optimizer'],
        metrics=[config['model']['metrics']]
    )
    logger.info(f"Compiling model with {config['model']['optimizer']} optimizer")
    logger.info(f"Compiling model with {config['model']['loss']} loss and {config['model']['metrics']} metrics")
    return model


def train_model(config_path, get_training_history: bool = True):
    config = read_params(config_path)

    model = compile_model(config_path)

    train_set = preprocessing.get_train_set(config_path)
    val_set = preprocessing.get_validation_set(config_path)
    bs = config['base']['batch_size']

    history = model.fit(
        train_set,
        validation_data=val_set,
        steps_per_epoch=len(train_set)//bs,
        validation_steps=len(val_set)//bs,
        epochs=config['training_config']['epoch'],
        callbacks=[get_callbacks(config_path)]
    )
    model.save(config['model']['savemodel'])
    if get_training_history:
        return history


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")

    parsed_args = args.parse_args()

    train_model(config_path=parsed_args.config)
