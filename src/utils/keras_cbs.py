import os
import time
import logging
from src.utils.getconfig import read_params
import tensorflow as tf


def get_log_path(DIR="Tensorboard_logs/logs/"):
    log_file_name = time.strftime("TB_log_%Y_%m_%d-%H_%M_%S")
    os.makedirs(DIR, exist_ok=True)
    log_path = os.path.join(DIR, log_file_name)
    logging.info(f"Tensorboard log path: {log_path}")
    print(f"Tensorboard log path: {log_path}")
    return log_path


def get_callbacks(config_path):
    log_path = get_log_path()

    config = read_params(config_path)
    file_name = os.path.join(config['model']['checkpoint'], "model_ckpt", "h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=file_name, save_best_only=True, monitor='val_loss', mode='min'),

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True),

        tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    ]

    return callbacks
