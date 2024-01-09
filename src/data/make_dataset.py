# -*- coding: utf-8 -*-
import click
import argparse
from src import logger
import opendatasets as od
from src.utils.getconfig import read_params
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def get_data(config_path):
    try:
        config = read_params(config_path)
        dataset_url = config["data_source"]["remote_source"]
        save_dir = config["data_source"]["local_source"]
        logger.info(f"Downloading the data from {dataset_url}")
        od.download(dataset_id_or_url=dataset_url, data_dir=save_dir, force=True)
        logger.info(f"Data downloaded at {save_dir}")
    except Exception as e:
        logger.error(f"Error at downloading the data: {e}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    get_data(parsed_args.config)
