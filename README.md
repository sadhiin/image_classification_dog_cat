Dog Cat Classification
==============================
<img src="https://storage.googleapis.com/kaggle-media/competitions/kaggle/3362/media/woof_meow.jpg" width=100% height=100%>


## Overview

This repository contains code and resources for a deep learning project that classifies images of dogs and cats. The project aims to build a model that can accurately distinguish between images of dogs and cats.

<img src="https://miro.medium.com/max/700/1*oB3S5yHHhvougJkPXuc8og.gif" width=100% height=100%>

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project structure follows the Cookiecutter Data Science template for organizing data science projects. Here's a brief overview of the directory structure:

Project Organization
```
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Prerequisites

Before you begin, ensure you have met the following requirements:
- [conda](https://docs.conda.io/en/latest/miniconda.html)
- [git](https://git-scm.com/)

### Environment Setup

1. Create a conda environment for this project:

```bash
conda create -n dogcat python=3.10 -y
```

2. Activate the environment:

```bash
conda activate dogcat
```

- Dependencies listed in `requirements.txt`

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone this repository:

```bash
git clone https://github.com/sadhiin/image_classification_dog_cat.git
cd IMAGE_CLASSIFICATION_DOG_CAT
```

2. Set up your environment and install dependencies as mentioned in the Prerequisites section.

## Dataset

The dataset used for this project consists of a collection of labeled images of dogs and cats. You can find the dataset [here](https://www.kaggle.com/datasets/tongpython/cat-and-dog
). Place the dataset in the `data/raw` directory.

## Training

To train the model, use the following command:

```shell
python src/models/train.py --config config.yaml
```

## Inference

You can use the trained model for inference by running:

```shell
python src/inference/predict.py --model-path /path/to/saved/model --input-image /path/to/input/image
```

## Evaluation

You can evaluate the model's performance using:

```shell
python src/evaluation/evaluate.py --model-path /path/to/saved/model --test-data /path/to/test/dataset
```

## Results

Provide details about the model's performance, including accuracy, loss, and any other relevant metrics. Include visualizations or graphs if possible.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/sadhiin/image_classification_dog_cat/blob/main/LICENSE) file for details.
