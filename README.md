tspipe - Time Series Pipeline
==============================
> A scalable ETL/ML pipeline for rapid testing of feature engineering and classical machine learning techniques. Featured in the paper *Machine Learning in Manufacturing: Best Practices*. Tested on the UC Berkeley milling dataset and a CNC manufacturing dataset. Leverages HPC infrastructure.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tvhahn/tspipe/blob/master/notebooks/milling-reproduction.ipynb) ![example workflow](https://github.com/tvhahn/feat-store/actions/workflows/main.yml/badge.svg)

The best way to reproduce the results from the paper is to run the [Colab notebook for the milling dataset](https://colab.research.google.com/github/tvhahn/tspipe/blob/master/notebooks/milling-reproduction.ipynb).



run tests: `python -m unittest discover -s tests`

Project Organization
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
    ├── tests              <- Tests for this project.
    │   └── integration    <- Integration tests for this project.
    │       │
    │       ├── test_integration_milling.py <- Integration test for the `milling` dataset
    │       │
    │       └── fixtures    <- Folder with files needed for integration tests
    │
    │        
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
