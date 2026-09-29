"""Paths and constants shared by the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KAGGLE_CSV = ROOT / "creditcard.csv"          # optional: the Kaggle file, if you downloaded it
CACHE_PARQUET = DATA_DIR / "creditcard.parquet"  # local cache of the OpenML download (not committed)
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "images"

OPENML_ID = 1597          # "creditcard" on OpenML: the same ULB dataset as Kaggle's mlg-ulb/creditcardfraud
# The dataset's ARFF file. sklearn's fetch_openml(data_id=1597) drops the Time column (OpenML marks it as a row
# identifier), and the notebook's hour-of-day features need it, so the file is downloaded directly.
OPENML_ARFF_URL = "https://www.openml.org/data/v1/download/1673544/creditcard.arff"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492

SEED = 42
VAL_SIZE = 0.2   # of all rows
TEST_SIZE = 0.2  # of all rows
N_JOBS = 4       # keep CPU use modest on shared machines
