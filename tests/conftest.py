import numpy as np
import pandas as pd
import pytest

from src.data import RAW_COLUMNS


@pytest.fixture()
def toy():
    """A small synthetic frame with the real columns: 2,000 rows, 2% fraud (fraud rows shifted on V14)."""
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(rng.normal(size=(n, len(RAW_COLUMNS))), columns=RAW_COLUMNS)
    df["Time"] = np.sort(rng.uniform(0, 172_792, n))
    df["Amount"] = rng.exponential(80, n).round(2)
    df.loc[rng.choice(n, 50, replace=False), "Amount"] = 0.0
    df["Class"] = 0
    fraud = rng.choice(n, 40, replace=False)
    df.loc[fraud, "Class"] = 1
    df.loc[fraud, "V14"] -= 4
    return df
