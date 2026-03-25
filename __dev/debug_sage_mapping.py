import pandas as pd
import re

from pandas_ops.io import read_df
from timstofu.conversion.reformat import open_pmsms
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 5)
