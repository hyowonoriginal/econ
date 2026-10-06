# %%
import pandas as pd

from mms_table4 import run

pd.set_option("display.float_format", "{:,.3f}".format, "display.max_columns", None)

# File path (.csv, tab-separated .txt, .xlsx) or a DataFrame already loaded in the notebook
monthly = "monthly.csv"
weekly = "weekly.csv"

# %%
tab_monthly = run(monthly)
tab_monthly.to_csv("table4_monthly.csv")
display(tab_monthly)

# %%
tab_weekly = run(weekly)
tab_weekly.to_csv("table4_weekly.csv")
display(tab_weekly)
