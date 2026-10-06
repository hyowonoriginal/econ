# %%
import pandas as pd

from mms_table4 import run

pd.set_option("display.float_format", "{:,.3f}".format, "display.max_columns", None, "display.max_rows", None)

weekly = "weekly.xlsx"  # columns woy, CNT_INDI, AMONG_SCREEN (YEAR for the excl. 2020 version)
participation = 0.52  # screened / eligible, same for all years

# %%
tab = run(weekly, participation)
tab.to_csv("table4_weekly.csv")
display(tab)
