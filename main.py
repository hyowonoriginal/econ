# %%
import pandas as pd

from mms_table4 import run

pd.set_option("display.float_format", "{:,.3f}".format, "display.max_columns", None, "display.max_rows", None)

weekly = "weekly.xlsx"  # columns woy, CNT_INDI, AMONG_SCREEN, YEAR (YEAR needed for excl. 2020)
q = 0.52  # participation rate (screened / eligible), same for all years

# %%
tab = run(weekly, q=q, exclude=2020)
tab.to_csv("table4_weekly.csv")
display(tab)
