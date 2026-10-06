# %%
import pandas as pd

from mms_table4 import run

pd.set_option("display.float_format", "{:,.3f}".format, "display.max_columns", None)

# File path (.csv, tab-separated .txt, .xlsx) or a DataFrame already loaded in the notebook
monthly = "monthly.csv"
weekly = "weekly.csv"
participation = 0.52  # screened / eligible, same for all years
c_fix = 1.347  # cost fixed in MMS A.6.1; None = c from the p_T = 1 fit

# %%
tab_monthly = run(monthly, participation, c_fix=c_fix)
tab_monthly.to_csv("table4_monthly.csv")
display(tab_monthly)

# %%
tab_weekly = run(weekly, participation, c_fix=c_fix)
tab_weekly.to_csv("table4_weekly.csv")
display(tab_weekly)
