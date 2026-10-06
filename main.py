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

# %% beta as a function of the fixed ybar (shock variance pi^2/3; beta is the same for 5x, 25x)
import matplotlib.pyplot as plt
import numpy as np

from mms_table4 import fit, load

df = load(weekly, q)
grid = np.arange(-8, 0.01, 0.25)
groups = {"pooled": df} | ({"excl. 2020": df[df.YEAR != 2020]} if 2020 in set(df.YEAR) else {})
for g, d in groups.items():
    for dist in ("logistic", "normal"):
        beta = np.array([fit(d, dist, 1, y)["beta"] for y in grid])
        plt.plot(grid, beta, label=f"{g}, {dist}")
        print(g, dist, "largest ybar with beta < 1:", grid[beta < 1].max() if (beta < 1).any() else None)
plt.axhline(1, color="gray", ls="--")
plt.xlabel("ybar (value of never screening)")
plt.ylabel("beta")
plt.legend()
plt.show()
