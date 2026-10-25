"""Merge the five source files into data/daily.csv (no filling beyond what the cleaned files already contain).
Usage: python scripts/prepare_data.py <folder with the 5 source files>
Expected names (any prefix): crude_oil_filled_ffill.csv, arrivals_indore_*.csv, price_data_cleaned_rajkot.xlsx,
price_indore_market_cleaned_*.csv, merged_quantity_2011_2025_filled.csv
"""
import sys, glob, pandas as pd
from pathlib import Path
src = Path(sys.argv[1]); out = Path(__file__).resolve().parent.parent / "data"
f = lambda pat: glob.glob(str(src / f"*{pat}*"))[0]
ind = pd.read_csv(f("price_indore_market"), parse_dates=["date"]).set_index("date")[["p_min","p_max","p_modal"]].add_prefix("ind_")
gon = pd.read_excel(f("price_data_cleaned_rajkot"), parse_dates=["Date"]).set_index("Date")[["p_min","p_max","p_modal"]].add_prefix("gon_")
iar = pd.read_csv(f("arrivals_indore"), parse_dates=["date"]).set_index("date")[["quantity_tonnes"]].rename(columns={"quantity_tonnes":"ind_arrivals_t"})
gar = pd.read_csv(f("merged_quantity"), parse_dates=["Date"]).set_index("Date")[["qty"]].rename(columns={"qty":"gon_arrivals_qty"})
oil = pd.read_csv(f("crude_oil_filled"), parse_dates=["date"]).set_index("date").rename(columns={"price":"crude"})
for n, d in [("ind",ind),("gon",gon),("iar",iar),("gar",gar),("oil",oil)]:
    assert d.index.is_unique, f"duplicate dates in {n}"
d = pd.concat([ind, gon, iar, gar], axis=1).sort_index()
d = d.loc["2011-01-01":"2025-12-31"].join(oil)
d.index.name = "date"
d.round(3).to_csv(out / "daily.csv")
print(d.shape, d.index.min().date(), d.index.max().date()); print(d.isna().sum().to_dict())
