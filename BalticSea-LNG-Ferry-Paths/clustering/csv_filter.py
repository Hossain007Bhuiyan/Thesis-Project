import csv
import pandas as pd

df = pd.read_csv("data_all.csv")

mystar_filter = df["IMO"] == 9892690
nilsholgersson_filter = df["IMO"] == 9865685
peterpan_filter = df["IMO"] == 9880946
vikingglory_filter = df["IMO"] == 9827877
vikinggrace_filter = df["IMO"] == 9606900

dfms = pd.DataFrame(df[mystar_filter])
dfms.to_csv("mystar.csv")

dfng = pd.DataFrame(df[nilsholgersson_filter])
dfng.to_csv("nilsholgersson.csv")

dfpp = pd.DataFrame(df[peterpan_filter])
dfpp.to_csv("peterpan.csv")

dfvgl = pd.DataFrame(df[vikingglory_filter])
dfvgl.to_csv("vikingglory.csv")

dfvgr = pd.DataFrame(df[vikinggrace_filter])
dfvgr.to_csv("vikinggrace.csv")


