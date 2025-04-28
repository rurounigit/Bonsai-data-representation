import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

papers = pd.read_csv('Single_cell_studies_database_data.tsv', sep='\t')
papers['Datetime'] = pd.to_datetime(papers['Date'], format='%Y%m%d')

papers['Reported cells total'] = papers['Reported cells total'].str.replace(',', '').map(float)

# plot number of studies over time
fig, ax = plt.subplots(figsize=(12, 5))

# papers['Datetime'] = pd.to_datetime(papers['Date'], format='%Y%m%d')
papers = papers.sort_values("Datetime")
papers["count"] = 1

x = papers.Datetime
y = papers["count"].groupby(papers.Datetime.dt.time).cumsum()

ax.plot(x, y, color="k")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative number of studies")

# plt.show()

# tools = pd.read_csv('https://raw.githubusercontent.com/Oshlack/scRNA-tools/master/database/tools.tsv', sep='\t')
tools = pd.read_csv('tools.tsv', sep='\t')
# tools.to_csv('tools.tsv', sep='\t')
tools["Datetime"] = pd.to_datetime(tools["Added"])
tools = tools.sort_values("Added")
tools["count"] = 1

fig, ax = plt.subplots(figsize=(12, 5))

x = tools.Datetime
y = tools["count"].groupby(tools.Datetime.dt.time).cumsum()

ax.plot(x, y, color="k")
ax.set_xlabel("Date")
ax.set_ylabel("Number of tools")
ax.tick_params(axis='x', rotation=45)

date_papers = papers.groupby("Datetime")["count"].sum()
date_tools = tools.groupby("Datetime")["count"].sum()
dates = pd.date_range(start='7/26/2002', end='12/31/2023')
combined = pd.DataFrame(index=dates)
combined["tool_counts"] = combined.index.map(date_tools)
combined["paper_counts"] = combined.index.map(date_papers)
combined = combined.fillna(0)
combined["Datetime"] = combined.index.values

fig, ax = plt.subplots(figsize=(5.5, 5))

x = combined["paper_counts"].groupby(combined.Datetime.dt.time).cumsum()
y = combined["tool_counts"].groupby(combined.Datetime.dt.time).cumsum()

ax.scatter(x, y, s=7)
# regr = linear_model.LinearRegression()
# x = x.values[:, np.newaxis]
# regr.fit(x, y.values)
# xx = np.linspace(0, max(x), 200)
# yy = regr.intercept_ + regr.coef_*xx
# ax.plot(xx, yy, color="r", label=f"{regr.intercept_:,.2f} + {regr.coef_[0]:,.2f}*x", linewidth=2)

lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes
ax.plot(lims, lims, '--', c='black', alpha=0.75, zorder=0, label="y=x")
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel("Cumulative # of scRNA-seq studies")
ax.set_ylabel("Cumulative # of scRNA-seq tools")
ax.legend()

# plt.savefig("/Users/Daan/Documents/postdoc/bonsai_paper/figures_highres/scaling_datasets_vs_tools.png", dpi=300)

plt.tight_layout()
plt.show()