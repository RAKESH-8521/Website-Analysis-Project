# Website-Analysis-Project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data-export (1).csv")

df.head()

df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)

# Rename only the first 5 columns to avoid the length mismatch error with the 10-column DataFrame
rename_dict = {
    df.columns[0]: "channel group",
    df.columns[1]: "DateHour",
    df.columns[2]: "Users",
    df.columns[3]: "Sessions",
    df.columns[4]: "Engaged Sess",
}
df = df.rename(columns=rename_dict)
df.head()

df.head()

df.info()

# cleaning data and data validation for the above data **set**
df.head()

df["DateHour"] = pd.to_datetime(df["DateHour"], format="%Y%m%d%H", errors='coerce')

# @title Default title text
numeric_cols = df.columns.drop(["channel group", "DateHour"])
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df["Hour"] = df["DateHour"].dt.hour

df.head()
df.info()

df.describe()
# **sessions and user over time**
sns.set(style="whitegrid")

plt.figure(figsize=(10,5))
df.groupby("DateHour")[["Sessions", "Users"]].sum().plot(ax=plt.gca())
plt.title("Sessions and users over time")
plt.xlabel("DateHour")
plt.ylabel("count")
plt.show()

# total users by **channel**
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="channel group",
    y="Users",
    estimator=np.sum,
    palette="viridis",
)
plt.title("Total Users by Channel")
plt.xticks(rotation=45)
plt.show()

df.head()

# Average engagement time by **channel**
plt.figure(figsize=(8, 5))

# Dynamically get the name of the 6th column (index 5) representing average engagement time
engagement_time_col = df.columns[5]

sns.barplot(
    data=df,
    x="channel group",
    y=engagement_time_col,
    estimator=np.mean,
    palette="magma",
    hue="channel group",
    legend=False,
)
plt.title("Avg Engagement Time by Channel")
plt.ylabel("Average engagement time per session")
plt.xticks(rotation=45)
plt.show()

df.head()

## Engagement Rate Distribution by **channel**
plt.figure(figsize=(8, 5))

# Dynamically get the name of the 9th column (index 8) representing the engagement rate
engagement_rate_col = df.columns[8]

sns.boxplot(
    data=df,
    x="channel group",
    y=engagement_rate_col,
    palette="coolwarm",
    hue="channel group",
    legend=False,
)
plt.title("Engagement Rate Distribution by Channel")
plt.ylabel("Engagement rate")
plt.xticks(rotation=45)
plt.show()

# Engaged vs non engaged **session**
df.head()

session_df = df.groupby("channel group")[["Sessions", "Engaged Sess"]].sum().reset_index()
session_df["Non-Engaged"] = session_df["Sessions"] - session_df["Engaged Sess"]

session_df_melted = session_df.melt(
    id_vars="channel group",
    value_vars=["Engaged Sess", "Non-Engaged"],
    var_name="variable",
    value_name="value"
)

plt.figure(figsize=(8, 5))
sns.barplot(data=session_df_melted, x="channel group", y="value", hue="variable")
plt.title("Engaged vs Non-Engaged Sessions")
plt.xticks(rotation=45)
plt.show()

# traffic by hour and **channel**
heatmap_data = df.groupby(["Hour", "channel group"])["Sessions"].sum().unstack()

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=.5, annot=True, fmt='.0f')
plt.title("Traffic by Hour and Channel")
plt.xlabel("Channel Group")
plt.ylabel("Hour of Day")
plt.show()

# engagement rate vs sessions over **time**
df.head()
session_df = df.groupby("channel group")[["Sessions", "Engaged Sess"]].sum().reset_index()
session_df["Non-Engaged"] = session_df["Sessions"] - session_df["Engaged Sess"]

session_df_melted = session_df.melt(
    id_vars="channel group",
    value_vars=["Engaged Sess", "Non-Engaged"],
    var_name="variable",
    value_name="value"
)

plt.figure(figsize=(8, 5))
sns.barplot(data=session_df_melted, x="channel group", y="value", hue="variable")
plt.title("Engaged vs Non-Engaged Sessions")
plt.xticks(rotation=45)
plt.show()

