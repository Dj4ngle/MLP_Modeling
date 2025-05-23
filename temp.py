import pandas as pd
df = pd.read_csv("bpa_dataset.csv")
print(df[["r1", "r2", "r3"]].describe())
