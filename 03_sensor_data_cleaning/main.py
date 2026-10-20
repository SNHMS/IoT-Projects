import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(23)
n = 300
data = pd.DataFrame({
    "temperature": rng.normal(28, 3, n),
    "humidity": rng.normal(65, 12, n),
    "soil_moisture": rng.uniform(15, 85, n)
})
data.loc[rng.choice(n, 18, replace=False), "humidity"] = np.nan
data.loc[rng.choice(n, 5, replace=False), "temperature"] = 95
data.to_csv("sensor_data_raw.csv", index=False)

clean = data.copy()
clean["humidity"] = clean["humidity"].fillna(clean["humidity"].median())
for column in clean.columns:
    low = clean[column].quantile(0.02)
    high = clean[column].quantile(0.98)
    clean[column] = clean[column].clip(low, high)
clean = clean.round(2)
os.makedirs("output", exist_ok=True)
clean.to_csv("output/sensor_data_clean.csv", index=False)
print("Rows:", len(clean))
print("Missing values after cleaning:", int(clean.isna().sum().sum()))
print(clean.describe().round(2))
