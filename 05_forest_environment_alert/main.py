import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(55)
n = 500
data = pd.DataFrame({
    "air_temperature": rng.uniform(18, 44, n),
    "humidity": rng.uniform(15, 95, n),
    "smoke_level": rng.uniform(0, 100, n),
    "soil_moisture": rng.uniform(5, 90, n)
})
data["risk_score"] = (
    (data["air_temperature"] > 35).astype(int) +
    (data["humidity"] < 35).astype(int) +
    (data["smoke_level"] > 65).astype(int) +
    (data["soil_moisture"] < 20).astype(int)
)
data["alert"] = np.select(
    [data["risk_score"] >= 3, data["risk_score"] == 2],
    ["HIGH", "MEDIUM"], default="LOW"
)
os.makedirs("output", exist_ok=True)
data.to_csv("output/forest_environment_alerts.csv", index=False)
print(data["alert"].value_counts())
print(data.sort_values("risk_score", ascending=False).head(10).to_string(index=False))
print("Saved output/forest_environment_alerts.csv")
