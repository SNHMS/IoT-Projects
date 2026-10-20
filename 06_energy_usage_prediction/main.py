import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.default_rng(63)
n = 600
temperature = rng.uniform(20, 38, n)
occupants = rng.integers(0, 7, n)
hour = rng.integers(0, 24, n)
appliances = rng.integers(1, 10, n)
energy = 0.25 * temperature + 0.42 * occupants + 0.18 * appliances + 0.06 * np.abs(hour - 13) + rng.normal(0, 0.5, n)
data = pd.DataFrame({"temperature": temperature, "occupants": occupants, "hour": hour, "appliances": appliances, "energy_kwh": energy})
X = data.drop(columns="energy_kwh")
y = data["energy_kwh"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=63)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred), 3))
print("R2:", round(r2_score(y_test, pred), 3))
os.makedirs("output", exist_ok=True)
pd.DataFrame({"actual": y_test, "predicted": pred}).to_csv("output/energy_predictions.csv", index=False)
print("Saved output/energy_predictions.csv")
