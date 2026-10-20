import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(41)
hours = np.arange(24)
temperature = 24 + 5 * np.sin((hours - 7) * np.pi / 12) + rng.normal(0, 0.7, 24)
power = 0.5 + 1.2 * ((hours >= 6) & (hours <= 9)) + 1.8 * ((hours >= 17) & (hours <= 22))
power = power + rng.uniform(0, 0.4, 24)
humidity = 68 - temperature * 0.45 + rng.normal(0, 2, 24)
data = pd.DataFrame({"hour": hours, "temperature": temperature, "humidity": humidity, "power_kwh": power})
os.makedirs("output", exist_ok=True)
data.to_csv("output/home_sensor_readings.csv", index=False)

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
axes[0].plot(hours, temperature, marker="o")
axes[0].set_ylabel("Temperature")
axes[1].plot(hours, humidity, marker="o")
axes[1].set_ylabel("Humidity")
axes[2].bar(hours, power)
axes[2].set_ylabel("Power kWh")
axes[2].set_xlabel("Hour")
fig.suptitle("Smart Home Sensor Dashboard")
fig.tight_layout()
fig.savefig("output/smart_home_dashboard.png", dpi=150)
plt.show()
print(data.round(2).head())
