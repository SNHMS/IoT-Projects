import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

rng = np.random.default_rng(17)
n = 1200
temperature = rng.normal(31, 6, n)
battery = rng.uniform(5, 100, n)
signal = rng.uniform(10, 100, n)
packet_loss = rng.uniform(0, 25, n)
status = ((temperature > 39) | (battery < 15) | (signal < 25) | (packet_loss > 18)).astype(int)

data = pd.DataFrame({
    "temperature": temperature,
    "battery": battery,
    "signal_strength": signal,
    "packet_loss": packet_loss,
    "needs_attention": status
})
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns="needs_attention"), data["needs_attention"],
    test_size=0.25, random_state=17, stratify=data["needs_attention"]
)
model = RandomForestClassifier(n_estimators=120, random_state=17)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions, zero_division=0))
os.makedirs("output", exist_ok=True)
data.assign(predicted_status=np.r_[model.predict(data.drop(columns="needs_attention"))]).to_csv(
    "output/device_status_predictions.csv", index=False
)
print("Saved output/device_status_predictions.csv")
