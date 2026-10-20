import time
import os
import numpy as np
import pandas as pd

count = 500000
start = time.perf_counter()
values = []
for i in range(count):
    values.append((i % 100) * 0.15 + 20)
loop_time = time.perf_counter() - start

start = time.perf_counter()
sensor_values = (np.arange(count) % 100) * 0.15 + 20
array_time = time.perf_counter() - start

result = pd.DataFrame({
    "method": ["Python loop", "NumPy array"],
    "samples": [count, count],
    "seconds": [loop_time, array_time]
})
os.makedirs("output", exist_ok=True)
result.to_csv("output/benchmark_results.csv", index=False)
print(result.to_string(index=False))
print("Results saved to output/benchmark_results.csv")
