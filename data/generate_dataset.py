import pandas as pd
import numpy as np

np.random.seed(42)

hardware = [
    ("GPU", "RTX 3090", 350, 10496, 24),
    ("GPU", "RTX 4090", 450, 16384, 24),
    ("GPU", "A100", 400, 6912, 80),
    ("GPU", "V100", 300, 5120, 32),
    ("CPU", "Xeon Gold", 165, 24, 128),
    ("CPU", "Ryzen 9", 105, 16, 64),
]

model_types = ["CNN", "Transformer", "LLM"]

rows = []

for _ in range(800):
    hw_type, name, tdp, cores, mem = hardware[np.random.randint(len(hardware))]
    model = np.random.choice(model_types)

    runtime = np.random.uniform(0.5, 10)
    utilization = np.random.uniform(0.4, 0.95)
    batch_size = np.random.choice([16, 32, 64, 128])

    energy = (tdp * utilization * runtime) / 1000
    energy += np.random.normal(0, 0.05 * energy)

    co2 = energy * 0.475

    rows.append([
        hw_type, name, tdp, cores, mem,
        model, batch_size, runtime, utilization,
        round(energy, 3), round(co2, 3)
    ])

df = pd.DataFrame(rows, columns=[
    "hardware_type", "device_name", "tdp_watts",
    "num_cores", "memory_gb", "model_type",
    "batch_size", "runtime_hours", "utilization_pct",
    "energy_kwh", "co2_kg"
])

df.to_csv("data/merged_dataset.csv", index=False)
print("Dataset generated:", df.shape)
