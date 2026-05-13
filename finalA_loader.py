from torch.utils.data import DataLoader, TensorDataset
import torch
import time
import pandas as pd

# Create synthetic dataset
x = torch.randn(10000, 100)
y = torch.randint(0, 2, (10000,))

dataset = TensorDataset(x, y)

# Different configurations to test
num_workers_list = [0, 2, 4, 8]
pin_memory_options = [False, True]

results = []

print("Starting benchmarks...\n")

for workers in num_workers_list:
    for pin in pin_memory_options:

        loader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=workers,
            pin_memory=pin
        )

        start_time = time.time()

        for batch_x, batch_y in loader:
            # Simulate GPU transfer
            if torch.cuda.is_available():
                batch_x = batch_x.cuda(non_blocking=True)
                batch_y = batch_y.cuda(non_blocking=True)

        end_time = time.time()

        total_time = end_time - start_time

        results.append({
            "num_workers": workers,
            "pin_memory": pin,
            "time_seconds": round(total_time, 4)
        })

        print(f"workers={workers}, pin_memory={pin} -> {total_time:.4f} sec")

# Create comparison table
df = pd.DataFrame(results)

print("\nComparison Table:")
print(df)

# Optional: Save results
df.to_csv("dataloader_benchmark.csv", index=False)

print("\nBenchmark completed.")
