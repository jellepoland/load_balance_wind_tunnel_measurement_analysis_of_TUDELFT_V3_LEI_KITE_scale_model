import os
import pandas as pd
import numpy as np
from load_balance_analysis.functions_utils import project_dir
from pathlib import Path


def printing_stats(project_dir: Path) -> None:
    # Path to the root folder (modify this with the correct path on your system)
    root_folder = Path(project_dir) / "processed_data" / "normal_csv"

    # Initialize variables to track global statistics
    columns = ["F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]
    global_min = {col: float("inf") for col in columns}
    global_max = {col: float("-inf") for col in columns}
    global_sum = {col: 0 for col in columns}
    global_count = 0
    all_values = {col: [] for col in columns}

    # Iterate through all subdirectories in the root folder
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.startswith("raw_normal_aoa"):  # Only process the raw files
                file_path = os.path.join(root, file)

                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path, sep=",")

                # Extract relevant columns
                subset = df[columns]

                # Update global statistics
                for col in columns:
                    global_min[col] = min(global_min[col], subset[col].min())
                    global_max[col] = max(global_max[col], subset[col].max())
                    global_sum[col] += subset[col].sum()
                    all_values[col].extend(subset[col].tolist())

                global_count += len(subset)

    # Calculate global mean and median
    global_mean = {col: global_sum[col] / global_count for col in columns}
    global_median = {col: np.median(all_values[col]) for col in columns}

    # Output the global statistics for each column
    print("Global Minimums:")
    for col in columns:
        print(f"{col}: {global_min[col]:.2f}")

    print("\nGlobal Maximums:")
    for col in columns:
        print(f"{col}: {global_max[col]:.2f}")

    print("\nGlobal Means:")
    for col in columns:
        print(f"{col}: {global_mean[col]:.2f}")

    print("\nGlobal Medians:")
    for col in columns:
        print(f"{col}: {global_median[col]:.2f}")


def main(project_dir):
    printing_stats(project_dir)


if __name__ == "__main__":
    main(project_dir)
