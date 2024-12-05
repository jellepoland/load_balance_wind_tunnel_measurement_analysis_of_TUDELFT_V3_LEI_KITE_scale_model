import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import project_dir, save_latex_table
from load_balance_analysis.functions_statistics import (
    hac_newey_west_confidence_interval,
)
from statsmodels.tsa.stattools import acf  # Corrected import
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def analyze_autocorrelation(data, column_name, max_lags=20):
    """
    Create comprehensive autocorrelation analysis plots for time series data.

    Args:
        data: array-like, the time series data to analyze
        column_name: str, name of the column being analyzed
        max_lags: int, maximum number of lags to analyze
    """
    # Create a figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Time Series Plot
    ax1.plot(data, label="Original Data")
    ax1.set_title(f"Time Series: {column_name}")
    ax1.set_xlabel("Time Index")
    ax1.set_ylabel("Value")
    ax1.legend()

    # 2. Autocorrelation Function (ACF) Plot
    plot_acf(
        data,
        lags=max_lags,
        ax=ax2,
        title=f"Autocorrelation Function (ACF): {column_name}",
    )

    # 3. Partial Autocorrelation Function (PACF) Plot
    plot_pacf(
        data,
        lags=max_lags,
        ax=ax3,
        title=f"Partial Autocorrelation (PACF): {column_name}",
    )

    # 4. Lag Plot (t vs t-1)
    ax4.scatter(data[:-1], data[1:], alpha=0.5)
    ax4.set_title(f"Lag Plot (t vs t-1): {column_name}")
    ax4.set_xlabel("Value at t")
    ax4.set_ylabel("Value at t+1")

    plt.tight_layout()
    plt.show()

    # Print numerical autocorrelation values for first few lags
    acf_values = acf(data, nlags=5)
    print(f"\nAutocorrelation values for {column_name}:")
    print("Lag  |  ACF Value")
    print("-" * 20)
    for i, value in enumerate(acf_values):
        print(f"{i:2d}   |  {value:.3f}")


def get_all_raw_data_for_vw_value(vw: int, normal_csv_dir: str) -> pd.DataFrame:
    dfs = []  # List to store processed DataFrames
    print(f"\nvw:{vw}")

    for folder in os.listdir(normal_csv_dir):
        # make sure only the alpha folder is processed
        if "alpha" not in folder:
            continue

        print(f"folder: {folder}")
        file_path = Path(normal_csv_dir) / folder / f"vw_{vw}.csv"

        try:
            df = pd.read_csv(file_path)

            # Create a list to store processed data for each sideslip
            processed_data = []

            # List of columns to calculate statistics for
            stat_columns = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]

            for sideslip in df["sideslip"].unique():
                # Get data for this sideslip
                sideslip_df = df[df["sideslip"] == sideslip]

                # Initialize a dictionary for this row
                row_data = {
                    "sideslip": sideslip,
                    "aoa_kite": sideslip_df["aoa_kite"].iloc[
                        0
                    ],  # Take first value as they should be same
                    "vw": vw,
                }

                # Calculate mean, std, and confidence interval for each column
                for col in stat_columns:
                    # print(f"col: {col}")
                    data = sideslip_df[col].values
                    row_data[f"{col}_mean"] = np.mean(data)
                    row_data[f"{col}_std"] = np.std(data)
                    # checking the autocorrelation
                    # analyze_autocorrelation(data, col)
                    row_data[f"{col}_ci"] = hac_newey_west_confidence_interval(
                        data, confidence_interval=99.99, max_lag=11
                    )

                processed_data.append(row_data)

            # Convert processed data to DataFrame
            if processed_data:
                processed_df = pd.DataFrame(processed_data)
                dfs.append(processed_df)

        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            print(f"Empty CSV file: {file_path}")

    # If no DataFrames were loaded, return empty DataFrame
    if not dfs:
        return pd.DataFrame()

    # Concatenate all processed DataFrames
    return pd.concat(dfs, ignore_index=True)


def load_and_store_files_per_vw(project_dir: str) -> pd.DataFrame:

    normal_csv_dir = Path(project_dir) / "processed_data" / "normal_csv"
    df_vw_5 = get_all_raw_data_for_vw_value(5, normal_csv_dir)
    df_vw_10 = get_all_raw_data_for_vw_value(10, normal_csv_dir)
    df_vw_15 = get_all_raw_data_for_vw_value(15, normal_csv_dir)
    df_vw_20 = get_all_raw_data_for_vw_value(20, normal_csv_dir)

    save_csv_dir = Path(project_dir) / "processed_data" / "uncertainty_table"
    df_vw_5.to_csv(save_csv_dir / "df_vw_5.csv", index=False)
    df_vw_10.to_csv(save_csv_dir / "df_vw_10.csv", index=False)
    df_vw_15.to_csv(save_csv_dir / "df_vw_15.csv", index=False)
    df_vw_20.to_csv(save_csv_dir / "df_vw_20.csv", index=False)


def generate_table(data):
    # Create a DataFrame for LaTeX styling
    table_data = {
        ("Re $\cdot 10^5$", "1.4"): [
            data["1.4"]["C_L"],
            data["1.4"]["C_D"],
            data["1.4"]["C_S"],
            data["1.4"]["C_roll"],
            data["1.4"]["C_pitch"],
            data["1.4"]["C_yaw"],
        ],
        ("Re $\cdot 10^5$", "2.8"): [
            data["2.8"]["C_L"],
            data["2.8"]["C_D"],
            data["2.8"]["C_S"],
            data["2.8"]["C_roll"],
            data["2.8"]["C_pitch"],
            data["2.8"]["C_yaw"],
        ],
        ("Re $\cdot 10^5$", "4.2"): [
            data["4.2"]["C_L"],
            data["4.2"]["C_D"],
            data["4.2"]["C_S"],
            data["4.2"]["C_roll"],
            data["4.2"]["C_pitch"],
            data["4.2"]["C_yaw"],
        ],
        ("Re $\cdot 10^5$", "5.6"): [
            data["5.6"]["C_L"],
            data["5.6"]["C_D"],
            data["5.6"]["C_S"],
            data["5.6"]["C_roll"],
            data["5.6"]["C_pitch"],
            data["5.6"]["C_yaw"],
        ],
    }
    index_labels = [
        r"$\bar{\sigma}_{\textrm{L}}$",
        r"$\bar{\sigma}_{\textrm{L}}$",
        r"$\bar{\sigma}_{\textrm{S}}$",
        r"$\bar{\sigma}_{\textrm{{M,x}}$",
        r"$\bar{\sigma}_{\textrm{{M,y}}$",
        r"$\bar{\sigma}_{\textrm{{M,z}}$",
    ]

    df_table = pd.DataFrame(table_data, index=index_labels)
    df_table.columns = pd.MultiIndex.from_tuples(
        df_table.columns
    )  # Format MultiIndex for column labels
    return df_table


def main(project_dir: str) -> pd.DataFrame:
    # Load data
    save_csv_dir = Path(project_dir) / "processed_data" / "uncertainty_table"

    df_vw_5 = pd.read_csv(save_csv_dir / "df_vw_5.csv")
    df_vw_10 = pd.read_csv(save_csv_dir / "df_vw_10.csv")
    df_vw_15 = pd.read_csv(save_csv_dir / "df_vw_15.csv")
    df_vw_20 = pd.read_csv(save_csv_dir / "df_vw_20.csv")

    df_list = [df_vw_5, df_vw_10, df_vw_15, df_vw_20]
    Reynolds_list = ["1.4", "2.8", "4.2", "5.6"]

    # Dictionary to store data for the table
    table_data = {Re: {} for Re in Reynolds_list}

    for df, Re in zip(df_list, Reynolds_list):
        # Compute relative standard deviation for each coefficient
        for col in ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]:
            mean_mean = np.abs(df[f"{col}_mean"].mean())
            std_mean = df[f"{col}_std"].mean()
            rel_std_mean = std_mean / mean_mean
            table_data[Re][col] = f"{rel_std_mean:.2f}"

    # Generate the table
    table_df = generate_table(table_data)
    save_latex_table(
        table_df,
        Path(project_dir) / "results" / "tables" / "relative_standard_deviation.tex",
    )
    print(f"\n--- Relative standard deviation table ---\n")
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    main(project_dir)
