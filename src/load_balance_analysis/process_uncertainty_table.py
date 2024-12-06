import pandas as pd
from pathlib import Path
import os
import numpy as np
from load_balance_analysis.functions_utils import project_dir
from load_balance_analysis.functions_statistics import (
    hac_newey_west_confidence_interval,
)


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


def main(project_dir: str) -> pd.DataFrame:

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


if __name__ == "__main__":
    main(project_dir)
