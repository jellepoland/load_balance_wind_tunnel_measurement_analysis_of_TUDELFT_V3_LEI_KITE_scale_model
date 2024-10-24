import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import project_dir
from load_balance_analysis.functions_statistics import (
    hac_newey_west_confidence_interval,
)


# def get_all_raw_data_for_vw_value(vw: int, normal_csv_dir: str) -> pd.DataFrame:
#     dfs = []  # List to store individual DataFrames
#     std_list = []  # List to store standard deviations
#     print(f"\nvw:{vw}")
#     for folder in os.listdir(normal_csv_dir):
#         # make sure only the alpha folder is processed
#         if "alpha" not in folder:
#             continue

#         print(f"folder: {folder}")
#         file_path = Path(normal_csv_dir) / folder / f"vw_{vw}.csv"

#         try:
#             df = pd.read_csv(file_path)
#             dfs.append(df)  # Add DataFrame to list
#         except FileNotFoundError:
#             print(f"File not found: {file_path}")
#         except pd.errors.EmptyDataError:
#             print(f"Empty CSV file: {file_path}")

#         for sideslip in df["sideslip"].unique():
#             if sideslip == 4:
#                 df_local = df[df["sideslip"] == sideslip]
#                 std_list.append(df_local.std())

#     # If no DataFrames were loaded, return empty DataFrame
#     if not dfs:
#         return pd.DataFrame()

#     # Concatenate all DataFrames
#     return pd.concat(dfs, ignore_index=True), pd.concat(std_list)


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


def print_std_SNR_uncertainty(project_dir: str) -> pd.DataFrame:
    # Load data
    save_csv_dir = Path(project_dir) / "processed_data" / "uncertainty_table"

    df_vw_5 = pd.read_csv(save_csv_dir / "df_vw_5.csv")
    df_vw_10 = pd.read_csv(save_csv_dir / "df_vw_10.csv")
    df_vw_15 = pd.read_csv(save_csv_dir / "df_vw_15.csv")
    df_vw_20 = pd.read_csv(save_csv_dir / "df_vw_20.csv")

    df_list = [df_vw_5, df_vw_10, df_vw_15, df_vw_20]
    vw_list = ["5", "10", "15", "20"]
    Reynolds_list = ["1.4", "2.8", "4.2", "5.6"]

    for df, vw, Re in zip(df_list, vw_list, Reynolds_list):
        print(f"\n--> Re: {Re}x10^5")
        print(f"Number of samples: {len(df)}")

        # Compute relative standard deviation
        rel_std_mean_list = []
        for col in ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]:
            # print average mean
            mean_mean = np.abs(df[f"{col}_mean"].mean())
            # print(f"{col} mean: {mean_mean:.2f}")

            # print average std
            std_mean = df[f"{col}_std"].mean()
            # print(f"{col} std: {std_mean:.2f}")

            # print average rel std
            rel_std_mean = std_mean / mean_mean
            # print(f"{col} rel std: {rel_std_mean:.2f}")

            # Append to list
            # rel_std_mean_list.append(rel_std_mean)

            # calc. CI
            mean_ci = np.abs(df[f"{col}_ci"].mean())

            # rel CI
            rel_ci_mean = mean_ci / mean_mean

            # printing
            print(f"{col} --> rel_std: {rel_std_mean:.2f}, rel_ci: {rel_ci_mean:.2f}")
            rel_std_mean_list.append(rel_ci_mean)

        print(f"FORCES Average rel std: {np.mean(rel_std_mean_list[0:3]):.2f}")
        print(f"MOMENTS Average rel std: {np.mean(rel_std_mean_list[3:]):.2f}")
        print(f"Average rel std: {np.mean(rel_std_mean_list):.2f}")


def main(project_dir: Path):
    # load_and_store_files_per_vw(project_dir)

    print_std_SNR_uncertainty(project_dir)


if __name__ == "__main__":

    main(project_dir)


# def print_std_SNR_uncertainty(project_dir: str) -> pd.DataFrame:
#     # # Load SNR data
#     # path_to_csv = (
#     #     Path(project_dir)
#     #     / "processed_data"
#     #     / "uncertainty_table"
#     #     / "stats_all_jelle.csv"
#     # )
#     # df_all = pd.read_csv(path_to_csv)

#     # df_vw_5 = df_all.loc[df_all["vw"] == 5]
#     # df_vw_10 = df_all.loc[df_all["vw"] == 10]
#     # df_vw_15 = df_all.loc[df_all["vw"] == 15]
#     # df_vw_20 = df_all.loc[df_all["vw"] == 20]

#     normal_csv_dir = Path(project_dir) / "processed_data" / "normal_csv"
#     df_vw_5, std_list_vw_5 = get_all_raw_data_for_vw_value(5, normal_csv_dir)
#     df_vw_10, std_list_vw_10 = get_all_raw_data_for_vw_value(10, normal_csv_dir)
#     df_vw_15, std_list_vw_15 = get_all_raw_data_for_vw_value(15, normal_csv_dir)
#     df_vw_20, std_list_vw_20 = get_all_raw_data_for_vw_value(20, normal_csv_dir)

#     df_list = [df_vw_5, df_vw_10, df_vw_15, df_vw_20]
#     vw_list = ["5", "10", "15", "20"]
#     Reynolds_list = ["1.4", "2.8", "4.2", "5.6"]
#     df_std_list = [std_list_vw_5, std_list_vw_10, std_list_vw_15, std_list_vw_20]

#     for df, vw, Re, std_list in zip(df_list, vw_list, Reynolds_list, df_std_list):
#         print(f"\n--> Re: {Re}x10^5")
#         # print(f"Number of samples: {len(df)}")

#         print(
#             f'CF_X SNR mean: {abs(df["SNR_CF_X"].mean()):.3f}, std: {df["SNR_CF_X"].std():.3f}'
#         )
#         # print(f'F_X min: {df["SNR_CF_X"].min():.3f}, max: {df["SNR_CF_X"].max():.3f}')
#         # print(
#         #     f'F_X_load_ratio mean: {df["F_X_load_ratio"].mean():.3f}, std: {df["F_X_load_ratio"].std():.3f}'
#         # )
#         # print(
#         #     f'F_X_load_ratio min: {df["F_X_load_ratio"].min():.3f}, max: {df["F_X_load_ratio"].max():.3f}'
#         # )

#         print(
#             f'CF_Y SNR mean: {abs(df["SNR_CF_Y"].mean()):.3f}, std: {df["SNR_CF_Y"].std():.3f}'
#         )
#         # print(f'F_Y min: {df["SNR_CF_Y"].min():.3f}, max: {df["SNR_CF_Y"].max():.3f}')
#         # print(
#         #     f'F_Y_load_ratio mean: {df["F_Y_load_ratio"].mean():.3f}, std: {df["F_Y_load_ratio"].std():.3f}'
#         # )
#         # print(
#         #     f'F_Y_load_ratio min: {df["F_Y_load_ratio"].min():.3f}, max: {df["F_Y_load_ratio"].max():.3f}'
#         # )

#         print(
#             f'CF_Z SNR mean: {abs(df["SNR_CF_Z"].mean()):.3f}, std: {df["SNR_CF_Z"].std():.3f}'
#         )
#         # print(f'F_Z min: {df["SNR_CF_Z"].min():.3f}, max: {df["SNR_CF_Z"].max():.3f}')
#         # print(
#         #     f'F_Z_load_ratio mean: {df["F_Z_load_ratio"].mean():.3f}, std: {df["F_Z_load_ratio"].std():.3f}'
#         # )
#         # print(
#         #     f'F_Z_load_ratio min: {df["F_Z_load_ratio"].min():.3f}, max: {df["F_Z_load_ratio"].max():.3f}'
#         # )

#         print(
#             f'CM_X SNR mean: {abs(df["SNR_CM_X"].mean()):.3f}, std: {df["SNR_CM_X"].std():.3f}'
#         )
#         # print(f'M_X min: {df["SNR_CM_X"].min():.3f}, max: {df["SNR_CM_X"].max():.3f}')
#         # print(
#         #     f'M_X_load_ratio mean: {df["M_X_load_ratio"].mean():.3f}, std: {df["M_X_load_ratio"].std():.3f}'
#         # )
#         # print(
#         #     f'M_X_load_ratio min: {df["M_X_load_ratio"].min():.3f}, max: {df["M_X_load_ratio"].max():.3f}'
#         # )

#         print(
#             f'CM_Y SNR mean: {abs(df["SNR_CM_Y"].mean()):.3f}, std: {df["SNR_CM_Y"].std():.3f}'
#         )
#         # print(f'M_Y min: {df["SNR_CM_Y"].min():.3f}, max: {df["SNR_CM_Y"].max():.3f}')
#         # print(
#         #     f'M_Y_load_ratio mean: {df["M_Y_load_ratio"].mean():.3f}, std: {df["M_Y_load_ratio"].std():.3f}'
#         # )
#         # print(
#         #     f'M_Y_load_ratio min: {df["M_Y_load_ratio"].min():.3f}, max: {df["M_Y_load_ratio"].max():.3f}'
#         # )

#         print(
#             f'CM_Z SNR mean: {abs(df["SNR_CM_Z"].mean()):.3f}, std: {df["SNR_CM_Z"].std():.3f}'
#         )
#         ## Average SNR
#         SNR_mean = (
#             abs(df["SNR_CF_X"].mean())
#             + abs(df["SNR_CF_Y"].mean())
#             + abs(df["SNR_CF_Z"].mean())
#             + abs(df["SNR_CM_X"].mean())
#             + abs(df["SNR_CM_Y"].mean())
#             + abs(df["SNR_CM_Z"].mean())
#         ) / 6
#         print(f"Average SNR: {SNR_mean}")

#         # print(f'M_Z min: {df["SNR_CM_Z"].min():.3f}, max: {df["SNR_CM_Z"].max():.3f}')
#         # print(
#         #     f'M_Z_load_ratio mean: {df["M_Z_load_ratio"].mean():.3f}, std: {df["M_Z_load_ratio"].std():.3f}'
#         # )
#         # print(
#         #     f'M_Z_load_ratio min: {df["M_Z_load_ratio"].min():.3f}, max: {df["M_Z_load_ratio"].max():.3f}'
#         # )
#         # New print statements for additional parameters
#         # print(
#         #     f'C_D_std mean: {df["C_D_std"].mean():.3f}, std: {df["C_D_std"].std():.3f}'
#         # )
#         # print(
#         #     f'C_S_std mean: {df["C_S_std"].mean():.3f}, std: {df["C_S_std"].std():.3f}'
#         # )
#         # print(
#         #     f'C_L_std mean: {df["C_L_std"].mean():.3f}, std: {df["C_L_std"].std():.3f}'
#         # )
#         # print(
#         #     f'C_roll_std mean: {df["C_roll_std"].mean():.3f}, std: {df["C_roll_std"].std():.3f}'
#         # )
#         # print(
#         #     f'C_pitch_std mean: {df["C_pitch_std"].mean():.3f}, std: {df["C_pitch_std"].std():.3f}'
#         # )
#         # print(
#         #     f'C_yaw_std mean: {df["C_yaw_std"].mean():.3f}, std: {df["C_yaw_std"].std():.3f}'
#         # )
#         # New print statements for additional parameters
#         print(
#             f'C_D_std mean: {std_list["C_D"].mean():.3f}, std: {std_list["C_D"].std():.3f}'
#         )
#         print(
#             f'C_S_std mean: {std_list["C_S"].mean():.3f}, std: {std_list["C_S"].std():.3f}'
#         )
#         print(
#             f'C_L_std mean: {std_list["C_L"].mean():.3f}, std: {std_list["C_L"].std():.3f}'
#         )
#         print(
#             f'C_roll_std mean: {std_list["C_roll"].mean():.3f}, std: {std_list["C_roll"].std():.3f}'
#         )
#         print(
#             f'C_pitch_std mean: {std_list["C_pitch"].mean():.3f}, std: {std_list["C_pitch"].std():.3f}'
#         )
#         print(
#             f'C_yaw_std mean: {std_list["C_yaw"].mean():.3f}, std: {std_list["C_yaw"].std():.3f}'
#         )
#         ## average STD
#         std_mean = (
#             std_list["C_D"].mean()
#             + std_list["C_S"].mean()
#             + std_list["C_L"].mean()
#             + std_list["C_roll"].mean()
#             + std_list["C_pitch"].mean()
#             + std_list["C_yaw"].mean()
#         ) / 6
#         print(f"Average STD: {std_mean}")

#         # ### calculating the standard deviation for a specific sideslip value
#         # for sideslip in [-20, -8, 0, 8, 20]:
#         #     print(f"\nsideslip:{sideslip}")
#         #     df_sideslip = df.loc[df["sideslip"] == sideslip]
#         #     col_list = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
#         #     for col in col_list:
#         #         mean = df_sideslip[col].mean()
#         #         std = df_sideslip[col].std()
#         #         print(f"{col} std/mean: {std/mean:.2f}")

#         # print(df.columns)
#         # ### calculating relative standard deviation wrt mean for each individual measurement within 1-windspeed
#         # for aoa in df["aoa_kite"].unique():
#         #     print(f"\naoa:{aoa}")
#         #     df_aoa = df.loc[df["aoa_kite"] == aoa]
#         #     for sideslip in [-20, -8, 0, 8, 20]:
#         #         print(f"\nsideslip:{sideslip}")
#         #         df_sideslip = df_aoa.loc[df_aoa["sideslip"] == sideslip]
#         #         col_list = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
#         #         for col in col_list:
#         #             mean = df_sideslip[col].mean()
#         #             std = df_sideslip[col].std()
#         #             print(f"{col} std/mean: {std/mean:.2f}")

#         # Initialize lists to store all relative standard deviations per column
#         rel_stds = {
#             col: [] for col in ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
#         }

#         # Calculate relative standard deviations for each condition
#         for aoa in df["aoa_kite"].unique():
#             df_aoa = df.loc[df["aoa_kite"] == aoa]
#             for sideslip in df["sideslip"].unique():
#                 df_sideslip = df_aoa.loc[df_aoa["sideslip"] == sideslip]

#                 # Only process if we have data for this condition
#                 if not df_sideslip.empty:
#                     for col in ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]:
#                         mean = df_sideslip[col].mean()
#                         std = df_sideslip[col].std()
#                         if mean != 0:  # Avoid division by zero
#                             rel_stds[col].append(std / mean)

#         # Calculate and print average relative standard deviations
#         print("\nAverage relative standard deviations per column:")
#         for col in ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]:
#             avg_rel_std = np.mean(rel_stds[col])
#             print(f"{col}: {avg_rel_std:.2f}")
