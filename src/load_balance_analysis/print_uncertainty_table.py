import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import project_dir


def get_all_raw_data_for_vw_value(vw: int, normal_csv_dir: str) -> pd.DataFrame:
    dfs = []  # List to store individual DataFrames
    std_list = []  # List to store standard deviations
    print(f"\nvw:{vw}")
    for folder in os.listdir(normal_csv_dir):
        # make sure only the alpha folder is processed
        if "alpha" not in folder:
            continue

        print(f"folder: {folder}")
        file_path = Path(normal_csv_dir) / folder / f"vw_{vw}.csv"

        try:
            df = pd.read_csv(file_path)
            dfs.append(df)  # Add DataFrame to list
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            print(f"Empty CSV file: {file_path}")

        for sideslip in df["sideslip"].unique():
            df_local = df[df["sideslip"] == sideslip]
            std_list.append(df_local.std())

    # If no DataFrames were loaded, return empty DataFrame
    if not dfs:
        return pd.DataFrame()

    # Concatenate all DataFrames
    return pd.concat(dfs, ignore_index=True), pd.concat(std_list)


def print_std_SNR_uncertainty(project_dir: str) -> pd.DataFrame:
    # # Load SNR data
    # path_to_csv = (
    #     Path(project_dir)
    #     / "processed_data"
    #     / "uncertainty_table"
    #     / "stats_all_jelle.csv"
    # )
    # df_all = pd.read_csv(path_to_csv)

    # df_vw_5 = df_all.loc[df_all["vw"] == 5]
    # df_vw_10 = df_all.loc[df_all["vw"] == 10]
    # df_vw_15 = df_all.loc[df_all["vw"] == 15]
    # df_vw_20 = df_all.loc[df_all["vw"] == 20]

    normal_csv_dir = Path(project_dir) / "processed_data" / "normal_csv"
    df_vw_5, std_list_vw_5 = get_all_raw_data_for_vw_value(5, normal_csv_dir)
    df_vw_10, std_list_vw_10 = get_all_raw_data_for_vw_value(10, normal_csv_dir)
    df_vw_15, std_list_vw_15 = get_all_raw_data_for_vw_value(15, normal_csv_dir)
    df_vw_20, std_list_vw_20 = get_all_raw_data_for_vw_value(20, normal_csv_dir)

    df_list = [df_vw_5, df_vw_10, df_vw_15, df_vw_20]
    vw_list = ["5", "10", "15", "20"]
    Reynolds_list = ["1.4", "2.8", "4.2", "5.6"]
    df_std_list = [std_list_vw_5, std_list_vw_10, std_list_vw_15, std_list_vw_20]

    for df, vw, Re, std_list in zip(df_list, vw_list, Reynolds_list, df_std_list):
        print(f"\n--> Re: {Re}x10^5")
        # print(f"Number of samples: {len(df)}")

        print(
            f'CF_X SNR mean: {abs(df["SNR_CF_X"].mean()):.3f}, std: {df["SNR_CF_X"].std():.3f}'
        )
        # print(f'F_X min: {df["SNR_CF_X"].min():.3f}, max: {df["SNR_CF_X"].max():.3f}')
        # print(
        #     f'F_X_load_ratio mean: {df["F_X_load_ratio"].mean():.3f}, std: {df["F_X_load_ratio"].std():.3f}'
        # )
        # print(
        #     f'F_X_load_ratio min: {df["F_X_load_ratio"].min():.3f}, max: {df["F_X_load_ratio"].max():.3f}'
        # )

        print(
            f'CF_Y SNR mean: {abs(df["SNR_CF_Y"].mean()):.3f}, std: {df["SNR_CF_Y"].std():.3f}'
        )
        # print(f'F_Y min: {df["SNR_CF_Y"].min():.3f}, max: {df["SNR_CF_Y"].max():.3f}')
        # print(
        #     f'F_Y_load_ratio mean: {df["F_Y_load_ratio"].mean():.3f}, std: {df["F_Y_load_ratio"].std():.3f}'
        # )
        # print(
        #     f'F_Y_load_ratio min: {df["F_Y_load_ratio"].min():.3f}, max: {df["F_Y_load_ratio"].max():.3f}'
        # )

        print(
            f'CF_Z SNR mean: {abs(df["SNR_CF_Z"].mean()):.3f}, std: {df["SNR_CF_Z"].std():.3f}'
        )
        # print(f'F_Z min: {df["SNR_CF_Z"].min():.3f}, max: {df["SNR_CF_Z"].max():.3f}')
        # print(
        #     f'F_Z_load_ratio mean: {df["F_Z_load_ratio"].mean():.3f}, std: {df["F_Z_load_ratio"].std():.3f}'
        # )
        # print(
        #     f'F_Z_load_ratio min: {df["F_Z_load_ratio"].min():.3f}, max: {df["F_Z_load_ratio"].max():.3f}'
        # )

        print(
            f'CM_X SNR mean: {abs(df["SNR_CM_X"].mean()):.3f}, std: {df["SNR_CM_X"].std():.3f}'
        )
        # print(f'M_X min: {df["SNR_CM_X"].min():.3f}, max: {df["SNR_CM_X"].max():.3f}')
        # print(
        #     f'M_X_load_ratio mean: {df["M_X_load_ratio"].mean():.3f}, std: {df["M_X_load_ratio"].std():.3f}'
        # )
        # print(
        #     f'M_X_load_ratio min: {df["M_X_load_ratio"].min():.3f}, max: {df["M_X_load_ratio"].max():.3f}'
        # )

        print(
            f'CM_Y SNR mean: {abs(df["SNR_CM_Y"].mean()):.3f}, std: {df["SNR_CM_Y"].std():.3f}'
        )
        # print(f'M_Y min: {df["SNR_CM_Y"].min():.3f}, max: {df["SNR_CM_Y"].max():.3f}')
        # print(
        #     f'M_Y_load_ratio mean: {df["M_Y_load_ratio"].mean():.3f}, std: {df["M_Y_load_ratio"].std():.3f}'
        # )
        # print(
        #     f'M_Y_load_ratio min: {df["M_Y_load_ratio"].min():.3f}, max: {df["M_Y_load_ratio"].max():.3f}'
        # )

        print(
            f'CM_Z SNR mean: {abs(df["SNR_CM_Z"].mean()):.3f}, std: {df["SNR_CM_Z"].std():.3f}'
        )
        # print(f'M_Z min: {df["SNR_CM_Z"].min():.3f}, max: {df["SNR_CM_Z"].max():.3f}')
        # print(
        #     f'M_Z_load_ratio mean: {df["M_Z_load_ratio"].mean():.3f}, std: {df["M_Z_load_ratio"].std():.3f}'
        # )
        # print(
        #     f'M_Z_load_ratio min: {df["M_Z_load_ratio"].min():.3f}, max: {df["M_Z_load_ratio"].max():.3f}'
        # )
        # New print statements for additional parameters
        # print(
        #     f'C_D_std mean: {df["C_D_std"].mean():.3f}, std: {df["C_D_std"].std():.3f}'
        # )
        # print(
        #     f'C_S_std mean: {df["C_S_std"].mean():.3f}, std: {df["C_S_std"].std():.3f}'
        # )
        # print(
        #     f'C_L_std mean: {df["C_L_std"].mean():.3f}, std: {df["C_L_std"].std():.3f}'
        # )
        # print(
        #     f'C_roll_std mean: {df["C_roll_std"].mean():.3f}, std: {df["C_roll_std"].std():.3f}'
        # )
        # print(
        #     f'C_pitch_std mean: {df["C_pitch_std"].mean():.3f}, std: {df["C_pitch_std"].std():.3f}'
        # )
        # print(
        #     f'C_yaw_std mean: {df["C_yaw_std"].mean():.3f}, std: {df["C_yaw_std"].std():.3f}'
        # )
        # New print statements for additional parameters
        print(
            f'C_D_std mean: {std_list["C_D"].mean():.3f}, std: {std_list["C_D"].std():.3f}'
        )
        print(
            f'C_S_std mean: {std_list["C_S"].mean():.3f}, std: {std_list["C_S"].std():.3f}'
        )
        print(
            f'C_L_std mean: {std_list["C_L"].mean():.3f}, std: {std_list["C_L"].std():.3f}'
        )
        print(
            f'C_roll_std mean: {std_list["C_roll"].mean():.3f}, std: {std_list["C_roll"].std():.3f}'
        )
        print(
            f'C_pitch_std mean: {std_list["C_pitch"].mean():.3f}, std: {std_list["C_pitch"].std():.3f}'
        )
        print(
            f'C_yaw_std mean: {std_list["C_yaw"].mean():.3f}, std: {std_list["C_yaw"].std():.3f}'
        )


def main(project_dir: Path):
    print_std_SNR_uncertainty(project_dir)


if __name__ == "__main__":

    main(project_dir)
