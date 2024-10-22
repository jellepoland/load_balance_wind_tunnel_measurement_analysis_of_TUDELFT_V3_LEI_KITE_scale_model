import pandas as pd
import numpy as np
from pathlib import Path
import os
from load_balance_analysis.functions_utils import project_dir
from load_balance_analysis.functions_processing import processing_raw_lvm_data_into_csv


def merging_zigzag_csv_files(
    zigzag_data_dir: Path, save_path_merged_zigzag: Path
) -> None:

    all_dfs = []
    for file in os.listdir(zigzag_data_dir):
        if "raw" in file or "lvm" in file:
            continue

        print(f"\n--> Processing file: {file}")
        # Read the csv file
        df = pd.read_csv(Path(zigzag_data_dir) / file)

        # Append the dataframe to the list
        all_dfs.append(df)

    # Concatenate all the dataframes
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Save the merged dataframe to a csv file
    merged_df.to_csv(save_path_merged_zigzag, index=False)


def main():
    S_ref = 0.46
    c_ref = 0.4

    # parameters necessary to translate moments (aka determine position of cg)
    x_hinge = (
        441.5  # x distance between force balance coord. sys. and hinge point in mm
    )
    z_hinge = 1359  # z distance between force balance coord. sys. and hinge point in mm
    l_cg = 625.4  # distance between hinge point and kite CG
    alpha_cg_delta_with_rod = 23.82
    delta_celsius_kelvin = 273.15
    mu_0 = 1.716e-5
    T_0 = 273.15
    C_suth = 110.4
    delta_aoa_rod_to_alpha = 7.25

    # processing all the folders for the normal
    support_struc_aero_interp_coeffs_path = (
        Path(project_dir) / "processed_data" / "interpolation_coefficients.csv"
    )
    is_kite = True
    is_zigzag = True
    print(f"\n Processing all the folders")
    for folder in os.listdir(Path(project_dir) / "processed_data" / "zigzag_csv"):
        if "alpha" not in folder:
            continue
        folder_dir = Path(project_dir) / "processed_data" / "zigzag_csv" / folder
        processing_raw_lvm_data_into_csv(
            folder_dir,
            S_ref,
            c_ref,
            is_kite,
            is_zigzag,
            support_struc_aero_interp_coeffs_path,
            x_hinge,
            z_hinge,
            l_cg,
            alpha_cg_delta_with_rod,
            delta_celsius_kelvin,
            mu_0,
            T_0,
            C_suth,
            delta_aoa_rod_to_alpha,
        )

    # Merging raw files
    processed_data_zigzag_dir = (
        Path(project_dir) / "processed_data" / "zigzag_csv" / "alpha_8.9"
    )
    save_path_lvm_data_processed = (
        Path(processed_data_zigzag_dir) / "lvm_data_processed.csv"
    )
    merging_zigzag_csv_files(processed_data_zigzag_dir, save_path_lvm_data_processed)


if __name__ == "__main__":
    main()
