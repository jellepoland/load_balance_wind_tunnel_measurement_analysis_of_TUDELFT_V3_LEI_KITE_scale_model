import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from utils import project_dir


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


def print_repeatability_uncertainty(project_dir: str) -> pd.DataFrame:

    print("\n--> Repeatability uncertainty:")
    # Read the interpolation coefficients
    path_support_struc_aero_interp_coeffs = (
        Path(project_dir) / "processed_data" / "without_kite_interp_coeff.csv"
    )
    support_struc_aero_interp_coeffs = pd.read_csv(
        path_support_struc_aero_interp_coeffs
    )

    # Read the labbook data
    path_labbook_double = (
        Path(project_dir)
        / "processed_data"
        / "uncertainty_table"
        / "labbook_double.csv"
    )
    data = pd.read_csv(path_labbook_double, delimiter=";")
    data["measurement"] = data.groupby("Filename").cumcount()

    # Define the column names for the data files
    column_names = ["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "error"]

    # Initialize an empty list to hold DataFrames
    dataframes = []

    # Read another data file and add information
    path_double_aoa_12_vw_20_unsteady = (
        Path(project_dir)
        / "data"
        / "uncertainty_table"
        / "double_aoa_12_vw_20_unsteady.txt"
    )
    df = pd.read_csv(
        path_double_aoa_12_vw_20_unsteady, delimiter="\t", names=column_names
    )  # Assuming tab-separated values for this example

    # Drop the 'error' column
    df = df.drop(columns=["error"])

    # Add a measurement column where each index is repeated every 19800 rows
    df["measurement"] = np.arange(len(df)) // 19800

    merged_df = pd.merge(data, df, on=["measurement"])
    merged_df.rename(columns={"vw": "vw_actual"}, inplace=True)
    merged_df["vw"] = np.ceil(merged_df["vw_actual"])
    merged_df = merged_df[merged_df["vw"] != 0]

    zerorun_ss_20 = np.array(
        [3.022004, 3.824555, 800.787406, 3.403333, 172.332697, 0.294079]
    )
    zerorun_ss_0 = np.array(
        [3.412569, 5.041744, 800.806084, 4.026951, 172.121360, 0.034760]
    )
    zerorun_ss_n20 = np.array(
        [3.337865, 6.318882, 800.781853, 4.693001, 172.152581, -0.232359]
    )

    cols = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    new_cols = ["Cd", "Cs", "Cl", "Cmx", "Cmy", "Cmz"]

    # Create new columns for the zero run adjusted values
    for new_col, col in zip(new_cols, cols):
        merged_df[new_col] = merged_df[col]

    # Subtract the zero run values from the new columns
    for sideslip, zero_runs in zip(
        [-20, 0, 20], [zerorun_ss_n20, zerorun_ss_0, zerorun_ss_20]
    ):
        for new_col, col, zero_run in zip(new_cols, cols, zero_runs):
            merged_df.loc[merged_df["sideslip"] == sideslip, new_col] -= zero_run

    # Nondimensionalize the columns "Cd", "Cs", "Cl", "Cmx", "Cmy", and "Cmz"
    rho = merged_df["Density"]
    V = merged_df["vw_actual"]
    S_ref = 0.46
    c_ref = 0.4

    # Nondimensionalize the force columns
    merged_df["Cd"] /= 0.5 * rho * V**2 * S_ref
    merged_df["Cs"] /= 0.5 * rho * V**2 * S_ref
    merged_df["Cl"] /= 0.5 * rho * V**2 * S_ref

    # Nondimensionalize the moment columns
    merged_df["Cmx"] /= 0.5 * rho * V**2 * S_ref * c_ref
    merged_df["Cmy"] /= 0.5 * rho * V**2 * S_ref * c_ref
    merged_df["Cmz"] /= 0.5 * rho * V**2 * S_ref * c_ref

    # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
    aoa = merged_df["aoa"].unique()[0] - 7.25
    v = 20

    supp_coeffs = support_struc_aero_interp_coeffs[
        support_struc_aero_interp_coeffs["vw"] == 20
    ]
    for k in merged_df["sideslip"].unique():
        # select support structure aero coefficients for this sideslip
        c_s = supp_coeffs[supp_coeffs["sideslip"] == k]
        F_x = c_s.loc[c_s["channel"] == "Cx", ["a", "b", "c"]]
        F_y = c_s.loc[c_s["channel"] == "Cy", ["a", "b", "c"]]
        F_z = c_s.loc[c_s["channel"] == "Cz", ["a", "b", "c"]]
        M_x = c_s.loc[c_s["channel"] == "Cmx", ["a", "b", "c"]]
        M_y = c_s.loc[c_s["channel"] == "Cmy", ["a", "b", "c"]]
        M_z = c_s.loc[c_s["channel"] == "Cmz", ["a", "b", "c"]]

        # compute support structure aero coefficients for this wind speed, sideslip and angle of attack combination
        C_Fx_s = np.array(F_x["a"] * (aoa**2) + F_x["b"] * aoa + F_x["c"])[0]
        C_Fy_s = np.array(F_y["a"] * (aoa**2) + F_y["b"] * aoa + F_y["c"])[0]
        C_Fz_s = np.array(F_z["a"] * (aoa**2) + F_z["b"] * aoa + F_z["c"])[0]
        C_Mx_s = np.array(M_x["a"] * (aoa**2) + M_x["b"] * aoa + M_x["c"])[0]
        C_My_s = np.array(M_y["a"] * (aoa**2) + M_y["b"] * aoa + M_y["c"])[0]
        C_Mz_s = np.array(M_z["a"] * (aoa**2) + M_z["b"] * aoa + M_z["c"])[0]

        # subtract support structure aero coefficients for this wind speed, sideslip and aoa combination from merged_df
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cd"
        ] -= C_Fx_s
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cs"
        ] -= C_Fy_s
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cl"
        ] -= C_Fz_s
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cmx"
        ] -= C_Mx_s
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cmy"
        ] -= C_My_s
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cmz"
        ] -= C_Mz_s

    # Translate coordinate system
    # Parameters necessary to translate moments (aka determine position of cg)
    x_hinge = 441.5
    z_hinge = 1359
    l_cg = 625.4
    alpha_cg = np.deg2rad(merged_df["aoa"] - 23.82)
    x_cg = l_cg * np.cos(alpha_cg)
    z_cg = l_cg * np.sin(alpha_cg)
    A_side = 5.646
    A_ref = 19.753
    Cs_scale = A_ref / A_side

    merged_df["x_kite"] = (x_hinge + x_cg) / (1000 * c_ref)
    merged_df["z_kite"] = (z_hinge + z_cg) / (1000 * c_ref)

    # Rotation of coordinate system: force coefficients change
    merged_df["C_L"] = merged_df["Cl"] * -1
    merged_df["C_S"] = merged_df["Cs"] * -1
    merged_df["C_D"] = merged_df["Cd"]

    merged_df["C_roll"] = merged_df["Cmx"] - merged_df["Cs"] * merged_df["z_kite"]
    merged_df["C_pitch"] = (
        -merged_df["Cmy"]
        + merged_df["Cl"] * merged_df["x_kite"]
        - merged_df["Cd"] * merged_df["z_kite"]
    )
    merged_df["C_yaw"] = -merged_df["Cmz"] - merged_df["Cs"] * merged_df["x_kite"]

    # Define the coefficients and other relevant columns
    coefficients = ["Cd", "Cs", "Cl", "Cmx", "Cmy", "Cmz"]
    yaxis_names = ["C_D", "C_S", "C_L", "C_{roll}", "C_{pitch}", "C_{yaw}"]
    coefficients_plot_xaxis = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
    plot_titles = [
        "Drag coefficient",
        "Side-force coefficient",
        "Lift coefficient",
        "Roll moment coefficient",
        "Pitching moment coefficient",
        "Yawing moment coefficient",
    ]
    sideslip_angles = [-20, 0, 20]

    merged_df["measurement"] = (
        merged_df.groupby("sideslip").cumcount() % 3 + 1
    )  # Measurement labels: 1, 2, 3

    for sideslip in sideslip_angles:
        df_local = merged_df[merged_df["sideslip"] == sideslip]
        df_local_m1 = df_local[df_local["measurement"] == 1]
        df_local_m2 = df_local[df_local["measurement"] == 2]
        df_local_m3 = df_local[df_local["measurement"] == 3]
        for coeff in coefficients_plot_xaxis:
            # print(
            #     f"coeff: {coeff}, m1_mean: {df_local_m1[coeff].mean()}, m2_mean: {df_local_m2[coeff].mean()}, m3_mean: {df_local_m3[coeff].mean()}"
            # )
            print(
                f"coeff: {coeff}, std of mean: {1e4*np.std([df_local_m1[coeff].mean(), df_local_m2[coeff].mean(), df_local_m3[coeff].mean()]):.3f} x 1e-4"
            )


def main(project_dir):
    print_std_SNR_uncertainty(project_dir)
    print_repeatability_uncertainty(project_dir)


if __name__ == "__main__":

    main(project_dir)