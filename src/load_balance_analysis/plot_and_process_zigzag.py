import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from utils import (
    saving_pdf_and_pdf_tex,
    x_axis_labels,
    y_axis_labels,
    project_dir,
)
import re
from collections import defaultdict


def read_lvm(filename: str) -> pd.DataFrame:
    # Read the entire data file
    print(f"filename: {filename}")
    df = pd.read_csv(filename, skiprows=21, delimiter="\t", engine="python")
    df.drop(
        columns=df.columns[-1], inplace=True
    )  # Assuming the last column is to be dropped
    df.columns = ["time", "F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]

    # Extract the filename and remove "_unsteady.lvm" from the end
    df["Filename"] = os.path.basename(filename).replace("_unsteady.lvm", "")

    # Extract angle of attack information if available
    if "aoa" in df.columns:
        df["aoa"] = df["aoa"].fillna(
            method="ffill"
        )  # Fill NaN values with the previous non-NaN value

    # Calculate sample index based on the number of rows and the assumption that each sample has 19800 entries
    num_samples = len(df) // 19800
    df["sample index"] = sum([[i] * 19800 for i in range(num_samples)], [])

    # Select only the first 17 samples
    selected_rows = 19800 * 17  # Select the first 17 samples
    df = df.iloc[:selected_rows]

    return df


def read_all_lvm_into_df(lvm_dir: str) -> list:

    labbook_df = pd.read_csv(lvm_dir, delimiter=";")

    # Remove rows where wind speed is zero from labbook
    # labbook_df = labbook_df[labbook_df["vw"] > 1.1]

    # print(f"Labbook: {labbook_df}")

    zigzag_data_dir = Path(project_dir) / "data" / "zigzag"
    all_data = []
    # Loop through each file in the zigzag directory
    for file in os.listdir(zigzag_data_dir):
        # skipping the files that do NOT end with ".lvm"
        if not file.endswith(".lvm"):
            continue

        print(f"\nProcessing file: {file}")

        # Read the lvm file
        lvm_file_path = Path(zigzag_data_dir) / file
        df = read_lvm(lvm_file_path)

        # Strip lvm from file
        filename = lvm_file_path.stem
        # print(f"Filename: {filename}")

        # Filter labbook for where folder matches the filename
        matching_rows = labbook_df[labbook_df["Filename"] == filename]

        print(f"Matching rows: {matching_rows}")
        n_rows = len(matching_rows)
        print(f"Number of matching rows: {n_rows}")

        # Calculate sample index based on the number of rows and the assumption that each sample has 19800 entries
        print(f"len(df): {len(df)}")
        num_samples = len(df) // 19800
        print(f"Number of samples: {num_samples}")

        if n_rows == 1:
            if num_samples == 1:
                df_sample = df.copy()
            elif num_samples == 2:
                df_sample = df.iloc[19800:].copy()

            last_matching_row = matching_rows.iloc[-1]  # Get the last occurrence
            properties_to_add = last_matching_row.to_dict()
            print(f"properties_to_add: {properties_to_add}")

            # Add properties from labbook to each row in df
            for col, value in properties_to_add.items():
                df_sample[col] = (
                    value  # This will create new columns in df with properties from labbook_df
                )

            all_data.append(df_sample)
        else:
            print(f"\nMore than 1 matching row")
            # Create a new DataFrame for each sample
            for i in range(num_samples):
                # Selecting the sample from the DataFrame
                df_sample = df.iloc[i * 19800 : (i + 1) * 19800].copy()

                # Finding the matching row
                row = matching_rows.iloc[i]
                print(f"Row: {row}")
                properties_to_add = row.to_dict()
                print(f"properties_to_add: {properties_to_add}")

                # Add properties from labbook to each row in df
                for col, value in properties_to_add.items():
                    df_sample[col] = (
                        value  # This will create new columns in df with properties from labbook_df
                    )

                all_data.append(df_sample)

    return all_data


def substracting_runs_with_zero_wind(
    merged_df: pd.DataFrame, zigzag_data_dir: Path
) -> pd.DataFrame:
    # Renaming columns
    merged_df.rename(columns={"vw": "vw_actual"}, inplace=True)

    # Defining the filename
    filename = merged_df["Filename"].unique()[0]
    print(f"filename working with: {filename}")

    if merged_df["Filename"].str.startswith("normal").any():
        filename_vw_0 = "normal_aoa_16_vw_00"
    elif merged_df["Filename"].str.startswith("ZZ").any():
        filename_vw_0 = "ZZnormal_aoa_16_vw_00"
    print(f"filename_vw_0: {filename_vw_0}")

    # Extracting the zero-run data, from txt
    zero_run_path = Path(zigzag_data_dir) / f"{filename_vw_0}.txt"
    # with open(zero_run_path, "r") as file:
    #     zero_run_data = file.read()
    #     print(zero_run_data[2:-1])
    data = []
    with open(zero_run_path, "r") as file:
        for line in file:
            # Split each line into a list of values (assumes tab-separated values)
            values = line.strip().split("\t")
            # Convert strings to floats (or other data types if needed)
            values = [float(value) for value in values]
            # Append the list of values to the data list
            data.append(values)

    zero_run_data = data[0][1:-1]
    cols = ["F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]
    print(f" col of merged_df: {merged_df.head()}")
    # Convert zero_run_data to a pandas Series to match the column names
    zero_run_series = pd.Series(zero_run_data, index=cols)

    # Subtract zero_run_series from the corresponding columns in merged_df
    merged_df[cols] = merged_df[cols] - zero_run_series
    return merged_df


def nondimensionalize(merged_df: pd.DataFrame) -> pd.DataFrame:
    rho = merged_df["Density"]
    V = merged_df["vw_actual"]
    S_ref = 0.46
    c_ref = 0.4

    # Nondimensionalize the force columns
    merged_df["F_X"] /= 0.5 * rho * V**2 * S_ref
    merged_df["F_Y"] /= 0.5 * rho * V**2 * S_ref
    merged_df["F_Z"] /= 0.5 * rho * V**2 * S_ref

    # Nondimensionalize the moment columns
    merged_df["M_X"] /= 0.5 * rho * V**2 * S_ref * c_ref
    merged_df["M_Y"] /= 0.5 * rho * V**2 * S_ref * c_ref
    merged_df["M_Z"] /= 0.5 * rho * V**2 * S_ref * c_ref

    return merged_df


def substract_support_structure_aero_coefficients(
    merged_df: pd.DataFrame, support_struc_aero_interp_coeffs: pd.DataFrame
) -> pd.DataFrame:

    # print(f"columns: {merged_df.columns}")
    # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
    cur_vw = merged_df["vw_actual"].unique()[0]
    # print(f"cur_vw: {np.around(cur_vw)}")
    # print(f'support_struc_aero_interp_coeffs["vw"]: {support_struc_aero_interp_coeffs["vw"].unique()}')
    supp_coeffs = support_struc_aero_interp_coeffs[
        support_struc_aero_interp_coeffs["vw"] == int(np.around(cur_vw))
    ]
    # print(f"supp_coeffs: {supp_coeffs}")
    aoa = merged_df["aoa"].unique()[0] - 7.25

    for k in merged_df["sideslip"].unique():
        # print(f"k: {k}")
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
        merged_df.loc[merged_df["sideslip"] == k, "F_X"] -= C_Fx_s
        merged_df.loc[merged_df["sideslip"] == k, "F_Y"] -= C_Fy_s
        merged_df.loc[merged_df["sideslip"] == k, "F_Z"] -= C_Fz_s
        merged_df.loc[merged_df["sideslip"] == k, "M_X"] -= C_Mx_s
        merged_df.loc[merged_df["sideslip"] == k, "M_Y"] -= C_My_s
        merged_df.loc[merged_df["sideslip"] == k, "M_Z"] -= C_Mz_s

    return merged_df


# function that translates coordinate system
def translateCoordsys(dataframe: pd.DataFrame) -> pd.DataFrame:
    # parameters necessary to translate moments (aka determine position of cg)
    x_hinge = (
        441.5  # x distance between force balance coord. sys. and hinge point in mm
    )
    z_hinge = 1359  # z distance between force balance coord. sys. and hinge point in mm
    l_cg = 625.4  # distance between hinge point and kite CG
    alpha_cg = np.deg2rad(
        dataframe["aoa"] - 23.82
    )  # 23.82 deg is the angle between the line that goes from the hinge to
    #                                                   # the kites CG and the rods protruding from the kite

    x_cg = l_cg * np.cos(alpha_cg)
    z_cg = l_cg * np.sin(alpha_cg)
    c_ref = 0.4

    x_hcg = (x_hinge + x_cg) / (1000 * c_ref)
    z_hcg = (z_hinge + z_cg) / (1000 * c_ref)

    # rotation of coordinate system: force coefficients change
    dataframe["C_L"] = dataframe["F_Z"] * -1
    dataframe["C_S"] = dataframe["F_Y"] * -1
    dataframe["C_D"] = dataframe["F_X"]

    dataframe["C_roll"] = dataframe["M_X"] - dataframe["F_Y"] * z_hcg
    dataframe["C_pitch"] = (
        -dataframe["M_Y"] + dataframe["F_Z"] * x_hcg - dataframe["F_X"] * z_hcg
    )
    dataframe["C_yaw"] = -dataframe["M_Z"] - dataframe["F_Y"] * x_hcg

    # save resulting dataframe
    df_result = dataframe
    return df_result


def add_dyn_visc_and_reynolds(df: pd.DataFrame) -> pd.DataFrame:
    # add dynamic viscosity and reynolds number column
    T = df["Temp"] + 273.15
    mu_0 = 1.716e-5
    T_0 = 273.15
    C_suth = 110.4
    c_ref = 0.4  # reference chord
    dynamic_viscosity = (
        mu_0 * (T / T_0) ** 0.5 * (T_0 + C_suth) / (T + C_suth)
    )  # sutherland's law
    df["dyn_vis"] = dynamic_viscosity
    df["Rey"] = df["Density"] * df["vw_actual"] * c_ref / df["dyn_vis"]
    return df


def processing_zigzag_lvm_data_into_csv(
    labbook_zz_path: Path,
    zigzag_data_dir: Path,
    support_struc_aero_interp_coeffs_path: Path,
    save_path_lvm_data_processed: Path,
) -> None:

    # Read processed_labbook.csv
    all_df = read_all_lvm_into_df(labbook_zz_path)

    all_df_processed = []
    for df in all_df:
        print(f"\n df.columns: {df.columns}")
        if df["vw"].unique()[0] > 1.1:

            # Substracting the self-induced side slip variation
            df["sideslip"] -= 1.8

            # 1. Substracting runs with zero wind
            merged_df_zero_substracted = substracting_runs_with_zero_wind(
                df, zigzag_data_dir
            )

            # 2. Nondimensionalizing
            merged_df_zero_substracted_nondim = nondimensionalize(
                merged_df_zero_substracted
            )

            # 3. Substracting support structure aerodynamic coefficients
            support_struc_aero_interp_coeffs = pd.read_csv(
                support_struc_aero_interp_coeffs_path
            )
            merged_df_zero_substracted_nondim_kite = (
                substract_support_structure_aero_coefficients(
                    merged_df_zero_substracted_nondim, support_struc_aero_interp_coeffs
                )
            )

            # 4. Translate coordinate system
            df_translated = translateCoordsys(merged_df_zero_substracted_nondim_kite)

            # 5. Add dynamic viscosity and Reynolds number
            df_final = add_dyn_visc_and_reynolds(df_translated)

            all_df_processed.append(df_final)

    # Concatenate all dataframes
    merged_df = pd.concat(all_df_processed, ignore_index=True)

    # Save the processed data to a csv file
    merged_df.to_csv(save_path_lvm_data_processed, index=False)


def plot_zigzag(
    csv_path: str, results_dir: str, figsize: tuple, fontsize: float
) -> None:

    # Read the processed data
    merged_df = pd.read_csv(csv_path)

    # Rounding the vw values
    merged_df["vw_actual_rounded"] = np.round(merged_df["vw_actual"], 0)
    merged_df["Rey_rounded"] = np.round(merged_df["Rey"], -4)

    print(f" sideslip_angles: {merged_df['sideslip'].unique()}")

    # Defining coefficients
    coefficients = ["C_L", "C_D", "C_pitch"]
    yaxis_names = ["CL", "CD", "CMx"]

    data_to_print = []
    labels_to_print = []
    rey_list = []
    for vw in merged_df["vw_actual_rounded"].unique():
        # finding the data for each vw
        df_vw = merged_df[merged_df["vw_actual_rounded"] == vw]
        # calculating reynolds number
        rey = round(df_vw["Rey"].mean() * 1e-5, 1)
        if rey == 2.9:
            rey = 2.8
        print(f"\nvw: {vw}, rey: {rey}")
        # seperating into zz and no zz
        data_zz = df_vw[df_vw["Filename"].str.startswith("ZZ")]
        data_no_zz = df_vw[df_vw["Filename"].str.startswith("ZZ")]

        # if vw == 15, extracting data for each sideslip
        if vw == 15:
            data_zz_sideslip_min10 = data_zz[data_zz["sideslip"] == -10]
            data_zz_sideslip_0 = data_zz[data_zz["sideslip"] == 0]
            data_zz_sideslip_10 = data_zz[data_zz["sideslip"] == 10]

            data_no_zz_sideslip_min10 = data_no_zz[data_no_zz["sideslip"] == -10]
            data_no_zz_sideslip_0 = data_no_zz[data_no_zz["sideslip"] == 0]
            data_no_zz_sideslip_10 = data_no_zz[data_no_zz["sideslip"] == 10]
            label_list = [
                rf"{rey:.1f} with zigzag ($\beta = -10^\circ$)",
                rf"{rey:.1f} without zigzag ($\beta = -10^\circ$)",
                rf"{rey:.1f} with zigzag ($\beta = 0^\circ$)",
                rf"{rey:.1f} without zigzag ($\beta = 0^\circ$)",
                rf"{rey:.1f} with zigzag ($\beta = 10^\circ$)",
                rf"{rey:.1f} without zigzag ($\beta = 10^\circ$)",
            ]
            data_list = [
                data_zz_sideslip_min10,
                data_no_zz_sideslip_min10,
                data_zz_sideslip_0,
                data_no_zz_sideslip_0,
                data_zz_sideslip_10,
                data_no_zz_sideslip_10,
            ]
        else:
            data_zz_sideslip = data_zz[data_zz["sideslip"] == 0]
            data_no_zz_sideslip = data_no_zz[data_no_zz["sideslip"] == 0]

            label_list = [
                rf"{rey:.1f} with zigzag",
                rf"{rey:.1f} without zigzag",
            ]
            data_list = [
                data_zz_sideslip,
                data_no_zz_sideslip,
            ]

        for data, label in zip(data_list, label_list):
            coefficients = ["C_L", "C_D", "C_pitch"]
            data_calculated = [
                [data[coeff].mean(), data[coeff].std()] for coeff in coefficients
            ]
            rey_list.append(rey)
            data_to_print.append(data_calculated)
            labels_to_print.append(label)

    create_grouped_plot(
        rey_list,
        data_to_print,
        labels_to_print,
        y_axis_labels,
        yaxis_names,
        results_dir,
    )


def create_grouped_plot(
    rey_list, data_to_print, labels_to_print, y_axis_labels, yaxis_names, results_dir
):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs = axes.flatten()

    # Group data by Reynolds number
    reynolds_groups = {1.4: [], 2.8: [], 4.2: [], 5.6: []}

    # Add shaded regions first so they appear behind the data points
    shaded_regions = [0, 2]  # Indices
    for ax in axs:
        for region_idx in shaded_regions:
            ax.axvspan(region_idx - 0.5, region_idx + 0.5, color="gray", alpha=0.15)

    for rey, data, label in zip(rey_list, data_to_print, labels_to_print):
        # Determine which Reynolds number group this belongs to
        for key in reynolds_groups.keys():
            if abs(rey - key) < 0.1:  # Use approximate matching
                reynolds_groups[key].append((data, label))
                break

    # Set up x-axis
    group_names = list(reynolds_groups.keys())
    n_groups = len(group_names)

    for ax_idx, ax in enumerate(axs):
        for group_idx, (rey_num, group_data) in enumerate(reynolds_groups.items()):
            if len(group_data) == 2:
                x_positions = np.linspace(
                    group_idx - 0.05, group_idx + 0.05, len(group_data)
                )
                color_list = ["red", "blue"]
                marker_list = ["o", "o"]
            elif len(group_data) == 6:
                x_positions = np.linspace(
                    group_idx - 0.25, group_idx + 0.25, len(group_data)
                )
                color_list = [
                    "red",
                    "blue",
                    "red",
                    "blue",
                    "red",
                    "blue",
                ]
                marker_list = ["o", "o", "x", "x", "*", "*"]

            for x_pos, (data, label), color, marker in zip(
                x_positions, group_data, color_list, marker_list
            ):
                ax.errorbar(
                    x_pos,
                    data[ax_idx][0],
                    yerr=data[ax_idx][1],
                    fmt=marker,
                    color=color,
                    capsize=5,
                )

        # Set x-axis
        # ax.set_xlim(0, 6)
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(group_names)
        ax.set_xlabel(r"Re $\times 10^5$ [-]")
        ax.set_ylabel(y_axis_labels[yaxis_names[ax_idx]])
        # Set only horizontal gridlines
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")

    # Add simplified legend to the first plot only
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            label="With zigzag",
            markersize=8,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="blue",
            label="Without zigzag",
            markersize=8,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color="red",
            label=r"With zigzag $\beta = -10^\circ$",
            markersize=8,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color="blue",
            label=r"Without zigzag $\beta = -10^\circ$",
            markersize=8,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="red",
            label=r"With zigzag $\beta = 10^\circ$",
            markersize=8,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="blue",
            label=r"Without zigzag $\beta = 10^\circ$",
            markersize=8,
            linestyle="None",
        ),
    ]
    axs[2].legend(handles=legend_elements, loc="lower right")
    axs[2].set_ylim(-5, 3)

    plt.tight_layout()

    # Save the figure
    saving_pdf_and_pdf_tex(results_dir, "zz_re_sweep_alpha_675_beta_0")


def main_process():
    ## Process zigzag data
    processed_data_zigzag_dir = Path(project_dir) / "processed_data" / "zigzag"
    if not os.path.exists(processed_data_zigzag_dir):
        os.makedirs(processed_data_zigzag_dir)

    zigzag_data_dir = Path(project_dir) / "data" / "zigzag"
    labbook_zz_path = Path(processed_data_zigzag_dir) / "labbook_zz.csv"
    support_struc_aero_interp_coeffs_path = (
        Path(project_dir) / "processed_data" / "support_struc_aero_interp_coeffs.csv"
    )
    save_path_lvm_data_processed = (
        Path(processed_data_zigzag_dir) / "lvm_data_processed.csv"
    )
    processing_zigzag_lvm_data_into_csv(
        labbook_zz_path,
        zigzag_data_dir,
        support_struc_aero_interp_coeffs_path,
        save_path_lvm_data_processed,
    )


def main(results_dir: str, project_dir: str) -> None:

    # Increase font size for readability
    fontsize = 18
    figsize = (20, 6)

    # Plot zigzag
    save_path_lvm_data_processed = (
        Path(project_dir) / "processed_data" / "zigzag" / "lvm_data_processed.csv"
    )
    plot_zigzag(save_path_lvm_data_processed, results_dir, figsize, fontsize)


if __name__ == "__main__":

    results_dir = Path(project_dir) / "results"
    main(results_dir, project_dir)
