import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from settings import saving_pdf_and_pdf_tex, x_axis_labels, y_axis_labels, root_dir
import re
from collections import defaultdict


def read_lvm(filename: str) -> pd.DataFrame:
    # Read the entire data file
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

    zigzag_data_dir = Path(root_dir) / "data" / "zigzag"
    all_data = []
    # Loop through each file in the zigzag directory
    for file in os.listdir(zigzag_data_dir):
        print(f"Processing file: {file}")

        # Read the lvm file
        lvm_file_path = Path(zigzag_data_dir) / file
        df = read_lvm(lvm_file_path)

        # Strip lvm from file
        filename = lvm_file_path.stem
        # print(f"Filename: {filename}")

        # Filter labbook for where folder matches the filename
        matching_rows = labbook_df[labbook_df["Filename"] == filename]

        # Get the last matching row if there are any matches
        if not matching_rows.empty:
            last_matching_row = matching_rows.iloc[-1]  # Get the last occurrence
            # print(f"Last matching row: {last_matching_row}")

            # Assuming you want to extract specific properties
            # For example, assuming properties are in columns like "Property1", "Property2", etc.
            properties_to_add = last_matching_row.to_dict()

            # Add properties from labbook to each row in df
            for col, value in properties_to_add.items():
                # print(f"col: {col}, value: {value}")
                df[col] = (
                    value  # This will create new columns in df with properties from labbook_df
                )

            # print(f"vw from df: {df['vw'].unique()[0]}")
            all_data.append(df)

        else:
            print(f"No matching rows found for {filename}")

        # Append the modified DataFrame to the all_data list

    return all_data


def substracting_runs_with_zero_wind(merged_df: pd.DataFrame) -> pd.DataFrame:
    # Renaming columns
    merged_df.rename(columns={"vw": "vw_actual"}, inplace=True)

    # Measurements of runs without wind
    output_normal = np.array(
        [3.616845, 4.836675, 802.681168, 4.386467, 171.123110, -0.111237]
    )
    output_zigzag = np.array(
        [3.545030, 4.522181, 802.495461, 4.192615, 171.237310, -0.041342]
    )
    output_normal_vw5 = np.array(
        [3.412569, 5.041744, 800.806084, 4.026951, 172.121360, 0.034760]
    )

    cols = ["F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]

    if merged_df["Filename"].str.startswith("ZZ").any():
        merged_df.loc[merged_df["Filename"].str.startswith("ZZ"), cols] -= output_zigzag

    elif merged_df["Filename"].str.startswith("normal").any():
        selected_rows = merged_df[
            (merged_df["Filename"].str.startswith("normal"))
            & (merged_df["vw_actual"] != 5)
        ]
        selected_rows_indices = selected_rows.index

        # Subtract output_normal from the selected rows
        merged_df.loc[selected_rows_indices, cols] -= output_normal

        # Ensure the output_normal is correctly aligned with the selected rows
        selected_rows = merged_df[
            (merged_df["Filename"].str.startswith("normal"))
            & (merged_df["vw_actual"] == 5)
        ]
        selected_rows_indices = selected_rows.index

        # Subtract output_normal from the selected rows
        merged_df.loc[selected_rows_indices, cols] -= output_normal_vw5

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
    merged_df: pd.DataFrame, interp_coeffs: pd.DataFrame
) -> pd.DataFrame:

    # print(f"columns: {merged_df.columns}")
    # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
    cur_vw = merged_df["vw_actual"].unique()[0]
    # print(f"cur_vw: {np.around(cur_vw)}")
    # print(f'interp_coeffs["vw"]: {interp_coeffs["vw"].unique()}')
    supp_coeffs = interp_coeffs[interp_coeffs["vw"] == int(np.around(cur_vw))]
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
        dataframe["aoa"][0] - 23.82
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


def processing_zigzag_lvm_data_into_csv(processed_data_zigzag_dir: str) -> None:

    # Read processed_labbook.csv
    all_df = read_all_lvm_into_df(Path(processed_data_zigzag_dir) / "labbook_zz.csv")

    all_df_processed = []
    for df in all_df:
        if df["vw"].unique()[0] > 1.1:

            # Substracting the self-induced side slip variation
            df["sideslip"] -= 1.8

            # 1. Substracting runs with zero wind
            merged_df_zero_substracted = substracting_runs_with_zero_wind(df)

            # 2. Nondimensionalizing
            merged_df_zero_substracted_nondim = nondimensionalize(
                merged_df_zero_substracted
            )

            # 3. Substracting support structure aerodynamic coefficients
            interp_coeffs = pd.read_csv(
                Path(processed_data_zigzag_dir) / "interp_coeff.csv"
            )
            merged_df_zero_substracted_nondim_kite = (
                substract_support_structure_aero_coefficients(
                    merged_df_zero_substracted_nondim, interp_coeffs
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
    merged_df.to_csv(
        Path(processed_data_zigzag_dir) / "lvm_data_processed.csv", index=False
    )


def plot_zigzag(
    csv_path: str, results_dir: str, figsize: tuple, fontsize: float
) -> None:

    # Read the processed data
    merged_df = pd.read_csv(csv_path)

    # Rounding the vw values
    merged_df["vw_actual_rounded"] = np.round(merged_df["vw_actual"], 0)
    merged_df["Rey_rounded"] = np.round(merged_df["Rey"], -4)

    # Printing the Rey
    print(f"Rey: {merged_df['Rey_rounded'].unique()}")
    print(f"vw: {merged_df['vw_actual_rounded'].unique()}")

    # Defining coefficients
    coefficients = ["C_L", "C_D", "C_pitch"]
    plot_titles = [
        "Lift coefficient",
        "Drag coefficient",
        "Pitching moment coefficient",
    ]
    yaxis_names = ["CL", "CD", "CMx"]

    filename_column = "Filename"
    sideslip_column = "sideslip"
    wind_speed_column = "Rey_rounded"

    # Filter data for zigzag tape and without zigzag tape
    data_zz = merged_df[merged_df[filename_column].str.startswith("ZZnor")]
    data_no_zz = merged_df[merged_df[filename_column].str.startswith("nor")]

    # Get unique sideslip angles
    sideslip_angles = np.unique(np.abs(merged_df[sideslip_column]))

    # Separate data
    sideslip = 0
    data_zz_sideslip = data_zz[data_zz[sideslip_column] == sideslip]
    data_no_zz_sideslip = data_no_zz[data_no_zz[sideslip_column] == sideslip]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    axs = axes.flatten()

    for idx, coeff in enumerate(coefficients):
        ax = axs[idx]
        label_added = {
            "zz": False,
            "no_zz": False,
        }  # Ensure each label is added only once

        for vw in merged_df["vw_actual_rounded"].unique():
            print(f"vw: {vw}")
            # Filter data for current vw
            data_zz_sideslip_vw = data_zz_sideslip[
                data_zz_sideslip["vw_actual_rounded"] == vw
            ]
            data_no_zz_sideslip_vw = data_no_zz_sideslip[
                data_no_zz_sideslip["vw_actual_rounded"] == vw
            ]

            # define rey
            rey = data_zz_sideslip_vw["Rey"].mean() * 1e-5
            if np.isnan(rey):
                rey = 4.2
            print(f"rey: {rey}")

            # Skip if no data for this vw
            if data_zz_sideslip_vw.empty and data_no_zz_sideslip_vw.empty:
                print(f"--> No data for vw: {vw}")
                continue  # No data for this vw, skip plotting

            # Calculate means and standard deviations for 'Without zigzag tape'
            if not data_no_zz_sideslip_vw.empty:
                means_no_zz = data_no_zz_sideslip_vw.groupby(wind_speed_column)[
                    coeff
                ].mean()
                std_no_zz = data_no_zz_sideslip_vw.groupby(wind_speed_column)[
                    coeff
                ].std()
                wind_speed_no_zz = means_no_zz.index * 1e-5

                # Plotting error bars for 'Without zigzag tape'
                ax.errorbar(
                    rey,
                    means_no_zz,
                    yerr=std_no_zz,
                    fmt="o",
                    color="red",
                    capsize=5,
                    label=(f"Without zigzag tape" if not label_added["no_zz"] else ""),
                )
                ax.plot(
                    rey,
                    means_no_zz,
                    marker="o",
                    linestyle="--",
                    color="red",
                )
                label_added["no_zz"] = True

            # Calculate means and standard deviations for 'With zigzag tape'
            if not data_zz_sideslip_vw.empty:
                means_zz = data_zz_sideslip_vw.groupby(wind_speed_column)[coeff].mean()
                std_zz = data_zz_sideslip_vw.groupby(wind_speed_column)[coeff].std()
                wind_speed_zz = means_zz.index * 1e-5

                # Plotting error bars for 'With zigzag tape'
                ax.errorbar(
                    rey,
                    means_zz,
                    yerr=std_zz,
                    fmt="x",
                    color="blue",
                    capsize=5,
                    label=(f"With zigzag tape" if not label_added["zz"] else ""),
                )
                ax.plot(
                    rey,
                    means_zz,
                    marker="x",
                    linestyle="--",
                    color="blue",
                )
                label_added["zz"] = True

        ax.set_xlim([1, 6])
        if idx == 0:
            ax.legend(loc="upper right")
        ax.set_xlabel(x_axis_labels["Re"])
        ax.set_ylabel(y_axis_labels[yaxis_names[idx]])
        ax.grid(True)

    plt.tight_layout()
    saving_pdf_and_pdf_tex(results_dir, "zz_re_sweep_alpha_675_beta_0")


# def transform_lvm_data(
#     lvm_data: pd.DataFrame,
#     metadata: pd.DataFrame,
#     interp_coeffs: pd.DataFrame,
#     is_zigzag: bool,
# ) -> pd.DataFrame:
#     """
#     Transform raw LVM data using the same process as the main data.

#     Args:
#     - lvm_data: DataFrame containing raw LVM data
#     - metadata: DataFrame containing metadata (from labbook_zz.csv)
#     - interp_coeffs: DataFrame containing interpolation coefficients
#     - is_zigzag: Boolean indicating if this is zigzag tape data

#     Returns:
#     - Transformed DataFrame
#     """
#     # Create a copy to avoid modifying original data
#     df = lvm_data.copy()

#     # 1. Apply offsets
#     output_normal = np.array(
#         [3.616845, 4.836675, 802.681168, 4.386467, 171.123110, -0.111237]
#     )
#     output_zigzag = np.array(
#         [3.545030, 4.522181, 802.495461, 4.192615, 171.237310, -0.041342]
#     )
#     output_normal_vw5 = np.array(
#         [3.412569, 5.041744, 800.806084, 4.026951, 172.121360, 0.034760]
#     )

#     cols = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

#     if is_zigzag:
#         df[cols] -= output_zigzag
#     else:
#         vw = metadata["vw"].iloc[0]  # Assuming consistent wind speed per file
#         if vw == 5:
#             df[cols] -= output_normal_vw5
#         else:
#             df[cols] -= output_normal

#     # 2. Nondimensionalize
#     rho = metadata["Density"].iloc[0]
#     V = metadata["vw"].iloc[0]
#     S_ref = 0.46
#     c_ref = 0.4

#     for col in cols[:3]:  # Force columns
#         df[col] /= 0.5 * rho * V**2 * S_ref
#     for col in cols[3:]:  # Moment columns
#         df[col] /= 0.5 * rho * V**2 * S_ref * c_ref

#     # 3. Adjust sideslip
#     sideslip = metadata["sideslip"].iloc[0] - 1.8

#     # 4. Process support structure coefficients
#     aoa = metadata["aoa"].iloc[0] - 7.25

#     # Get support coefficients for this wind speed and sideslip
#     supp_coeffs = interp_coeffs[
#         (interp_coeffs["vw"] == V) & (interp_coeffs["sideslip"] == sideslip)
#     ]

#     # Calculate and subtract support structure coefficients
#     coeff_types = ["Cx", "Cy", "Cz", "Cmx", "Cmy", "Cmz"]
#     for col, coeff_type in zip(cols, coeff_types):
#         coeff_data = supp_coeffs.loc[
#             supp_coeffs["channel"] == coeff_type, ["a", "b", "c"]
#         ]
#         if not coeff_data.empty:
#             correction = (
#                 coeff_data["a"].iloc[0] * (aoa**2)
#                 + coeff_data["b"].iloc[0] * aoa
#                 + coeff_data["c"].iloc[0]
#             )
#             df[col] -= correction

#     # 5. Translate coordinate system
#     x_hinge = 441.5
#     z_hinge = 1359
#     l_cg = 625.4
#     alpha_cg = np.deg2rad(metadata["aoa"].iloc[0] - 23.82)
#     x_cg = l_cg * np.cos(alpha_cg)
#     z_cg = l_cg * np.sin(alpha_cg)

#     x_kite = (x_hinge + x_cg) / (1000 * c_ref)
#     z_kite = (z_hinge + z_cg) / (1000 * c_ref)

#     # Calculate final coefficients
#     df["C_L"] = df["Fz"] * -1
#     df["C_D"] = df["Fx"]
#     df["C_pitch"] = -df["My"] + df["Fz"] * x_kite - df["Fx"] * z_kite

#     # 6. Calculate Reynolds number
#     T = metadata["Temp"].iloc[0] + 273.15
#     mu_0 = 1.716e-5
#     T_0 = 273.15
#     C_suth = 110.4
#     dynamic_viscosity = mu_0 * (T / T_0) ** 0.5 * (T_0 + C_suth) / (T + C_suth)
#     df["Rey"] = rho * V * c_ref / dynamic_viscosity

#     return df


# def calculate_stats_from_lvm(
#     root_dir: str, labbook_data: pd.DataFrame, interp_coeffs: pd.DataFrame
# ) -> dict:
#     """
#     Calculate statistics from raw LVM files after proper transformation.

#     Returns:
#     - Dictionary with statistics for each condition and Reynolds number
#     """
#     stats = {
#         "ZZnor": defaultdict(lambda: defaultdict(list)),
#         "nor": defaultdict(lambda: defaultdict(list)),
#     }

#     # Group labbook data by filename to get metadata for each file
#     file_metadata = labbook_data.groupby("Filename").first().reset_index()

#     for _, row in file_metadata.iterrows():
#         filename = row["Filename"]
#         is_zigzag = filename.startswith("ZZ")
#         prefix = "ZZnor" if is_zigzag else "nor"

#         # Read raw LVM file
#         lvm_path = Path(root_dir) / "raw_data" / "zigzag" / f"{filename}.lvm"
#         if lvm_path.exists():
#             # Read LVM file - adjust skiprows and names as needed
#             lvm_data = pd.read_csv(
#                 lvm_path,
#                 delimiter="\t",
#                 skiprows=23,  # Adjust based on your file structure
#                 names=["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "error"],
#             )

#             # Transform the data
#             transformed_data = transform_lvm_data(
#                 lvm_data,
#                 row.to_frame().T,  # Convert row to DataFrame
#                 interp_coeffs,
#                 is_zigzag,
#             )

#             # Store statistics for each coefficient
#             rey = transformed_data["Rey"].mean()  # Use mean Rey number for the file
#             for coeff in ["C_L", "C_D", "C_pitch"]:
#                 stats[prefix][rey][coeff].append(transformed_data[coeff].mean())

#     return stats


# def plot_zigzag(
#     root_dir: str, results_dir: str, fontsize: float, figsize: tuple
# ) -> None:

#     path_interp_coeffs = (
#         Path(root_dir) / "processed_data" / "zigzag" / "interp_coeff.csv"
#     )
#     interp_coeffs = pd.read_csv(path_interp_coeffs)

#     path_labbook_zz = Path(root_dir) / "processed_data" / "zigzag" / "labbook_zz.csv"
#     data = pd.read_csv(path_labbook_zz, delimiter=";")
#     data["SampleIndex"] = data.groupby("Filename").cumcount()

#     # Step 2: Get unique filenames from the 'Filename' column
#     unique_filenames = data["Filename"].unique()

#     # Step 4: Define the column names for the data files
#     column_names = ["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "error"]

#     # Step 5: Initialize an empty list to hold DataFrames
#     dataframes = []

#     # Step 6: Loop through each filename, read the content with column names, add a sample index and filename, and append to the list
#     for filename in unique_filenames:
#         # Adjust the delimiter as needed, e.g., '\t' for tab-separated values or ' ' for space-separated values
#         path_filename = Path(root_dir) / "processed_data" / "zigzag" / f"{filename}.txt"
#         df = pd.read_csv(
#             path_filename, delimiter="\t", names=column_names
#         )  # Assuming tab-separated values for this example
#         df["SampleIndex"] = range(len(df))  # Add a SampleIndex column
#         df["Filename"] = filename  # Add the filename as a column
#         dataframes.append(df)

#     # Step 7: Concatenate all DataFrames into a single DataFrame
#     big_dataframe = pd.concat(dataframes, ignore_index=True)

#     merged_df = pd.merge(data, big_dataframe, on=["Filename", "SampleIndex"])
#     merged_df.rename(columns={"vw": "vw_actual"}, inplace=True)
#     merged_df["vw"] = np.around(merged_df["vw_actual"], 0)
#     merged_df = merged_df[merged_df["vw"] != 0]

#     output_normal = np.array(
#         [3.616845, 4.836675, 802.681168, 4.386467, 171.123110, -0.111237]
#     )
#     output_zigzag = np.array(
#         [3.545030, 4.522181, 802.495461, 4.192615, 171.237310, -0.041342]
#     )
#     output_normal_vw5 = np.array(
#         [3.412569, 5.041744, 800.806084, 4.026951, 172.121360, 0.034760]
#     )

#     cols = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

#     merged_df.loc[merged_df["Filename"].str.startswith("ZZ"), cols] -= output_zigzag
#     # merged_df[(merged_df['Filename'].str.startswith('normal')) & (merged_df['vw'] != 5)] -= output_normal

#     # Ensure the output_normal is correctly aligned with the selected rows
#     selected_rows = merged_df[
#         (merged_df["Filename"].str.startswith("normal")) & (merged_df["vw"] != 5)
#     ]
#     selected_rows_indices = selected_rows.index

#     # Subtract output_normal from the selected rows
#     merged_df.loc[selected_rows_indices, cols] -= output_normal

#     # Ensure the output_normal is correctly aligned with the selected rows
#     selected_rows = merged_df[
#         (merged_df["Filename"].str.startswith("normal")) & (merged_df["vw"] == 5)
#     ]
#     selected_rows_indices = selected_rows.index

#     # Subtract output_normal from the selected rows
#     merged_df.loc[selected_rows_indices, cols] -= output_normal_vw5
#     # merged_df[(merged_df['Filename'].str.startswith('normal')) & (merged_df['vw'] == 5)] -= output_normal_vw5

#     # Step 3: Nondimensionalize the columns "F_X", "F_Y", "F_Z", "M_X", "M_Y", and "M_Z"
#     rho = merged_df["Density"]
#     V = merged_df["vw_actual"]
#     S_ref = 0.46
#     c_ref = 0.4

#     # Nondimensionalize the force columns
#     merged_df["Fx"] /= 0.5 * rho * V**2 * S_ref
#     merged_df["Fy"] /= 0.5 * rho * V**2 * S_ref
#     merged_df["Fz"] /= 0.5 * rho * V**2 * S_ref

#     # Nondimensionalize the moment columns
#     merged_df["Mx"] /= 0.5 * rho * V**2 * S_ref * c_ref
#     merged_df["My"] /= 0.5 * rho * V**2 * S_ref * c_ref
#     merged_df["Mz"] /= 0.5 * rho * V**2 * S_ref * c_ref

#     # subtract the correction from the sideslip
#     merged_df["sideslip"] -= 1.8

#     # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
#     aoa = merged_df["aoa"].unique()[0] - 7.25

#     for v in merged_df["vw"].unique():
#         supp_coeffs = interp_coeffs[interp_coeffs["vw"] == v]
#         for k in merged_df["sideslip"].unique():
#             # select support structure aero coefficients for this sideslip
#             c_s = supp_coeffs[supp_coeffs["sideslip"] == k]
#             F_x = c_s.loc[c_s["channel"] == "Cx", ["a", "b", "c"]]
#             F_y = c_s.loc[c_s["channel"] == "Cy", ["a", "b", "c"]]
#             F_z = c_s.loc[c_s["channel"] == "Cz", ["a", "b", "c"]]
#             M_x = c_s.loc[c_s["channel"] == "Cmx", ["a", "b", "c"]]
#             M_y = c_s.loc[c_s["channel"] == "Cmy", ["a", "b", "c"]]
#             M_z = c_s.loc[c_s["channel"] == "Cmz", ["a", "b", "c"]]

#             # compute support structure aero coefficients for this wind speed, sideslip and angle of attack combination
#             C_Fx_s = np.array(F_x["a"] * (aoa**2) + F_x["b"] * aoa + F_x["c"])[0]
#             C_Fy_s = np.array(F_y["a"] * (aoa**2) + F_y["b"] * aoa + F_y["c"])[0]
#             C_Fz_s = np.array(F_z["a"] * (aoa**2) + F_z["b"] * aoa + F_z["c"])[0]
#             C_Mx_s = np.array(M_x["a"] * (aoa**2) + M_x["b"] * aoa + M_x["c"])[0]
#             C_My_s = np.array(M_y["a"] * (aoa**2) + M_y["b"] * aoa + M_y["c"])[0]
#             C_Mz_s = np.array(M_z["a"] * (aoa**2) + M_z["b"] * aoa + M_z["c"])[0]

#             # subtract support structure aero coefficients for this wind speed, sideslip and aoa combination from merged_df
#             merged_df.loc[
#                 (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Fx"
#             ] -= C_Fx_s
#             merged_df.loc[
#                 (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Fy"
#             ] -= C_Fy_s
#             merged_df.loc[
#                 (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Fz"
#             ] -= C_Fz_s
#             merged_df.loc[
#                 (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Mx"
#             ] -= C_Mx_s
#             merged_df.loc[
#                 (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "My"
#             ] -= C_My_s
#             merged_df.loc[
#                 (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Mz"
#             ] -= C_Mz_s

#     # translate coordinate system
#     # parameters necessary to translate moments (aka determine position of cg)
#     x_hinge = 441.5
#     z_hinge = 1359
#     l_cg = 625.4
#     alpha_cg = np.deg2rad(merged_df["aoa"] - 23.82)
#     x_cg = l_cg * np.cos(alpha_cg)
#     z_cg = l_cg * np.sin(alpha_cg)
#     A_side = 5.646
#     A_ref = 19.753
#     Cs_scale = A_ref / A_side

#     merged_df["x_kite"] = (x_hinge + x_cg) / (1000 * c_ref)
#     merged_df["z_kite"] = (z_hinge + z_cg) / (1000 * c_ref)

#     # rotation of coordinate system: force coefficients change
#     merged_df["C_L"] = merged_df["Fz"] * -1
#     merged_df["C_S"] = merged_df["Fy"] * -1
#     merged_df["C_D"] = merged_df["Fx"]

#     merged_df["C_roll"] = merged_df["Mx"] - merged_df["Fy"] * merged_df["z_kite"]
#     merged_df["C_pitch"] = (
#         -merged_df["My"]
#         + merged_df["Fz"] * merged_df["x_kite"]
#         - merged_df["Fx"] * merged_df["z_kite"]
#     )
#     merged_df["C_yaw"] = -merged_df["Mz"] - merged_df["Fy"] * merged_df["x_kite"]

#     # # Constants for Sutherland's law
#     # C1 = 1.458e-6  # Sutherland's constant in kg/(m·s·sqrt(K))
#     # S = 110.4  # Sutherland's temperature in K

#     # # Reference length
#     # c_ref = 0.4

#     # # Calculate dynamic viscosity using Sutherland's law
#     # merged_df['dynamic_viscosity'] = C1 * merged_df['Temp'] ** 1.5 / (merged_df['Temp'] + S)

#     # # Calculate Reynolds number
#     # merged_df['Re'] = (merged_df['Density'] * merged_df['vw_actual'] * c_ref) / merged_df['dynamic_viscosity'] #/ 1e5

#     # add dynamic viscosity and reynolds number column
#     T = merged_df["Temp"] + 273.15
#     mu_0 = 1.716e-5
#     T_0 = 273.15
#     C_suth = 110.4
#     c_ref = 0.4  # reference chord
#     dynamic_viscosity = (
#         mu_0 * (T / T_0) ** 0.5 * (T_0 + C_suth) / (T + C_suth)
#     )  # sutherland's law
#     merged_df["dyn_vis"] = dynamic_viscosity
#     merged_df["Rey"] = (
#         merged_df["Density"] * merged_df["vw_actual"] * c_ref / merged_df["dyn_vis"]
#     )

#     # # Define the coefficients and other relevant columns
#     # coefficients = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
#     # plot_titles = [
#     #     "Drag coefficient",
#     #     "Side-force coefficient",
#     #     "Lift coefficient",
#     #     "Roll moment coefficient",
#     #     "Pitching moment coefficient",
#     #     "Yawing moment coefficient",
#     # ]
#     # yaxis_names = ["C_D", "C_S", "C_L", "C_{roll}", "C_{pitch}", "C_{yaw}"]

#     # defining coefficients
#     coefficients = ["C_L", "C_D", "C_pitch"]
#     plot_titles = [
#         "Lift coefficient",
#         "Drag coefficient",
#         "Pitching moment coefficient",
#     ]
#     yaxis_names = ["CL", "CD", "CMx"]

#     filename_column = "Filename"
#     sideslip_column = "sideslip"
#     wind_speed_column = "Rey"

#     # Filter data for zigzag tape and without zigzag tape
#     data_zz = merged_df[merged_df[filename_column].str.startswith("ZZnor")]
#     data_no_zz = merged_df[merged_df[filename_column].str.startswith("nor")]

#     # Get unique sideslip angles
#     sideslip_angles = np.unique(np.abs(merged_df[sideslip_column]))
#     # separate data
#     sideslip = 0
#     data_zz_sideslip = data_zz[data_zz[sideslip_column] == sideslip]
#     data_no_zz_sideslip = data_no_zz[data_no_zz[sideslip_column] == sideslip]

#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
#     # fig.suptitle(rf'Sideslip angle $\beta$= {np.int64(sideslip)}$^o$', fontsize=16)
#     axs = axes.flatten()
#     for idx, coeff in enumerate(coefficients):
#         ax = axs[idx]
#         ax.plot(
#             1e-5 * data_no_zz_sideslip[wind_speed_column],
#             data_no_zz_sideslip[coeff],
#             label="Without zigzag tape",
#             marker="o",
#             color="red",
#         )
#         ax.plot(
#             1e-5 * data_zz_sideslip[wind_speed_column],
#             data_zz_sideslip[coeff],
#             label="With zigzag tape",
#             marker="x",
#             color="blue",
#         )
#         ax.set_xlim([1, 6])
#         if idx == 0:
#             ax.legend()

#             ax.set_ylim([0.7, 0.9])
#         elif idx == 1:  # CD
#             ax.set_ylim([0.06, 0.16])
#         elif idx == 2:  # C_pitch
#             ax.set_ylim([0.1, 0.55])
#         ax.set_xlabel(x_axis_labels["Re"])  # , fontsize=fontsize)
#         ax.set_ylabel(y_axis_labels[yaxis_names[idx]])  # , fontsize=fontsize)
#         # ax.set_title(plot_titles[idx])
#         ax.grid(True)

#     # plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.tight_layout()
#     # Adjust layout to make room for the main title
#     # path_save = Path(results_dir) / f"zz_re_sweep_alpha_675_beta_0.pdf"
#     # plt.savefig(path_save)
#     # plt.show()
#     saving_pdf_and_pdf_tex(results_dir, "zz_re_sweep_alpha_675_beta_0")


# def plot_zigzag_with_errors(
#     root_dir: str, results_dir: str, fontsize: float, figsize: tuple
# ) -> None:
#     # Load and process data as in original function
#     path_interp_coeffs = (
#         Path(root_dir) / "processed_data" / "zigzag" / "interp_coeff.csv"
#     )
#     interp_coeffs = pd.read_csv(path_interp_coeffs)

#     path_labbook_zz = Path(root_dir) / "processed_data" / "zigzag" / "labbook_zz.csv"
#     data = pd.read_csv(path_labbook_zz, delimiter=";")
#     data["SampleIndex"] = data.groupby("Filename").cumcount()

#     # Get unique filenames and prepare for data processing
#     unique_filenames = data["Filename"].unique()
#     column_names = ["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "error"]
#     dataframes = []

#     # Process each file
#     for filename in unique_filenames:
#         path_filename = Path(root_dir) / "processed_data" / "zigzag" / f"{filename}.txt"
#         df = pd.read_csv(path_filename, delimiter="\t", names=column_names)
#         df["SampleIndex"] = range(len(df))
#         df["Filename"] = filename
#         dataframes.append(df)

#     # Combine all data and merge
#     big_dataframe = pd.concat(dataframes, ignore_index=True)
#     merged_df = pd.merge(data, big_dataframe, on=["Filename", "SampleIndex"])
#     merged_df.rename(columns={"vw": "vw_actual"}, inplace=True)
#     merged_df["vw"] = np.around(merged_df["vw_actual"], 0)
#     merged_df = merged_df[merged_df["vw"] != 0]

#     # Apply offsets
#     output_normal = np.array(
#         [3.616845, 4.836675, 802.681168, 4.386467, 171.123110, -0.111237]
#     )
#     output_zigzag = np.array(
#         [3.545030, 4.522181, 802.495461, 4.192615, 171.237310, -0.041342]
#     )
#     output_normal_vw5 = np.array(
#         [3.412569, 5.041744, 800.806084, 4.026951, 172.121360, 0.034760]
#     )

#     cols = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

#     # Apply corrections as in original function
#     merged_df.loc[merged_df["Filename"].str.startswith("ZZ"), cols] -= output_zigzag

#     selected_rows = merged_df[
#         (merged_df["Filename"].str.startswith("normal")) & (merged_df["vw"] != 5)
#     ]
#     merged_df.loc[selected_rows.index, cols] -= output_normal

#     selected_rows = merged_df[
#         (merged_df["Filename"].str.startswith("normal")) & (merged_df["vw"] == 5)
#     ]
#     merged_df.loc[selected_rows.index, cols] -= output_normal_vw5

#     # Nondimensionalize
#     rho = merged_df["Density"]
#     V = merged_df["vw_actual"]
#     S_ref = 0.46
#     c_ref = 0.4

#     for col in cols[:3]:  # Force columns
#         merged_df[col] /= 0.5 * rho * V**2 * S_ref
#     for col in cols[3:]:  # Moment columns
#         merged_df[col] /= 0.5 * rho * V**2 * S_ref * c_ref

#     # Adjust sideslip
#     merged_df["sideslip"] -= 1.8

#     # Process support structure coefficients
#     aoa = merged_df["aoa"].unique()[0] - 7.25

#     # Your existing support structure coefficient processing here
#     for v in merged_df["vw"].unique():
#         supp_coeffs = interp_coeffs[interp_coeffs["vw"] == v]
#         for k in merged_df["sideslip"].unique():
#             c_s = supp_coeffs[supp_coeffs["sideslip"] == k]

#             # Process each coefficient type
#             coeff_types = ["Cx", "Cy", "Cz", "Cmx", "Cmy", "Cmz"]
#             coeffs_calc = {}

#             for coeff_type in coeff_types:
#                 coeff_data = c_s.loc[c_s["channel"] == coeff_type, ["a", "b", "c"]]
#                 coeffs_calc[coeff_type] = np.array(
#                     coeff_data["a"] * (aoa**2) + coeff_data["b"] * aoa + coeff_data["c"]
#                 )[0]

#             # Subtract support structure coefficients
#             for orig_col, calc_coeff in zip(cols, coeffs_calc.values()):
#                 merged_df.loc[
#                     (merged_df["sideslip"] == k) & (merged_df["vw"] == v), orig_col
#                 ] -= calc_coeff

#     # Translate coordinate system
#     x_hinge = 441.5
#     z_hinge = 1359
#     l_cg = 625.4
#     alpha_cg = np.deg2rad(merged_df["aoa"] - 23.82)
#     x_cg = l_cg * np.cos(alpha_cg)
#     z_cg = l_cg * np.sin(alpha_cg)

#     merged_df["x_kite"] = (x_hinge + x_cg) / (1000 * c_ref)
#     merged_df["z_kite"] = (z_hinge + z_cg) / (1000 * c_ref)

#     # Calculate final coefficients
#     merged_df["C_L"] = merged_df["Fz"] * -1
#     merged_df["C_D"] = merged_df["Fx"]
#     merged_df["C_pitch"] = (
#         -merged_df["My"]
#         + merged_df["Fz"] * merged_df["x_kite"]
#         - merged_df["Fx"] * merged_df["z_kite"]
#     )

#     # Calculate Reynolds number
#     T = merged_df["Temp"] + 273.15
#     mu_0 = 1.716e-5
#     T_0 = 273.15
#     C_suth = 110.4
#     dynamic_viscosity = mu_0 * (T / T_0) ** 0.5 * (T_0 + C_suth) / (T + C_suth)
#     merged_df["dyn_vis"] = dynamic_viscosity
#     merged_df["Rey"] = (
#         merged_df["Density"] * merged_df["vw_actual"] * c_ref / merged_df["dyn_vis"]
#     )

#     # Plotting setup
#     coefficients = ["C_L", "C_D", "C_pitch"]
#     yaxis_names = ["CL", "CD", "CMx"]

#     # Filter and plot data
#     filename_column = "Filename"
#     sideslip_column = "sideslip"
#     wind_speed_column = "Rey"

#     data_zz = merged_df[merged_df[filename_column].str.startswith("ZZnor")]
#     data_no_zz = merged_df[merged_df[filename_column].str.startswith("nor")]

#     sideslip = 0
#     data_zz_sideslip = data_zz[data_zz[sideslip_column] == sideslip]
#     data_no_zz_sideslip = data_no_zz[data_no_zz[sideslip_column] == sideslip]

#     # Calculate statistics from raw LVM files
#     folder_path = Path(root_dir) / "data" / "zigzag"
#     suffix = "_unsteady.lvm"
#     labbook_data_lvm = read_and_process_files(folder_path, suffix)
#     stats = calculate_stats_from_lvm(root_dir, labbook_data_lvm, interp_coeffs)

#     # Plotting
#     coefficients = ["C_L", "C_D", "C_pitch"]
#     yaxis_names = ["CL", "CD", "CMx"]

#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
#     axs = axes.flatten()

#     for idx, coeff in enumerate(coefficients):
#         ax = axs[idx]

#         ax.plot(
#             1e-5 * data_no_zz_sideslip[wind_speed_column],
#             data_no_zz_sideslip[coeff],
#             label="Without zigzag tape",
#             marker="o",
#             color="red",
#         )
#         ax.plot(
#             1e-5 * data_zz_sideslip[wind_speed_column],
#             data_zz_sideslip[coeff],
#             label="With zigzag tape",
#             marker="x",
#             color="blue",
#         )
#         for prefix, label, marker, color in [
#             ("nor", "Without zigzag tape", "o", "red"),
#             ("ZZnor", "With zigzag tape", "x", "blue"),
#         ]:
#             rey_values = sorted(stats[prefix].keys())
#             means = [np.mean(stats[prefix][rey][coeff]) for rey in rey_values]
#             stds = [
#                 (
#                     np.std(stats[prefix][rey][coeff])
#                     if len(stats[prefix][rey][coeff]) > 1
#                     else 0
#                 )
#                 for rey in rey_values
#             ]

#             ax.errorbar(
#                 1e-5 * np.array(rey_values),
#                 means,
#                 yerr=stds,
#                 label=label,
#                 marker=marker,
#                 color=color,
#                 capsize=5,
#             )

#         ax.set_xlim([1, 6])
#         if idx == 0:
#             ax.legend()
#         #     ax.set_ylim([0.7, 0.9])
#         # elif idx == 1:
#         #     ax.set_ylim([0.06, 0.16])
#         # elif idx == 2:
#         #     ax.set_ylim([0.1, 0.55])
#         ax.set_xlabel("Re [-]")
#         ax.set_ylabel(f"{yaxis_names[idx]} [-]")
#         ax.grid(True)

#     plt.tight_layout()
#     saving_pdf_and_pdf_tex(results_dir, "zz_re_sweep_alpha_675_beta_0_with_errors")
#     return fig


def main(results_dir: str, root_dir: str) -> None:

    # Increase font size for readability
    # plt.rcParams.update({"font.size": 14})
    fontsize = 18
    figsize = (20, 6)

    # # Process zigzag data
    processed_data_zigzag_dir = Path(root_dir) / "processed_data" / "zigzag"
    # processing_zigzag_lvm_data_into_csv(processed_data_zigzag_dir)

    # Plot zigzag
    csv_path = Path(processed_data_zigzag_dir) / "lvm_data_processed.csv"
    plot_zigzag(csv_path, results_dir, figsize, fontsize)


if __name__ == "__main__":

    results_dir = Path(root_dir) / "results"
    main(results_dir, root_dir)
