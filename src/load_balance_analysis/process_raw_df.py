import pandas as pd
import numpy as np
from pathlib import Path
from utils import project_dir


# def substracting_runs_with_zero_wind(
#     merged_df: pd.DataFrame, processed_data_zigzag_dir: Path
# ) -> pd.DataFrame:
#     # Renaming columns
#     merged_df.rename(columns={"vw": "vw_actual"}, inplace=True)

#     # Defining the filename
#     filename = merged_df["Filename"].unique()[0]
#     print(f"filename working with: {filename}")

#     if merged_df["Filename"].str.startswith("normal").any():
#         filename_vw_0 = "normal_aoa_16_vw_00"
#     elif merged_df["Filename"].str.startswith("ZZ").any():
#         filename_vw_0 = "ZZnormal_aoa_16_vw_00"
#     print(f"filename_vw_0: {filename_vw_0}")

#     # Extracting the zero-run data, from txt
#     zero_run_path = Path(processed_data_zigzag_dir) / f"{filename_vw_0}.txt"
#     # with open(zero_run_path, "r") as file:
#     #     zero_run_data = file.read()
#     #     print(zero_run_data[2:-1])
#     data = []
#     with open(zero_run_path, "r") as file:
#         for line in file:
#             # Split each line into a list of values (assumes tab-separated values)
#             values = line.strip().split("\t")
#             # Convert strings to floats (or other data types if needed)
#             values = [float(value) for value in values]
#             # Append the list of values to the data list
#             data.append(values)

#     zero_run_data = data[0][1:-1]
#     cols = ["F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]
#     print(f" col of merged_df: {merged_df.head()}")
#     # Convert zero_run_data to a pandas Series to match the column names
#     zero_run_series = pd.Series(zero_run_data, index=cols)

#     # Subtract zero_run_series from the corresponding columns in merged_df
#     merged_df[cols] = merged_df[cols] - zero_run_series
#     return merged_df


def nondimensionalize(df: pd.DataFrame, S_ref: float, c_ref: float) -> pd.DataFrame:
    rho = df["Density"]
    V = df["vw"]

    # Check the types of rho, V, and S_ref
    print(f"rho: type: {type(rho.iloc[0])}")
    print(f"V type: {type(V.iloc[0])}")
    print(f"S_ref: {S_ref}, type: {type(S_ref)}")

    # Nondimensionalize the force columns
    # Check for non-numeric values in the "F_X" column
    df["F_X"] = pd.to_numeric(df["F_X"], errors="coerce")
    df = df.dropna(subset=["F_X"])
    # Check the data types of the 'F_X' column
    print("counting", df["F_X"].apply(type).value_counts())

    df["F_X"] /= 0.5 * rho * V**2 * S_ref
    df["F_Y"] /= 0.5 * rho * V**2 * S_ref
    df["F_Z"] /= 0.5 * rho * V**2 * S_ref

    # Nondimensionalize the moment columns
    df["M_X"] /= 0.5 * rho * V**2 * S_ref * c_ref
    df["M_Y"] /= 0.5 * rho * V**2 * S_ref * c_ref
    df["M_Z"] /= 0.5 * rho * V**2 * S_ref * c_ref

    return df


def substract_support_structure_aero_coefficients(
    merged_df: pd.DataFrame, interp_coeffs_path: Path
) -> pd.DataFrame:

    # reading the interpolated coefficients
    interp_coeffs = pd.read_csv(interp_coeffs_path)
    # print(f"columns: {merged_df.columns}")
    # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
    cur_vw = merged_df["vw"].unique()[0]
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
def translate_coordinate_system(
    dataframe: pd.DataFrame,
    x_hinge: float,
    z_hinge: float,
    l_cg: float,
    alpha_cg_delta: float,
    c_ref: float,
) -> pd.DataFrame:

    alpha_cg = np.deg2rad(dataframe["aoa"] - alpha_cg_delta)
    x_cg = l_cg * np.cos(alpha_cg)
    z_cg = l_cg * np.sin(alpha_cg)
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


def add_dyn_visc_and_reynolds(
    df: pd.DataFrame,
    delta_celsius_kelvin: float,
    mu_0: float,
    T_0: float,
    C_suth: float,
    c_ref: float,
) -> pd.DataFrame:
    # add dynamic viscosity and reynolds number column
    T = df["Temp"] + delta_celsius_kelvin

    dynamic_viscosity = (
        mu_0 * (T / T_0) ** 0.5 * (T_0 + C_suth) / (T + C_suth)
    )  # sutherland's law
    df["dyn_vis"] = dynamic_viscosity
    df["Rey"] = df["Density"] * df["vw"] * c_ref / df["dyn_vis"]
    return df


def processing_raw_lvm_data_into_csv(
    folder_dir: Path,
    S_ref: float,
    c_ref: float,
    interp_coeffs_path: Path,
    x_hinge: float,
    z_hinge: float,
    l_cg: float,
    alpha_cg_delta: float,
    delta_celsius_kelvin: float,
    mu_0: float,
    T_0: float,
    C_suth: float,
):

    # loop through all files in the folder
    all_df_processed = []

    for i, file in enumerate(folder_dir.iterdir()):
        df = pd.read_csv(file)
        file_name = file.name
        print(f"\n File: {file_name}")

        # Storing the raw values
        df["F_X_raw"] = df["F_X"]
        df["F_Y_raw"] = df["F_Y"]
        df["F_Z_raw"] = df["F_Z"]
        df["M_X_raw"] = df["M_X"]
        df["M_Y_raw"] = df["M_Y"]
        df["M_Z_raw"] = df["M_Z"]

        # Processing string values into floats
        columns_to_process = ["aoa", "vw", "Dpa", "Pressure", "Temp", "Density"]
        for column in columns_to_process:
            df[column] = df[column].astype(str).str.replace(",", ".").astype(float)

        # Compute Average Zero-Run values
        if i == 0 and df["vw"].unique()[0] < 1.1:
            print(f'i: {i}, vw: {df["vw"].unique()}')
            # Computing the average measured zero-run values
            zero_F_X = df["F_X"].mean()
            zero_F_Y = df["F_Y"].mean()
            zero_F_Z = df["F_Z"].mean()
            zero_M_X = df["M_X"].mean()
            zero_M_Y = df["M_Y"].mean()
            zero_M_Z = df["M_Z"].mean()

        else:

            # 1. Substracting the zero-run values
            df["F_X"] -= zero_F_X
            df["F_Y"] -= zero_F_Y
            df["F_Z"] -= zero_F_Z
            df["M_X"] -= zero_M_X
            df["M_Y"] -= zero_M_Y
            df["M_Z"] -= zero_M_Z

            # 2. Nondimensionalizing
            df = nondimensionalize(df, S_ref, c_ref)

            # 3. Substracting support structure aerodynamic coefficients
            df = substract_support_structure_aero_coefficients(df, interp_coeffs_path)

            # 4. Translate coordinate system
            df = translate_coordinate_system(
                df,
                x_hinge,
                z_hinge,
                l_cg,
                alpha_cg_delta,
                c_ref,
            )

            # 5. Correct for sideslip

            ## defining rotation matrix
            R_yaw = np.array(
                [
                    [np.cos(beta), -np.sin(beta), 0],
                    [np.sin(beta), np.cos(beta), 0],
                    [0, 0, 1],
                ]
            )
            ## calculating true forces and moments
            F_D_true = np.dot(R_yaw, F_D_meas)
            F_S_true = np.dot(R_yaw, F_S_meas)

            # Add dynamic viscosity and Reynolds number
            df = add_dyn_visc_and_reynolds(
                df, delta_celsius_kelvin, mu_0, T_0, C_suth, c_ref
            )

            # Save the processed data to a csv file
            file_name_without_raw = df["Filename"].unique()[0]
            df.to_csv(folder_dir / f"{file_name_without_raw}.csv", index=False)


if __name__ == "__main__":

    S_ref = 0.46
    c_ref = 0.4
    interp_coeffs_path = (
        Path(project_dir) / "processed_data" / "normal_csv" / "interp_coeff.csv"
    )
    # parameters necessary to translate moments (aka determine position of cg)
    x_hinge = (
        441.5  # x distance between force balance coord. sys. and hinge point in mm
    )
    z_hinge = 1359  # z distance between force balance coord. sys. and hinge point in mm
    l_cg = 625.4  # distance between hinge point and kite CG
    alpha_cg_delta = 23.82
    delta_celsius_kelvin = 273.15
    mu_0 = 1.716e-5
    T_0 = 273.15
    C_suth = 110.4

    folder_dir = Path(project_dir) / "processed_data" / "normal_csv" / "aoa_00"
    processing_raw_lvm_data_into_csv(
        folder_dir,
        S_ref,
        c_ref,
        interp_coeffs_path,
        x_hinge,
        z_hinge,
        l_cg,
        alpha_cg_delta,
        delta_celsius_kelvin,
        mu_0,
        T_0,
        C_suth,
    )
