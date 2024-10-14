import pandas as pd
import numpy as np
from pathlib import Path
import os
from utils import project_dir


def nondimensionalize(df: pd.DataFrame, S_ref: float, c_ref: float) -> pd.DataFrame:
    """
    Nondimensionalize the force and moment columns in the dataframe

    Input:
        df (pandas.DataFrame): The input dataframe
        S_ref (float): Reference area
        c_ref (float): Reference chord length

    Output:
        pandas.DataFrame: The dataframe with the force and moment columns nondimensionalized
    """
    rho = df["Density"]
    V = df["vw"]

    # Nondimensionalize the force columns
    df["CF_X"] = df["F_X"] / (0.5 * rho * V**2 * S_ref)
    df["CF_Y"] = df["F_Y"] / (0.5 * rho * V**2 * S_ref)
    df["CF_Z"] = df["F_Z"] / (0.5 * rho * V**2 * S_ref)

    # Nondimensionalize the moment columns
    df["CM_X"] = df["M_X"] / (0.5 * rho * V**2 * S_ref * c_ref)
    df["CM_Y"] = df["M_Y"] / (0.5 * rho * V**2 * S_ref * c_ref)
    df["CM_Z"] = df["M_Z"] / (0.5 * rho * V**2 * S_ref * c_ref)

    ## Also do this for the raw columns, to later calculate SNR
    df["CF_X_raw"] = df["F_X_raw"] / (0.5 * rho * V**2 * S_ref)
    df["CF_Y_raw"] = df["F_Y_raw"] / (0.5 * rho * V**2 * S_ref)
    df["CF_Z_raw"] = df["F_Z_raw"] / (0.5 * rho * V**2 * S_ref)
    df["CM_X_raw"] = df["M_X_raw"] / (0.5 * rho * V**2 * S_ref * c_ref)
    df["CM_Y_raw"] = df["M_Y_raw"] / (0.5 * rho * V**2 * S_ref * c_ref)
    df["CM_Z_raw"] = df["M_Z_raw"] / (0.5 * rho * V**2 * S_ref * c_ref)

    return df


def substract_support_structure_aero_coefficients(
    df: pd.DataFrame, support_struc_aero_interp_coeffs_path: Path
) -> pd.DataFrame:

    # reading the interpolated coefficients
    support_struc_aero_interp_coeffs = pd.read_csv(
        support_struc_aero_interp_coeffs_path
    )
    # print(f"columns: {merged_df.columns}")
    # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
    cur_vw = df["vw"].unique()[0]
    # print(f"cur_vw: {np.around(cur_vw)}")
    # print(f'support_struc_aero_interp_coeffs["vw"]: {support_struc_aero_interp_coeffs["vw"].unique()}')
    supp_coeffs = support_struc_aero_interp_coeffs[
        support_struc_aero_interp_coeffs["vw"] == int(np.around(cur_vw))
    ]
    # print(f"supp_coeffs: {supp_coeffs}")
    aoa_kite = df["aoa_kite"].unique()[0]

    for k in df["sideslip"].unique():
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
        C_Fx_s = np.array(F_x["a"] * (aoa_kite**2) + F_x["b"] * aoa_kite + F_x["c"])[0]
        C_Fy_s = np.array(F_y["a"] * (aoa_kite**2) + F_y["b"] * aoa_kite + F_y["c"])[0]
        C_Fz_s = np.array(F_z["a"] * (aoa_kite**2) + F_z["b"] * aoa_kite + F_z["c"])[0]
        C_Mx_s = np.array(M_x["a"] * (aoa_kite**2) + M_x["b"] * aoa_kite + M_x["c"])[0]
        C_My_s = np.array(M_y["a"] * (aoa_kite**2) + M_y["b"] * aoa_kite + M_y["c"])[0]
        C_Mz_s = np.array(M_z["a"] * (aoa_kite**2) + M_z["b"] * aoa_kite + M_z["c"])[0]

        # subtract support structure aero coefficients for this wind speed, sideslip and aoa combination from merged_df
        df.loc[df["sideslip"] == k, "CF_X"] -= C_Fx_s
        df.loc[df["sideslip"] == k, "CF_Y"] -= C_Fy_s
        df.loc[df["sideslip"] == k, "CF_Z"] -= C_Fz_s
        df.loc[df["sideslip"] == k, "CM_X"] -= C_Mx_s
        df.loc[df["sideslip"] == k, "CM_Y"] -= C_My_s
        df.loc[df["sideslip"] == k, "CM_Z"] -= C_Mz_s

    return df


# function that translates coordinate system
def translate_coordinate_system(
    df: pd.DataFrame,
    x_hinge: float,
    z_hinge: float,
    l_cg: float,
    alpha_cg_delta_with_rod: float,
    c_ref: float,
) -> pd.DataFrame:

    alpha_cg = np.deg2rad(df["aoa"] - alpha_cg_delta_with_rod)
    x_cg = l_cg * np.cos(alpha_cg)
    z_cg = l_cg * np.sin(alpha_cg)
    x_hcg = (x_hinge + x_cg) / (1000 * c_ref)
    z_hcg = (z_hinge + z_cg) / (1000 * c_ref)

    # rotation of coordinate system: force coefficients change
    df["C_L"] = df["CF_Z"] * -1
    df["C_S"] = df["CF_Y"] * -1
    df["C_D"] = df["CF_X"]

    df["C_roll"] = df["CM_X"] - df["CF_Y"] * z_hcg
    df["C_pitch"] = -df["CM_Y"] + df["CF_Z"] * x_hcg - df["CF_X"] * z_hcg
    df["C_yaw"] = -df["CM_Z"] - df["CF_Y"] * x_hcg

    return df


def correcting_for_sideslip(df: pd.DataFrame) -> pd.DataFrame:
    """
    [F_X_new]   [cos(β)  -sin(β)  0] [F_X_old]
    [F_Y_new] = [sin(β)   cos(β)  0] [F_Y_old]
    [F_Z_new]   [  0       0      1] [F_Z_old]
    """

    ## Grabbing sideslip array and converting to radians
    beta = np.deg2rad(df["sideslip"])
    # beta = np.deg2rad(10) * np.ones_like(df["sideslip"])

    ## Defining rotation matrix for each row
    def create_rotation_matrix(beta_angle):
        return np.array(
            [
                [np.cos(beta_angle), np.sin(beta_angle), 0],
                [-np.sin(beta_angle), np.cos(beta_angle), 0],
                [0, 0, 1],
            ]
        )

    # Create arrays for forces and moments
    forces = np.array([df["C_D"], df["C_S"], df["C_L"]]).T
    moments = np.array([df["C_pitch"], df["C_yaw"], df["C_roll"]]).T

    # Initialize arrays for corrected forces and moments
    corrected_forces = np.zeros_like(forces)
    corrected_moments = np.zeros_like(moments)

    # Apply rotation to each row
    for i in range(len(df)):
        R = create_rotation_matrix(beta[i])
        corrected_forces[i] = R @ forces[i]
        corrected_moments[i] = R @ moments[i]

    # Update dataframe with corrected values
    df["C_D"], df["C_S"], df["C_L"] = corrected_forces.T
    df["C_pitch"], df["C_yaw"], df["C_roll"] = corrected_moments.T

    return df


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
    df["Rey"] = df["Density"] * df["vw"] * c_ref / dynamic_viscosity

    return df


def read_csv_with_locale(file_path: Path) -> pd.DataFrame:
    """
    Reads a CSV file handling different locale decimal separators.

    Parameters:
    file_path (str): Path to the CSV file
    dtypes (dict): Dictionary of column data types

    Returns:
    pandas.DataFrame: The loaded and corrected dataframe
    """
    # Specifying the datatypes of each column
    dtypes = {
        "Filename": "str",
        "vw": "float64",
        "aoa": "float64",
        "sideslip": "float64",
        "Date": "str",
        "F_X": "float64",
        "F_Y": "float64",
        "F_Z": "float64",
        "M_X": "float64",
        "M_Y": "float64",
        "M_Z": "float64",
        "Dpa": "float64",
        "Pressure": "float64",
        "Temp": "float64",
        "Density": "float64",
    }

    # First, try to read the CSV with a decimal handler
    try:
        df = pd.read_csv(
            file_path,
            dtype={
                col: "str" for col in dtypes.keys()
            },  # Read everything as string first
            thousands=None,  # Disable thousands separator handling
        )

        # Function to safely convert string to float
        def safe_float_convert(x):
            if pd.isna(x):
                return np.nan
            try:
                return float(x)
            except ValueError:
                # Try replacing comma with period
                try:
                    return float(x.replace(",", "."))
                except:
                    return np.nan

        # Convert numeric columns
        for col, dtype in dtypes.items():
            if dtype == "float64" and col in df.columns:
                df[col] = df[col].apply(safe_float_convert)

        return df

    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        raise


def processing_raw_lvm_data_into_csv(
    folder_dir: Path,
    S_ref: float,
    c_ref: float,
    support_struc_aero_interp_coeffs_path: Path,
    x_hinge: float,
    z_hinge: float,
    l_cg: float,
    alpha_cg_delta_with_rod: float,
    delta_celsius_kelvin: float,
    mu_0: float,
    T_0: float,
    C_suth: float,
    delta_aoa: float,
):

    for i, file in enumerate(folder_dir.iterdir()):
        file_name = file.name
        aoa_kite = file_name.split("_")
        # print(f"aoa_kite: {aoa_kite}")
        if "raw" in file.name:
            df = read_csv_with_locale(file)
            print(f"\n File: {file_name}")
            # print(f"BEGIN columns: {df.columns}")

            # Storing the raw values
            df["F_X_raw"] = df["F_X"]
            df["F_Y_raw"] = df["F_Y"]
            df["F_Z_raw"] = df["F_Z"]
            df["M_X_raw"] = df["M_X"]
            df["M_Y_raw"] = df["M_Y"]
            df["M_Z_raw"] = df["M_Z"]

            # Compute Average Zero-Run values
            if i == 0:  # and df["vw"].unique()[0] < 1.5:
                # print(f'i: {i}, vw: {df["vw"].unique()}')
                # Computing the average measured zero-run values
                zero_F_X = df["F_X"].mean()
                zero_F_Y = df["F_Y"].mean()
                zero_F_Z = df["F_Z"].mean()
                zero_M_X = df["M_X"].mean()
                zero_M_Y = df["M_Y"].mean()
                zero_M_Z = df["M_Z"].mean()

            else:

                # TODO: This can be simplfied and taken from the filename
                # 1. Correcting the angle of attack, rounding to 1 decimal (like the folder name)
                df["aoa_kite"] = round(df["aoa"] - delta_aoa, 1)

                # 2. Substracting the zero-run values
                df["F_X"] -= zero_F_X
                df["F_Y"] -= zero_F_Y
                df["F_Z"] -= zero_F_Z
                df["M_X"] -= zero_M_X
                df["M_Y"] -= zero_M_Y
                df["M_Z"] -= zero_M_Z

                # 3. Nondimensionalizing
                df = nondimensionalize(df, S_ref, c_ref)

                # 4. Substracting support structure aerodynamic coefficients
                df = substract_support_structure_aero_coefficients(
                    df, support_struc_aero_interp_coeffs_path
                )

                # 5. Calculate signal-to-noise ratio
                # ## comparing no support--to--raw
                # df["SNR_CF_X"] = df["CF_X"] / df["CF_X_raw"]
                # df["SNR_CF_Y"] = df["CF_Y"] / df["CF_Y_raw"]
                # df["SNR_CF_Z"] = df["CF_Z"] / df["CF_Z_raw"]
                # df["SNR_CM_X"] = df["CM_X"] / df["CM_X_raw"]
                # df["SNR_CM_Y"] = df["CM_Y"] / df["CM_Y_raw"]
                # df["SNR_CM_Z"] = df["CM_Z"] / df["CM_Z_raw"]

                ## comparing with support measured --to-- raw, only difference is zero-run
                df["SNR_CF_X"] = df["F_X"] / df["F_X_raw"]
                df["SNR_CF_Y"] = df["F_Y"] / df["F_Y_raw"]
                df["SNR_CF_Z"] = df["F_Z"] / df["F_Z_raw"]
                df["SNR_CM_X"] = df["M_X"] / df["M_X_raw"]
                df["SNR_CM_Y"] = df["M_Y"] / df["M_Y_raw"]
                df["SNR_CM_Z"] = df["M_Z"] / df["M_Z_raw"]

                # 4. Translate coordinate system
                df = translate_coordinate_system(
                    df,
                    x_hinge,
                    z_hinge,
                    l_cg,
                    alpha_cg_delta_with_rod,
                    c_ref,
                )

                # 5. Correct for sideslip
                df = correcting_for_sideslip(df)

                # Add dynamic viscosity and Reynolds number
                df = add_dyn_visc_and_reynolds(
                    df, delta_celsius_kelvin, mu_0, T_0, C_suth, c_ref
                )
                # Dropping columns that are no longer needed
                columns_to_drop = [
                    "aoa",
                    "F_X",
                    "F_Y",
                    "F_Z",
                    "M_X",
                    "M_Y",
                    "M_Z",
                    "F_X_raw",
                    "F_Y_raw",
                    "F_Z_raw",
                    "M_X_raw",
                    "M_Y_raw",
                    "M_Z_raw",
                    "CF_X",
                    "CF_Y",
                    "CF_Z",
                    "CM_X",
                    "CM_Y",
                    "CM_Z",
                    "CF_X_raw",
                    "CF_Y_raw",
                    "CF_Z_raw",
                    "CM_X_raw",
                    "CM_Y_raw",
                    "CM_Z_raw",
                    "Filename",
                    "Date",
                    "Dpa",
                    "Pressure",
                    "Temp",
                    "Density",
                ]
                df.drop(columns=columns_to_drop, inplace=True)

                # Save the processed data to a csv file
                # file_name_without_raw = df["Filename"].unique()[0]
                vw_unique_round = round(df["vw"].unique()[0], 0)
                new_file_name = f"vw_{vw_unique_round:.0f}"
                df.to_csv(folder_dir / f"{new_file_name}.csv", index=False)
                # print(f"END columns: {df.columns}")


def main():
    S_ref = 0.46
    c_ref = 0.4
    support_struc_aero_interp_coeffs_path = (
        Path(project_dir) / "processed_data" / "support_struc_aero_interp_coeffs.csv"
    )
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
    delta_aoa = 7.25

    # processing all the folders
    print(f"\n Processing all the folders")
    for folder in os.listdir(Path(project_dir) / "processed_data" / "normal_csv"):
        if "alpha" in folder:
            folder_dir = Path(project_dir) / "processed_data" / "normal_csv" / folder
            processing_raw_lvm_data_into_csv(
                folder_dir,
                S_ref,
                c_ref,
                support_struc_aero_interp_coeffs_path,
                x_hinge,
                z_hinge,
                l_cg,
                alpha_cg_delta_with_rod,
                delta_celsius_kelvin,
                mu_0,
                T_0,
                C_suth,
                delta_aoa,
            )


if __name__ == "__main__":
    main()
