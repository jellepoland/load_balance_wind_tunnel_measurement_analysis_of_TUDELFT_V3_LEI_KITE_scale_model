import numpy as np
import pandas as pd
from pathlib import Path


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


def substract_support_structure_aero_coefficients_1_line(
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
        df.loc[df["sideslip"] == k, "C_D"] -= C_Fx_s
        df.loc[df["sideslip"] == k, "C_S"] -= C_Fy_s
        df.loc[df["sideslip"] == k, "C_L"] -= C_Fz_s
        df.loc[df["sideslip"] == k, "C_roll"] -= C_Mx_s
        df.loc[df["sideslip"] == k, "C_pitch"] -= C_My_s
        df.loc[df["sideslip"] == k, "C_yaw"] -= C_Mz_s

    return df


def substract_support_structure_aero_coefficients(
    df: pd.DataFrame, support_struc_aero_interp_coeffs_path: Path
) -> pd.DataFrame:

    # Read the interpolated coefficients from the CSV file
    support_struc_aero_interp_coeffs = pd.read_csv(
        support_struc_aero_interp_coeffs_path
    )

    # print(f"columns: {support_struc_aero_interp_coeffs.columns}")

    # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
    cur_vw = df["vw"].unique()[0]
    supp_coeffs = support_struc_aero_interp_coeffs[
        support_struc_aero_interp_coeffs["vw"] == int(np.around(cur_vw))
    ]
    # Retrieve the angle of attack for the kite
    aoa_kite = df["aoa_kite"].unique()[0]

    for k in df["sideslip"].unique():
        # Select support structure aero coefficients for this sideslip
        c_s = supp_coeffs[supp_coeffs["sideslip"] == k]

        # For each aerodynamic channel (forces and moments), get the interpolation coefficients (m1, c1, m2, c2)
        F_x = c_s.loc[c_s["channel"] == "C_D", ["m1", "c1", "m2", "c2"]]
        F_y = c_s.loc[c_s["channel"] == "C_S", ["m1", "c1", "m2", "c2"]]
        F_z = c_s.loc[c_s["channel"] == "C_L", ["m1", "c1", "m2", "c2"]]
        M_x = c_s.loc[c_s["channel"] == "C_roll", ["m1", "c1", "m2", "c2"]]
        M_y = c_s.loc[c_s["channel"] == "C_pitch", ["m1", "c1", "m2", "c2"]]
        M_z = c_s.loc[c_s["channel"] == "C_yaw", ["m1", "c1", "m2", "c2"]]

        # Define a function to compute the interpolated value based on the angle of attack
        def interpolate_value(aoa, m1, c1, m2, c2, middle_alpha):
            if aoa <= middle_alpha:
                return m1 * aoa + c1  # Left segment
            else:
                return m2 * aoa + c2  # Right segment

        # Compute the middle alpha (mean value of the alpha range) to determine the split point
        middle_alpha = support_struc_aero_interp_coeffs["middle_alpha"].mean()

        # Compute support structure aero coefficients for this wind speed, sideslip, and angle of attack combination
        C_Fx_s = interpolate_value(
            aoa_kite,
            F_x["m1"].values[0],
            F_x["c1"].values[0],
            F_x["m2"].values[0],
            F_x["c2"].values[0],
            middle_alpha,
        )
        C_Fy_s = interpolate_value(
            aoa_kite,
            F_y["m1"].values[0],
            F_y["c1"].values[0],
            F_y["m2"].values[0],
            F_y["c2"].values[0],
            middle_alpha,
        )
        C_Fz_s = interpolate_value(
            aoa_kite,
            F_z["m1"].values[0],
            F_z["c1"].values[0],
            F_z["m2"].values[0],
            F_z["c2"].values[0],
            middle_alpha,
        )
        C_Mx_s = interpolate_value(
            aoa_kite,
            M_x["m1"].values[0],
            M_x["c1"].values[0],
            M_x["m2"].values[0],
            M_x["c2"].values[0],
            middle_alpha,
        )
        C_My_s = interpolate_value(
            aoa_kite,
            M_y["m1"].values[0],
            M_y["c1"].values[0],
            M_y["m2"].values[0],
            M_y["c2"].values[0],
            middle_alpha,
        )
        C_Mz_s = interpolate_value(
            aoa_kite,
            M_z["m1"].values[0],
            M_z["c1"].values[0],
            M_z["m2"].values[0],
            M_z["c2"].values[0],
            middle_alpha,
        )

        # Subtract support structure aero coefficients from the main dataframe for this wind speed, sideslip, and aoa combination
        df.loc[df["sideslip"] == k, "C_D"] -= C_Fx_s
        df.loc[df["sideslip"] == k, "C_S"] -= C_Fy_s
        df.loc[df["sideslip"] == k, "C_L"] -= C_Fz_s
        df.loc[df["sideslip"] == k, "C_roll"] -= C_Mx_s
        df.loc[df["sideslip"] == k, "C_pitch"] -= C_My_s
        df.loc[df["sideslip"] == k, "C_yaw"] -= C_Mz_s

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


def processing_raw_lvm_data_into_csv(
    folder_dir: Path,
    S_ref: float,
    c_ref: float,
    is_kite: bool,
    is_zigzag: bool,
    support_struc_aero_interp_coeffs_path: Path,
    x_hinge: float,
    z_hinge: float,
    l_cg: float,
    alpha_cg_delta_with_rod: float,
    delta_celsius_kelvin: float,
    mu_0: float,
    T_0: float,
    C_suth: float,
    delta_aoa_rod_to_alpha: float,
):
    if is_kite:
        print(f"PROCESSING KITE, substracting support-structure")
    else:
        print(f"PROCESSING SUPPORT-STRUCTURE")

    if is_zigzag:
        print(f"Processing ZIGZAG")

    for i, file in enumerate(folder_dir.iterdir()):
        # only read the raw files
        if "raw" not in file.name:
            continue

        # Read csv file
        df = read_csv_with_locale(file)
        # compute rounded vw
        vw_unique_round = int(round(df["vw"].unique()[0], 0))
        print(f"\n---------------------")
        print(f"File: {file.name}, \nvw: {vw_unique_round}")

        # If zz correcting for the sideslip
        if is_zigzag:
            df["sideslip"] -= 1.8

        # Storing the raw values
        df["F_X_raw"] = df["F_X"]
        df["F_Y_raw"] = df["F_Y"]
        df["F_Z_raw"] = df["F_Z"]
        df["M_X_raw"] = df["M_X"]
        df["M_Y_raw"] = df["M_Y"]
        df["M_Z_raw"] = df["M_Z"]

        # 2. compute Average Zero-Run values
        if df["vw"].unique()[0] < 1.5:
            print(f'i: {i}, vw: {df["vw"].unique()}')
            # print(f'i: {i}, vw: {df["vw"].unique()}')
            # Computing the average measured zero-run values
            zero_F_X = df["F_X"].mean()
            zero_F_Y = df["F_Y"].mean()
            zero_F_Z = df["F_Z"].mean()
            zero_M_X = df["M_X"].mean()
            zero_M_Y = df["M_Y"].mean()
            zero_M_Z = df["M_Z"].mean()

        ## Skipping vw = 5 for zigzag
        elif vw_unique_round == 5 and is_zigzag:
            print(f"Not processing because zigzag and vw: {vw_unique_round}")
            continue
        else:

            # TODO: This can be simplfied and taken from the filename
            # Correcting the angle of attack, rounding to 1 decimal (like the folder name)
            df["aoa_kite"] = round(df["aoa"] - delta_aoa_rod_to_alpha, 1)

            # And add dynamic viscosity and Reynolds number
            df = add_dyn_visc_and_reynolds(
                df, delta_celsius_kelvin, mu_0, T_0, C_suth, c_ref
            )

            # 1. Substracting the zero-run values
            df["F_X"] -= zero_F_X
            df["F_Y"] -= zero_F_Y
            df["F_Z"] -= zero_F_Z
            df["M_X"] -= zero_M_X
            df["M_Y"] -= zero_M_Y
            df["M_Z"] -= zero_M_Z

            # 2. Non-dimensionalize
            df = nondimensionalize(df, S_ref, c_ref)

            # 7. Calculate signal-to-noise ratio
            ## comparing RAW --to-- (RAW - zero-run)
            df["SNR_CF_X"] = df["CF_X"] / df["CF_X_raw"]
            df["SNR_CF_Y"] = df["CF_Y"] / df["CF_Y_raw"]
            df["SNR_CF_Z"] = df["CF_Z"] / df["CF_Z_raw"]
            df["SNR_CM_X"] = df["CM_X"] / df["CM_X_raw"]
            df["SNR_CM_Y"] = df["CM_Y"] / df["CM_Y_raw"]
            df["SNR_CM_Z"] = df["CM_Z"] / df["CM_Z_raw"]

            # ## comparing with support measured --to-- raw, only difference is zero-run
            # df["SNR_CF_X"] = df["F_X"] / df["F_X_raw"]
            # df["SNR_CF_Y"] = df["F_Y"] / df["F_Y_raw"]
            # df["SNR_CF_Z"] = df["F_Z"] / df["F_Z_raw"]
            # df["SNR_CM_X"] = df["M_X"] / df["M_X_raw"]
            # df["SNR_CM_Y"] = df["M_Y"] / df["M_Y_raw"]
            # df["SNR_CM_Z"] = df["M_Z"] / df["M_Z_raw"]

            # 3. Translate coordinate system ("CF_X" in "C_D" out)
            df = translate_coordinate_system(
                df,
                x_hinge,
                z_hinge,
                l_cg,
                alpha_cg_delta_with_rod,
                c_ref,
            )
            # 4. Correct for sideslip ("C_D" in "C_D" out)
            df = correcting_for_sideslip(df)

            if is_kite:
                # 5. Substracting support structure aerodynamic coefficients ("C_D" in "C_D" out)
                df = substract_support_structure_aero_coefficients(
                    df, support_struc_aero_interp_coeffs_path
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
                # "Filename",
                "Date",
                "Dpa",
                "Pressure",
                "Temp",
                "Density",
            ]
            for col in columns_to_drop:
                if col not in df.columns:
                    continue
                df.drop(columns=[col], inplace=True)

            # Save the processed data to a csv file
            # file_name_without_raw = df["Filename"].unique()[0]
            new_file_name = f"vw_{vw_unique_round:.0f}"
            if "ZZ" in file.name:
                new_file_name = f"ZZ_{new_file_name}"
            print(f"saving file: {new_file_name}")
            df.to_csv(folder_dir / f"{new_file_name}.csv", index=False)
            # print(f"END columns: {df.columns}")
