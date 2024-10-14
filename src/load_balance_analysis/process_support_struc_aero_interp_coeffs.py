import os
from pathlib import Path
import pandas as pd
import numpy as np
from utils import project_dir


def read_raw_files_into_df(without_folder: Path) -> pd.DataFrame:
    # Initialize an empty list to store dataframes
    dfs = []

    # Iterate over each angle of attack folder
    for angle_folder in os.listdir(without_folder):
        angle_folder_path = os.path.join(without_folder, angle_folder)

        # Skip if it's not a directory
        if not os.path.isdir(angle_folder_path):
            continue

        # Iterate over each .txt file in the angle of attack folder
        for file_name in os.listdir(angle_folder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(angle_folder_path, file_name)

                # Read the .txt file into a pandas dataframe, skipping the first column (time)
                df = pd.read_csv(
                    file_path,
                    delim_whitespace=True,
                    header=None,
                    usecols=[1, 2, 3, 4, 5, 6],
                    names=["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
                )

                # Remove the ".txt" extension from the filename
                file_name_without_extension = os.path.splitext(file_name)[0]

                # Add filename (without extension) and sample index columns
                df["filename"] = file_name_without_extension
                df["sample_index"] = df.index

                # Append the dataframe to the list
                dfs.append(df)

    # Concatenate all dataframes into a single dataframe
    df_raw = pd.concat(dfs, ignore_index=True)
    return df_raw


def merge_df_with_labbook(
    df_raw: pd.DataFrame, data_dir: Path, labook_path: Path
) -> pd.DataFrame:
    labbook_path = Path(data_dir) / "processed_labbook.csv"
    labbook_df = pd.read_csv(labbook_path)

    # Rename columns in labbook_df
    labbook_df.rename(
        columns={"Filename": "filename", "sample index": "sample_index"}, inplace=True
    )

    # Merge combined_df with labbook_df based on "filename" and "sample_index" columns
    merged_df = pd.merge(
        df_raw, labbook_df, how="inner", on=["filename", "sample_index"]
    )

    # Rename the column "vw" to "vw_actual"
    merged_df.rename(columns={"vw": "vw_actual"}, inplace=True)

    return merged_df


def substract_zero_runs(
    merged_df: pd.DataFrame,
    columns_to_subtract: list,
) -> pd.DataFrame:
    # Find all distinct aoa values
    distinct_aoa = merged_df["aoa"].unique()

    # columns_range = ['Fx/maxFx', 'Fy/maxFy', 'Fz/maxFz', 'Mx/maxMx', 'M/maxMy', 'Mz/maxMz']

    # Create a new dataframe to store results
    result_df = pd.DataFrame()

    # Iterate over each distinct aoa value
    for aoa_value in distinct_aoa:
        # Get the reference rows where vw = 0 for the current aoa value
        reference_rows = merged_df[
            (merged_df["aoa"] == aoa_value) & (merged_df["vw"] == 0)
        ]

        # Iterate over each reference row
        for _, ref_row in reference_rows.iterrows():
            sideslip_value = ref_row["sideslip"]

            # Find matching rows where vw != 0 and sideslip and aoa are the same
            matching_rows = merged_df[
                (merged_df["aoa"] == aoa_value)
                & (merged_df["vw"] != 0)
                & (merged_df["sideslip"] == sideslip_value)
            ]

            # Subtract the specified columns
            for col in columns_to_subtract:
                matching_rows[col] = matching_rows[col] - ref_row[col]

            # Append the updated matching rows to the result dataframe
            result_df = pd.concat([result_df, matching_rows])

    # Optionally, you can reset the index of the result dataframe
    result_df.reset_index(drop=True, inplace=True)
    merged_df = result_df

    return merged_df


def nondimensionalize(
    merged_df: pd.DataFrame, S_ref: float, c_ref: float
) -> pd.DataFrame:

    # Non-dimensionalize the force balance outputs during merging
    force_coeffs = ["Cx", "Cy", "Cz"]
    for i, force_component in enumerate(["Fx", "Fy", "Fz"]):
        merged_df[force_coeffs[i]] = merged_df[force_component] / (
            0.5 * merged_df["Density"] * merged_df["vw_actual"] ** 2 * S_ref
        )

    mom_coeffs = ["Cmx", "Cmy", "Cmz"]
    for j, moment_component in enumerate(["Mx", "My", "Mz"]):
        merged_df[mom_coeffs[j]] = merged_df[moment_component] / (
            0.5 * merged_df["Density"] * merged_df["vw_actual"] ** 2 * S_ref * c_ref
        )
    return merged_df


def add_dynamic_viscosity_reynolds_number(
    merged_df: pd.DataFrame,
    celsius_to_kelvin: float,
    mu_0: float,
    T_0: float,
    C_suth: float,
    c_ref: float,
) -> pd.DataFrame:
    # add dynamic viscosity and reynolds number column
    T = merged_df["Temp"] + celsius_to_kelvin
    # mu_0 = 1.716e-5
    # T_0 = 273.15
    # C_suth = 110.4
    # c_ref = 0.4  # reference chord
    dynamic_viscosity = (
        mu_0 * (T / T_0) ** 0.5 * (T_0 + C_suth) / (T + C_suth)
    )  # sutherland's law
    merged_df["dyn_vis"] = dynamic_viscosity
    merged_df["Rey"] = (
        merged_df["Density"] * merged_df["vw_actual"] * c_ref / merged_df["dyn_vis"]
    )
    return merged_df


def process_merged_df(
    merged_df: pd.DataFrame,
    delta_aoa: float,
    S_ref: float,
    c_ref: float,
    celsius_to_kelvin: float,
    mu_0: float,
    T_0: float,
    C_suth: float,
) -> pd.DataFrame:
    # Extract approximate wind speed from the "filename" column
    merged_df["vw"] = merged_df["filename"].str.extract(r"vw_(\d+)").astype(int)

    # subtract 7.25 from the aoa column to switch to the kites angle of attack
    merged_df["aoa"] = (merged_df["aoa"] - delta_aoa).round(2)

    # Convert the sideslip column from degrees to radians and apply the cosine function
    # And multiply the vw_actual column by the resulting cosine values and update the vw_actual column
    # cos_sideslip = np.cos(np.radians(merged_df['sideslip']))
    # merged_df['vw_actual'] *= cos_sideslip

    ### 1. Subtract zero runs
    # Columns to be subtracted
    columns_to_subtract = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    merged_df = substract_zero_runs(merged_df, columns_to_subtract)

    # Replace infinite values with zero when wind speed is zero
    merged_df.loc[merged_df["vw_actual"] < 2, "vw_actual"] = 0

    ### 2. Non-dimensionalize
    merged_df = nondimensionalize(merged_df, S_ref, c_ref)

    ### 3. Determine the percentage of the max range measured by the balance
    ## Determine the percentage of the max range measured by the balance
    max_loads = [250, 600, 3500, 550, 500, 125]
    columns_range = [
        "Fx/maxFx",
        "Fy/maxFy",
        "Fz/maxFz",
        "Mx/maxMx",
        "My/maxMy",
        "Mz/maxMz",
    ]

    for k, maxload in enumerate(max_loads):
        merged_df[columns_range[k]] = np.abs(
            100 * merged_df[columns_to_subtract[k]] / maxload
        )

    ### 4. Add dynamic viscosity and reynolds number column
    merged_df = add_dynamic_viscosity_reynolds_number(
        merged_df, celsius_to_kelvin, mu_0, T_0, C_suth, c_ref
    )

    return merged_df


def interpolation(processed_df: pd.DataFrame, save_path: Path) -> None:
    ########################## interpolation #################################
    test = processed_df[
        ["aoa", "sideslip", "vw", "Cx", "Cy", "Cz", "Cmx", "Cmy", "Cmz"]
    ]
    alphalist = test["aoa"].unique()
    betalist = test["sideslip"].unique()
    vwlist = test["vw"].unique()
    # test = merged_df[(merged_df['vw']==5) & (merged_df['sideslip']==-20)]

    output_columns = ["Cx", "Cy", "Cz", "Cmx", "Cmy", "Cmz"]

    # polynomial interpolation
    def pol_inter(x, y, order):
        V = np.vander(x, order + 1)

        if order == 2:
            coefficients = np.linalg.solve(V, y)

        elif order == 1:
            coeff, res, rank, s = np.linalg.lstsq(V, y, rcond=None)
            b, c = coeff
            a = 0
            coefficients = np.array([a, b, c])

        return coefficients

    # create an empty dataframe to store the interpolation coefficients
    c_interp = pd.DataFrame(columns=["channel", "a", "b", "c", "sideslip", "vw"])

    # set row counter to 0
    row = 0

    # set order of polynomial to interpolate
    order = 1  # 1 for linear, 2 for quadratic

    for beta in betalist:
        for vw in vwlist:
            for channel in output_columns:

                x_val = alphalist
                y_val = np.array(
                    test[(test["sideslip"] == beta) & (test["vw"] == vw)][channel]
                )
                a, b, c = pol_inter(x_val, y_val, order)

                # append to dataframe
                c_interp.loc[row] = [channel, a, b, c, beta, vw]
                row = row + 1

    # save the interpolation coefficients as csv
    c_interp.to_csv(save_path, index=False)


def main():
    # Reading out raw files
    data_dir = Path(project_dir) / "data"
    without_folder = Path(data_dir) / "without"
    labook_path = Path(data_dir) / "processed_labbook.csv"
    df_raw = read_raw_files_into_df(without_folder)

    # merging
    merged_df = merge_df_with_labbook(df_raw, data_dir, labook_path)

    # processing
    delta_aoa = 7.25
    S_ref = 0.46  # reference area
    c_ref = 0.4  # reference chord
    celsius_to_kelvin = 273.15
    mu_0 = 1.716e-5
    T_0 = 273.15
    C_suth = 110.4
    processed_df = process_merged_df(
        merged_df, delta_aoa, S_ref, c_ref, celsius_to_kelvin, mu_0, T_0, C_suth
    )

    # interpolation
    save_path = (
        Path(project_dir) / "processed_data" / "support_struc_aero_interp_coeffs.csv"
    )
    interpolation(processed_df, save_path)


if __name__ == "__main__":
    main()
