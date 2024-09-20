import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path


def defining_root_dir() -> str:
    # Find the root directory of the repository
    root_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
        root_dir = os.path.abspath(os.path.join(root_dir, ".."))
        if root_dir == "/":
            raise FileNotFoundError(
                "Could not find the root directory of the repository."
            )
    return root_dir


def plot_zigzag(
    root_dir: str, results_dir: str, fontsize: float, figsize: tuple
) -> None:

    path_interp_coeffs = (
        Path(root_dir) / "processed_data" / "zigzag" / "interp_coeff.csv"
    )
    interp_coeffs = pd.read_csv(path_interp_coeffs)

    path_labbook_zz = Path(root_dir) / "processed_data" / "zigzag" / "labbook_zz.csv"
    data = pd.read_csv(path_labbook_zz, delimiter=";")
    data["SampleIndex"] = data.groupby("Filename").cumcount()

    # Step 2: Get unique filenames from the 'Filename' column
    unique_filenames = data["Filename"].unique()

    # Step 4: Define the column names for the data files
    column_names = ["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "error"]

    # Step 5: Initialize an empty list to hold DataFrames
    dataframes = []

    # Step 6: Loop through each filename, read the content with column names, add a sample index and filename, and append to the list
    for filename in unique_filenames:
        # Adjust the delimiter as needed, e.g., '\t' for tab-separated values or ' ' for space-separated values
        path_filename = Path(root_dir) / "processed_data" / "zigzag" / f"{filename}.txt"
        df = pd.read_csv(
            path_filename, delimiter="\t", names=column_names
        )  # Assuming tab-separated values for this example
        df["SampleIndex"] = range(len(df))  # Add a SampleIndex column
        df["Filename"] = filename  # Add the filename as a column
        dataframes.append(df)

    # Step 7: Concatenate all DataFrames into a single DataFrame
    big_dataframe = pd.concat(dataframes, ignore_index=True)

    merged_df = pd.merge(data, big_dataframe, on=["Filename", "SampleIndex"])
    merged_df.rename(columns={"vw": "vw_actual"}, inplace=True)
    merged_df["vw"] = np.around(merged_df["vw_actual"], 0)
    merged_df = merged_df[merged_df["vw"] != 0]

    output_normal = np.array(
        [3.616845, 4.836675, 802.681168, 4.386467, 171.123110, -0.111237]
    )
    output_zigzag = np.array(
        [3.545030, 4.522181, 802.495461, 4.192615, 171.237310, -0.041342]
    )
    output_normal_vw5 = np.array(
        [3.412569, 5.041744, 800.806084, 4.026951, 172.121360, 0.034760]
    )

    cols = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    merged_df.loc[merged_df["Filename"].str.startswith("ZZ"), cols] -= output_zigzag
    # merged_df[(merged_df['Filename'].str.startswith('normal')) & (merged_df['vw'] != 5)] -= output_normal

    # Ensure the output_normal is correctly aligned with the selected rows
    selected_rows = merged_df[
        (merged_df["Filename"].str.startswith("normal")) & (merged_df["vw"] != 5)
    ]
    selected_rows_indices = selected_rows.index

    # Subtract output_normal from the selected rows
    merged_df.loc[selected_rows_indices, cols] -= output_normal

    # Ensure the output_normal is correctly aligned with the selected rows
    selected_rows = merged_df[
        (merged_df["Filename"].str.startswith("normal")) & (merged_df["vw"] == 5)
    ]
    selected_rows_indices = selected_rows.index

    # Subtract output_normal from the selected rows
    merged_df.loc[selected_rows_indices, cols] -= output_normal_vw5
    # merged_df[(merged_df['Filename'].str.startswith('normal')) & (merged_df['vw'] == 5)] -= output_normal_vw5

    # Step 3: Nondimensionalize the columns "F_X", "F_Y", "F_Z", "M_X", "M_Y", and "M_Z"
    rho = merged_df["Density"]
    V = merged_df["vw_actual"]
    S_ref = 0.46
    c_ref = 0.4

    # Nondimensionalize the force columns
    merged_df["Fx"] /= 0.5 * rho * V**2 * S_ref
    merged_df["Fy"] /= 0.5 * rho * V**2 * S_ref
    merged_df["Fz"] /= 0.5 * rho * V**2 * S_ref

    # Nondimensionalize the moment columns
    merged_df["Mx"] /= 0.5 * rho * V**2 * S_ref * c_ref
    merged_df["My"] /= 0.5 * rho * V**2 * S_ref * c_ref
    merged_df["Mz"] /= 0.5 * rho * V**2 * S_ref * c_ref

    # subtract the correction from the sideslip
    merged_df["sideslip"] -= 1.8

    # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
    aoa = merged_df["aoa"].unique()[0] - 7.25

    for v in merged_df["vw"].unique():
        supp_coeffs = interp_coeffs[interp_coeffs["vw"] == v]
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
                (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Fx"
            ] -= C_Fx_s
            merged_df.loc[
                (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Fy"
            ] -= C_Fy_s
            merged_df.loc[
                (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Fz"
            ] -= C_Fz_s
            merged_df.loc[
                (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Mx"
            ] -= C_Mx_s
            merged_df.loc[
                (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "My"
            ] -= C_My_s
            merged_df.loc[
                (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Mz"
            ] -= C_Mz_s

    # translate coordinate system
    # parameters necessary to translate moments (aka determine position of cg)
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

    # rotation of coordinate system: force coefficients change
    merged_df["C_L"] = merged_df["Fz"] * -1
    merged_df["C_S"] = merged_df["Fy"] * -1
    merged_df["C_D"] = merged_df["Fx"]

    merged_df["C_roll"] = merged_df["Mx"] - merged_df["Fy"] * merged_df["z_kite"]
    merged_df["C_pitch"] = (
        -merged_df["My"]
        + merged_df["Fz"] * merged_df["x_kite"]
        - merged_df["Fx"] * merged_df["z_kite"]
    )
    merged_df["C_yaw"] = -merged_df["Mz"] - merged_df["Fy"] * merged_df["x_kite"]

    # # Constants for Sutherland's law
    # C1 = 1.458e-6  # Sutherland's constant in kg/(m·s·sqrt(K))
    # S = 110.4  # Sutherland's temperature in K

    # # Reference length
    # c_ref = 0.4

    # # Calculate dynamic viscosity using Sutherland's law
    # merged_df['dynamic_viscosity'] = C1 * merged_df['Temp'] ** 1.5 / (merged_df['Temp'] + S)

    # # Calculate Reynolds number
    # merged_df['Re'] = (merged_df['Density'] * merged_df['vw_actual'] * c_ref) / merged_df['dynamic_viscosity'] #/ 1e5

    # add dynamic viscosity and reynolds number column
    T = merged_df["Temp"] + 273.15
    mu_0 = 1.716e-5
    T_0 = 273.15
    C_suth = 110.4
    c_ref = 0.4  # reference chord
    dynamic_viscosity = (
        mu_0 * (T / T_0) ** 0.5 * (T_0 + C_suth) / (T + C_suth)
    )  # sutherland's law
    merged_df["dyn_vis"] = dynamic_viscosity
    merged_df["Rey"] = (
        merged_df["Density"] * merged_df["vw_actual"] * c_ref / merged_df["dyn_vis"]
    )

    # # Define the coefficients and other relevant columns
    # coefficients = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
    # plot_titles = [
    #     "Drag coefficient",
    #     "Side-force coefficient",
    #     "Lift coefficient",
    #     "Roll moment coefficient",
    #     "Pitching moment coefficient",
    #     "Yawing moment coefficient",
    # ]
    # yaxis_names = ["C_D", "C_S", "C_L", "C_{roll}", "C_{pitch}", "C_{yaw}"]

    # defining coefficients
    coefficients = ["C_L", "C_D", "C_pitch"]
    plot_titles = [
        "Lift coefficient",
        "Drag coefficient",
        "Pitching moment coefficient",
    ]
    yaxis_names = ["C_L", "C_D", "C_{pitch}"]

    filename_column = "Filename"
    sideslip_column = "sideslip"
    wind_speed_column = "Rey"

    # Filter data for zigzag tape and without zigzag tape
    data_zz = merged_df[merged_df[filename_column].str.startswith("ZZnor")]
    data_no_zz = merged_df[merged_df[filename_column].str.startswith("nor")]

    # Get unique sideslip angles
    sideslip_angles = np.unique(np.abs(merged_df[sideslip_column]))
    # separate data
    sideslip = 0
    data_zz_sideslip = data_zz[data_zz[sideslip_column] == sideslip]
    data_no_zz_sideslip = data_no_zz[data_no_zz[sideslip_column] == sideslip]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    # fig.suptitle(rf'Sideslip angle $\beta$= {np.int64(sideslip)}$^o$', fontsize=16)
    axs = axes.flatten()
    for idx, coeff in enumerate(coefficients):
        ax = axs[idx]
        ax.plot(
            1e-5 * data_no_zz_sideslip[wind_speed_column],
            data_no_zz_sideslip[coeff],
            label="Without zigzag tape",
            marker="o",
            color="red",
        )
        ax.plot(
            1e-5 * data_zz_sideslip[wind_speed_column],
            data_zz_sideslip[coeff],
            label="With zigzag tape",
            marker="x",
            color="blue",
        )
        ax.set_xlim([1, 6])
        if idx == 0:
            ax.legend()

            ax.set_ylim([0.7, 0.9])
        elif idx == 1:  # CD
            ax.set_ylim([0.06, 0.16])
        elif idx == 2:  # C_pitch
            ax.set_ylim([0.1, 0.55])
        ax.set_xlabel(r"$Re \times 10^5$ [-]", fontsize=fontsize)
        ax.set_ylabel(rf"${yaxis_names[idx]}$ [-]", fontsize=fontsize)
        # ax.set_title(plot_titles[idx])
        ax.grid(True)

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.tight_layout()
    # Adjust layout to make room for the main title
    path_save = Path(results_dir) / f"zz_re_sweep_alpha_675_beta_0.pdf"
    plt.savefig(path_save)
    plt.show()


def main(results_dir: str, root_dir: str) -> None:

    # Increase font size for readability
    plt.rcParams.update({"font.size": 14})
    fontsize = 18
    figsize = (20, 6)

    plot_zigzag(root_dir, results_dir, fontsize, figsize)


if __name__ == "__main__":
    root_dir = defining_root_dir()
    results_dir = Path(root_dir) / "results"
    main(results_dir, root_dir)
