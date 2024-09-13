import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution, plot_geometry


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


def running_vsm_to_generate_csv_data(root_dir: str, vw: float) -> None:

    # Defining discretisation
    n_panels = 54
    spanwise_panel_distribution = "split_provided"

    ### rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
    csv_file_path = (
        Path(root_dir)
        / "data"
        / "TUDELFT_V3_LEI_KITE_rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.csv"
    )
    (
        LE_x_array,
        LE_y_array,
        LE_z_array,
        TE_x_array,
        TE_y_array,
        TE_z_array,
        d_tube_array,
        camber_array,
    ) = np.loadtxt(csv_file_path, delimiter=",", skiprows=1, unpack=True)
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs = []

    for i in range(len(LE_x_array)):
        LE = np.array([LE_x_array[i], LE_y_array[i], LE_z_array[i]])
        TE = np.array([TE_x_array[i], TE_y_array[i], TE_z_array[i]])
        rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.append(
            [LE, TE, ["lei_airfoil_breukels", [d_tube_array[i], camber_array[i]]]]
        )
    CAD_wing = Wing(n_panels, spanwise_panel_distribution)

    for i, CAD_rib_i in enumerate(
        rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
    ):
        CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
    wing_aero_CAD_19ribs = WingAerodynamics([CAD_wing])

    # Solvers
    VSM = Solver(
        aerodynamic_model_type="VSM",
        is_with_artificial_damping=False,
    )
    VSM_with_stall_correction = Solver(
        aerodynamic_model_type="VSM",
        is_with_artificial_damping=True,
    )

    # Initialize lists to store the values
    aoa_list = []
    cl_list = []
    cl_list_stall = []
    cd_list = []
    cd_list_stall = []

    # Define input values
    plot_speeds = [vw]
    betas_to_be_plotted = [
        -20,
        -14,
        -12,
        -10,
        -8,
        -6,
        -4,
        -2,
        0,
        2,
        4,
        6,
        8,
        12,
        14,
        20,
    ]
    alphas_to_be_plotted = [
        -12.65,
        -7.15,
        -3.0,
        -2.25,
        2.35,
        4.75,
        6.75,
        8.8,
        10.95,
        11.95,
        12.8,
        14.0,
        15.75,
        17.85,
        19.75,
        22.55,
        24.0,
    ]
    # # selection
    # betas_to_be_plotted = [0, 5]
    # alphas_to_be_plotted = [2.35, 4.75]

    alpha_path_list = []
    beta_path_list = []
    for vw in plot_speeds:

        # Initialize lists to store the values
        aoa_list = []
        cl_list = []
        cl_list_stall = []
        cd_list = []
        cd_list_stall = []
        for aoa in alphas_to_be_plotted:
            side_slip = 0
            aoa_rad = np.deg2rad(aoa)

            # Setting va
            wing_aero_CAD_19ribs.va = (
                np.array(
                    [
                        np.cos(aoa_rad) * np.cos(side_slip),
                        np.sin(side_slip),
                        np.sin(aoa_rad),
                    ]
                )
                * vw
            )

            # Solving the model
            results = VSM.solve(wing_aero_CAD_19ribs)
            results_with_stall_correction = VSM_with_stall_correction.solve(
                wing_aero_CAD_19ribs
            )

            # Storing the values in lists
            aoa_list.append(aoa)
            cl_list.append(results["cl"])
            cl_list_stall.append(results_with_stall_correction["cl"])
            cd_list.append(results["cd"])
            cd_list_stall.append(results_with_stall_correction["cd"])

        # Save the data in a CSV file inside 'processed_data'
        Rey = results["Rey"]
        path_to_csv = (
            Path(root_dir)
            / "processed_data"
            / f"VSM_results_alpha_sweep_Rey_{(Rey/1e5):.1f}_new.csv"
        )
        alpha_path_list.append(path_to_csv)
        # Create dataframe and save to CSV
        pd.DataFrame(
            {
                "aoa": aoa_list,
                "CL": cl_list,
                "CL_stall": cl_list_stall,
                "CD": cd_list,
                "CD_stall": cd_list_stall,
                "CL/CD": np.array(cl_list) / np.array(cd_list),
                "CL/CD_stall": np.array(cl_list_stall) / np.array(cd_list_stall),
            }
        ).to_csv(path_to_csv, index=False)

        ### beta sweep
        # Initialize lists to store the values
        beta_list = []
        cl_list = []
        cl_list_stall = []
        cd_list = []
        cd_list_stall = []
        cs_list = []
        cs_list_stall = []
        for beta in betas_to_be_plotted:
            aoa = 11.95
            aoa_rad = np.deg2rad(aoa)
            side_slip = np.deg2rad(beta)

            # Setting va
            wing_aero_CAD_19ribs.va = (
                np.array(
                    [
                        np.cos(aoa_rad) * np.cos(side_slip),
                        np.sin(side_slip),
                        np.sin(aoa_rad),
                    ]
                )
                * vw
            )

            # Solving the model
            results = VSM.solve(wing_aero_CAD_19ribs)
            results_with_stall_correction = VSM_with_stall_correction.solve(
                wing_aero_CAD_19ribs
            )

            # Storing the values in lists
            beta_list.append(beta)
            cl_list.append(results["cl"])
            cl_list_stall.append(results_with_stall_correction["cl"])
            cd_list.append(results["cd"])
            cd_list_stall.append(results_with_stall_correction["cd"])
            cs_list.append(results["cs"])
            cs_list_stall.append(results_with_stall_correction["cs"])

        # Save the data in a CSV file inside 'processed_data'
        path_to_csv = (
            Path(root_dir)
            / "processed_data"
            / f"VSM_results_beta_sweep_Rey_{(Rey/1e5):.1f}_new.csv"
        )
        beta_path_list.append(path_to_csv)
        # Create dataframe and save to CSV
        pd.DataFrame(
            {
                "beta": beta_list,
                "CL": cl_list,
                "CL_stall": cl_list_stall,
                "CD": cd_list,
                "CD_stall": cd_list_stall,
                "CL/CD": np.array(cl_list) / np.array(cd_list),
                "CL/CD_stall": np.array(cl_list_stall) / np.array(cd_list_stall),
                "CS": cs_list,
                "CS_stall": cs_list_stall,
            }
        ).to_csv(path_to_csv, index=False)
        print(f"----> Reynolds number: {Rey/1e5:.3f}1e5")

    return


def main():
    root_dir = defining_root_dir()
    vw = 3.15
    running_vsm_to_generate_csv_data(root_dir, vw)


if __name__ == "__main__":
    main()
