import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import project_dir

from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from VSM.plotting import generate_polar_data
from VSM.interactive import interactive_plot


def running_vsm_to_generate_csv_data(
    project_dir: str,
    vw: float,
    geom_scaling=6.5,
    height_correction_factor=0.82,
    is_with_corrected_polar=True,
    mu=1.76e-5,
    reference_point=None,
) -> None:
    if is_with_corrected_polar:
        print("Running VSM with corrected polar")
        name_appendix = ""
    else:
        print("Running VSM with breukels polar")
        name_appendix = "_no_correction"

    # Defining discretisation
    n_panels = 54
    spanwise_panel_distribution = "split_provided"

    ### rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
    vsm_input_path = Path(project_dir) / "data" / "vsm_input"
    csv_file_path = (
        Path(vsm_input_path)
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
        ### Correcting the geometry
        CAD_rib_i_0 = CAD_rib_i[0] / geom_scaling
        CAD_rib_i_1 = CAD_rib_i[1] / geom_scaling

        ## Height was off even tho chord and span are matching perfectly...
        CAD_rib_i_0[2] = CAD_rib_i_0[2] * height_correction_factor
        CAD_rib_i_1[2] = CAD_rib_i_1[2] * height_correction_factor

        if is_with_corrected_polar:
            ### using corrected polar
            df_polar_data = pd.read_csv(
                Path(vsm_input_path) / f"corrected_polar_{i}.csv"
            )
            alpha = df_polar_data["alpha"].values
            cl = df_polar_data["cl"].values
            cd = df_polar_data["cd"].values
            cm = df_polar_data["cm"].values
            polar_data = ["polar_data", np.array([alpha, cl, cd, cm])]
            CAD_wing.add_section(CAD_rib_i_0, CAD_rib_i_1, polar_data)
        else:
            ### using breukels
            CAD_wing.add_section(CAD_rib_i_0, CAD_rib_i_1, CAD_rib_i[2])

    wing_aero_CAD_19ribs = WingAerodynamics([CAD_wing])

    # ### Plotting reference point at mid-span plane
    # plt.figure()
    # LE = rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs[8][0] / geom_scaling
    # TE = rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs[8][1] / geom_scaling
    # LE[2] = LE[2] * height_correction_factor
    # TE[2] = TE[2] * height_correction_factor
    # plt.plot(LE[0], LE[2], "ro", label="LE")
    # plt.plot(TE[0], TE[2], "bo", label="TE")
    # plt.plot((TE[0] - 0.395), TE[2], "bx", label="LE (approximate)")
    # plt.plot(reference_point[0], reference_point[2], "go", label="Reference point")
    # plt.legend()
    # plt.axis("equal")
    # plt.grid()
    # plt.show()
    # plt.close()
    # breakpoint()

    # ### INTERACTIVE PLOT
    # interactive_plot(
    #     wing_aero_CAD_19ribs,
    #     vel=3.15,
    #     angle_of_attack=6.75,
    #     side_slip=10,
    #     yaw_rate=0,
    #     is_with_aerodynamic_details=True,
    # )
    # breakpoint()

    # Solvers
    VSM = Solver(
        aerodynamic_model_type="VSM",
        is_with_artificial_damping=False,
        mu=mu,
        reference_point=reference_point,
    )
    VSM_with_stall_correction = Solver(
        aerodynamic_model_type="VSM",
        is_with_artificial_damping=True,
        mu=mu,
        reference_point=reference_point,
    )

    ### alpha sweep
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
    polar_data, reynolds_number = generate_polar_data(
        solver=VSM,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=alphas_to_be_plotted,
        angle_type="angle_of_attack",
        angle_of_attack=0,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    polar_data_stall, _ = generate_polar_data(
        solver=VSM_with_stall_correction,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=alphas_to_be_plotted,
        angle_type="angle_of_attack",
        angle_of_attack=0,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    # Create dataframe and save to CSV
    path_to_csv = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"VSM_results_alpha_sweep_Rey_{(reynolds_number/1e5):.1f}{name_appendix}.csv"
    )
    pd.DataFrame(
        {
            "aoa": polar_data[0],
            "CL_no_stall": polar_data[1],
            "CL": polar_data_stall[1],
            "CD_no_stall": polar_data[2],
            "CD": polar_data_stall[2],
            "CL/CD_no_stall": np.array(polar_data[1]) / np.array(polar_data[2]),
            "CL/CD": np.array(polar_data_stall[1]) / np.array(polar_data_stall[2]),
        }
    ).to_csv(path_to_csv, index=False)

    ### beta sweep
    alpha = 6.75
    betas_to_be_plotted = [
        # -20,
        # -14,
        # -12,
        # -10,
        # -8,
        # -6,
        # -4,
        # -2,
        0,
        2,
        4,
        6,
        8,
        12,
        14,
        20,
    ]
    polar_data, _ = generate_polar_data(
        solver=VSM,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=alpha,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    polar_data_stall, _ = generate_polar_data(
        solver=VSM_with_stall_correction,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=alpha,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    # Create dataframe and save to CSV
    path_to_csv = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"VSM_results_beta_sweep_Rey_{(reynolds_number/1e5):.1f}_alpha_{alpha*100:.0f}{name_appendix}.csv"
    )
    pd.DataFrame(
        {
            "beta": polar_data[0],
            "CL": polar_data[1],
            "CL_stall": polar_data_stall[1],
            "CD": polar_data[2],
            "CD_stall": polar_data_stall[2],
            "CL/CD": np.array(polar_data[1]) / np.array(polar_data[2]),
            "CL/CD_stall": np.array(polar_data_stall[1])
            / np.array(polar_data_stall[2]),
            "CS": polar_data[3],
            "CS_stall": polar_data_stall[3],
        }
    ).to_csv(path_to_csv, index=False)

    ### beta sweep
    alpha = 11.95
    # betas_to_be_plotted = [
    #     # -20,
    #     # -14,
    #     # -12,
    #     # -10,
    #     # -8,
    #     # -6,
    #     # -4,
    #     # -2,
    #     0,
    #     2,
    #     4,
    #     6,
    #     8,
    #     12,
    #     14,
    #     20,
    # ]
    polar_data, _ = generate_polar_data(
        solver=VSM,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=alpha,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    polar_data_stall, _ = generate_polar_data(
        solver=VSM_with_stall_correction,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=alpha,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    # Create dataframe and save to CSV
    path_to_csv = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"VSM_results_beta_sweep_Rey_{(reynolds_number/1e5):.1f}_alpha_{alpha*100:.0f}{name_appendix}.csv"
    )
    pd.DataFrame(
        {
            "beta": polar_data[0],
            "CL": polar_data[1],
            "CL_stall": polar_data_stall[1],
            "CD": polar_data[2],
            "CD_stall": polar_data_stall[2],
            "CL/CD": np.array(polar_data[1]) / np.array(polar_data[2]),
            "CL/CD_stall": np.array(polar_data_stall[1])
            / np.array(polar_data_stall[2]),
            "CS": polar_data[3],
            "CS_stall": polar_data_stall[3],
        }
    ).to_csv(path_to_csv, index=False)

    #######################################################################
    ### Create additional data for plotting the moment coefficients
    #######################################################################

    ### alpha sweep
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
    polar_data, reynolds_number = generate_polar_data(
        solver=VSM,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=alphas_to_be_plotted,
        angle_type="angle_of_attack",
        angle_of_attack=0,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    polar_data_stall, _ = generate_polar_data(
        solver=VSM_with_stall_correction,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=alphas_to_be_plotted,
        angle_type="angle_of_attack",
        angle_of_attack=0,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    # Create dataframe and save to CSV
    path_to_csv = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"VSM_results_alpha_sweep_Rey_{(reynolds_number/1e5):.1f}{name_appendix}_moment.csv"
    )
    pd.DataFrame(
        {
            "aoa": polar_data[0],
            "CL_no_stall": polar_data[1],
            "CL": polar_data_stall[1],
            "CD_no_stall": polar_data[2],
            "CD": polar_data_stall[2],
            "CL/CD_no_stall": np.array(polar_data[1]) / np.array(polar_data[2]),
            "CL/CD": np.array(polar_data_stall[1]) / np.array(polar_data_stall[2]),
            "CMx": polar_data[4],
            "CMx_stall": polar_data_stall[4],
            "CMy": polar_data[5],
            "CMy_stall": polar_data_stall[5],
            "CMz": polar_data[6],
            "CMz_stall": polar_data_stall[6],
        }
    ).to_csv(path_to_csv, index=False)

    ### beta sweep
    alpha = 6.75
    betas_to_be_plotted = [
        # -20,
        # -14,
        # -12,
        # -10,
        # -8,
        # -6,
        # -4,
        # -2,
        0,
        2,
        4,
        6,
        8,
        12,
        14,
        20,
    ]
    polar_data, _ = generate_polar_data(
        solver=VSM,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=alpha,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    polar_data_stall, _ = generate_polar_data(
        solver=VSM_with_stall_correction,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=alpha,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    # Create dataframe and save to CSV
    path_to_csv = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"VSM_results_beta_sweep_Rey_{(reynolds_number/1e5):.1f}_alpha_{alpha*100:.0f}{name_appendix}_moment.csv"
    )

    pd.DataFrame(
        {
            "beta": polar_data[0],
            "CL": polar_data[1],
            "CL_stall": polar_data_stall[1],
            "CD": polar_data[2],
            "CD_stall": polar_data_stall[2],
            "CL/CD": np.array(polar_data[1]) / np.array(polar_data[2]),
            "CL/CD_stall": np.array(polar_data_stall[1])
            / np.array(polar_data_stall[2]),
            "CS": polar_data[3],
            "CS_stall": polar_data_stall[3],
            "CMx": polar_data[4],
            "CMx_stall": polar_data_stall[4],
            "CMy": polar_data[5],
            "CMy_stall": polar_data_stall[5],
            "CMz": polar_data[6],
            "CMz_stall": polar_data_stall[6],
        }
    ).to_csv(path_to_csv, index=False)

    ### beta sweep
    alpha = 11.95

    polar_data, _ = generate_polar_data(
        solver=VSM,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=alpha,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    polar_data_stall, _ = generate_polar_data(
        solver=VSM_with_stall_correction,
        wing_aero=wing_aero_CAD_19ribs,
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=alpha,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
        # reference_point=reference_point,
    )
    # Create dataframe and save to CSV
    path_to_csv = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"VSM_results_beta_sweep_Rey_{(reynolds_number/1e5):.1f}_alpha_{alpha*100:.0f}{name_appendix}_moment.csv"
    )
    pd.DataFrame(
        {
            "beta": polar_data[0],
            "CL": polar_data[1],
            "CL_stall": polar_data_stall[1],
            "CD": polar_data[2],
            "CD_stall": polar_data_stall[2],
            "CL/CD": np.array(polar_data[1]) / np.array(polar_data[2]),
            "CL/CD_stall": np.array(polar_data_stall[1])
            / np.array(polar_data_stall[2]),
            "CS": polar_data[3],
            "CS_stall": polar_data_stall[3],
            "CMx": polar_data[4],
            "CMx_stall": polar_data_stall[4],
            "CMy": polar_data[5],
            "CMy_stall": polar_data_stall[5],
            "CMz": polar_data[6],
            "CMz_stall": polar_data_stall[6],
        }
    ).to_csv(path_to_csv, index=False)

    return


def main():

    ## Computing the reference point, to be equal as used for calc. the wind tunnel data Moments
    x_displacement_from_te = -0.157  # -0.172
    z_displacement_from_te = -0.252
    te_point_full_size = np.array([2.16733994663813, 0, 10.889841638961673])
    geom_scaling = 6.5
    te_point_scaled = te_point_full_size / geom_scaling
    ## height was off even tho chord and span are matching perfectly...
    height_correction_factor = 0.82
    te_point_scaled[2] = te_point_scaled[2] * height_correction_factor
    reference_point = te_point_scaled + np.array(
        [x_displacement_from_te, 0, z_displacement_from_te]
    )
    print(f"reference_point: {reference_point}")
    # breakpoint()

    running_vsm_to_generate_csv_data(
        project_dir,
        vw=20,
        is_with_corrected_polar=True,
        geom_scaling=geom_scaling,
        height_correction_factor=height_correction_factor,
        reference_point=reference_point,
    )
    running_vsm_to_generate_csv_data(
        project_dir,
        vw=20,
        is_with_corrected_polar=False,
        geom_scaling=geom_scaling,
        height_correction_factor=height_correction_factor,
        reference_point=reference_point,
    )


if __name__ == "__main__":
    main()
