import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from pathlib import Path
from VSM.Solver import Solver
from VSM.plotting import generate_polar_data
from utils import PROJECT_DIR, create_wing_aero

file_path = Path(PROJECT_DIR) / "data" / "vsm_input" / "geometry_corrected.csv"
path_polar_data_dir = Path(PROJECT_DIR) / "data" / "vsm_input"
n_panels = 150
angle_of_attack = 6.8
side_slip = 0
yaw_rate = 0
Umag = 3.15
spanwise_panel_distribution = "linear"
wing_aero_breukels = create_wing_aero(
    file_path,
    n_panels,
    spanwise_panel_distribution,
    is_with_corrected_polar=False,
    path_polar_data_dir=path_polar_data_dir,
)
wing_aero_polar = create_wing_aero(
    file_path,
    n_panels,
    spanwise_panel_distribution,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)
# wing_aero_breukels.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
# wing_aero_polar.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

#### Solvers
VSM_base = Solver(
    aerodynamic_model_type="VSM",
    is_with_artificial_damping=False,
)
VSM_with_stall_correction = Solver(
    aerodynamic_model_type="VSM",
    is_with_artificial_damping=True,
)

#### INTERACTIVE PLOT
# interactive_plot(
#     wing_aero_breukels,
#     vel=Umag,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     is_with_aerodynamic_details=True,
# )
# breakpoint()

angle_range = np.linspace(0, 20, 20)
angle_type = "side_slip"
angle_of_attack = 6.8

save_folder = Path(PROJECT_DIR) / "processed_data" / "beta_sweep"


# ##
# polar_data, reynolds_number = generate_polar_data(
#     solver=VSM_base,
#     wing_aero=wing_aero_breukels,
#     angle_range=angle_range,
#     angle_type=angle_type,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     Umag=Umag,
# )
# polar_data = {
#     "beta": polar_data[0],
#     "CL": polar_data[1],
#     "CD": polar_data[2],
#     "CS": polar_data[3],
#     "CMx": polar_data[4],
#     "CMy": polar_data[5],
#     "CMz": polar_data[6],
# }
# df = pd.DataFrame(polar_data)
# df.to_csv(Path(save_folder) / f"polar_data_vsm_breukels.csv", index=False)
# ##
# polar_data, reynolds_number = generate_polar_data(
#     solver=VSM_with_stall_correction,
#     wing_aero=wing_aero_breukels,
#     angle_range=angle_range,
#     angle_type=angle_type,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     Umag=Umag,
# )
# polar_data = {
#     "beta": polar_data[0],
#     "CL": polar_data[1],
#     "CD": polar_data[2],
#     "CS": polar_data[3],
#     "CMx": polar_data[4],
#     "CMy": polar_data[5],
#     "CMz": polar_data[6],
# }
# df = pd.DataFrame(polar_data)
# df.to_csv(Path(save_folder) / f"polar_data_vsm_breukels_stall.csv", index=False)
# ##
# polar_data, reynolds_number = generate_polar_data(
#     solver=VSM_base,
#     wing_aero=wing_aero_polar,
#     angle_range=angle_range,
#     angle_type=angle_type,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     Umag=Umag,
# )
# polar_data = {
#     "beta": polar_data[0],
#     "CL": polar_data[1],
#     "CD": polar_data[2],
#     "CS": polar_data[3],
#     "CMx": polar_data[4],
#     "CMy": polar_data[5],
#     "CMz": polar_data[6],
# }
# df = pd.DataFrame(polar_data)
# df.to_csv(Path(save_folder) / f"polar_data_vsm_polar.csv", index=False)
# ##
# polar_data, reynolds_number = generate_polar_data(
#     solver=VSM_with_stall_correction,
#     wing_aero=wing_aero_polar,
#     angle_range=angle_range,
#     angle_type=angle_type,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     Umag=Umag,
# )
# polar_data = {
#     "beta": polar_data[0],
#     "CL": polar_data[1],
#     "CD": polar_data[2],
#     "CS": polar_data[3],
#     "CMx": polar_data[4],
#     "CMy": polar_data[5],
#     "CMz": polar_data[6],
# }
# df = pd.DataFrame(polar_data)
# df.to_csv(Path(save_folder) / f"polar_data_vsm_polar_stall.csv", index=False)

### Running once each script at alpha = 19 and storing the gamma_distribution
angle_of_attack = 6.8
side_slip = 16
for side_slip in [20]:
    wing_aero_breukels.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
    wing_aero_polar.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
    results_breukels = VSM_base.solve(wing_aero_breukels)
    results_breukels_stall = VSM_with_stall_correction.solve(wing_aero_breukels)
    results_polar = VSM_base.solve(wing_aero_polar)
    results_polar_stall = VSM_with_stall_correction.solve(wing_aero_polar)

    y_locations = [panel.aerodynamic_center[1] for panel in wing_aero_breukels.panels]
    min_y = min(y_locations)
    max_y = max(y_locations)
    span = max_y - min_y
    z_locations = [panel.aerodynamic_center[2] for panel in wing_aero_breukels.panels]
    min_z = min(z_locations)
    max_z = max(z_locations)
    height = max_z - min_z

    df_gamma = pd.DataFrame(
        {
            "y": y_locations / span,
            "z": z_locations / span,
            "gamma_breukels": results_breukels["gamma_distribution"],
            "gamma_breukels_stall": results_breukels_stall["gamma_distribution"],
            "gamma_polar": results_polar["gamma_distribution"],
            "gamma_polar_stall": results_polar_stall["gamma_distribution"],
            "cl_dist_breukels": results_breukels["cl_distribution"],
            "cl_dist_breukels_stall": results_breukels_stall["cl_distribution"],
            "cl_dist_polar": results_polar["cl_distribution"],
            "cl_dist_polar_stall": results_polar_stall["cl_distribution"],
            "cd_dist_breukels": results_breukels["cd_distribution"],
            "cd_dist_breukels_stall": results_breukels_stall["cd_distribution"],
            "cd_dist_polar": results_polar["cd_distribution"],
            "cd_dist_polar_stall": results_polar_stall["cd_distribution"],
        }
    )
    df_gamma.to_csv(
        Path(save_folder) / f"gamma_distribution_{int(side_slip)}.csv", index=False
    )
