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
n_panels = 130
angle_of_attack = 0
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


save_folder = Path(PROJECT_DIR) / "processed_data" / "alpha_sweep"

### Running once each script at alpha = 19 and storing the gamma_distribution
wing_aero_polar.va_initialize(3.15 * (4.2 / 5.6), 6, side_slip, yaw_rate)
results_polar = VSM_base.solve(wing_aero_polar)

y_locations = [panel.aerodynamic_center[1] for panel in wing_aero_breukels.panels]
min_y = min(y_locations)
max_y = max(y_locations)
span = max_y - min_y
z_locations = [panel.aerodynamic_center[2] for panel in wing_aero_breukels.panels]
min_z = min(z_locations)
max_z = max(z_locations)
z_locations = [z - min_z for z in z_locations]
height = max_z - min_z

df_gamma = pd.DataFrame(
    {
        "y": np.array(y_locations) / 6.5,
        "gamma_polar": results_polar["gamma_distribution"],
    }
)
df_gamma.to_csv(Path(save_folder) / "PIV_gamma_distribution.csv", index=False)
