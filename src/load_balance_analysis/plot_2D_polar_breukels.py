from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_balance_analysis.functions_utils import project_dir, saving_pdf_and_pdf_tex
from plot_styling import plot_on_ax
import math


# def main(results_dir, project_dir):

#     # Load data
#     vsm_input_dir = Path(project_dir) / "data" / "vsm_input"
#     panel_index = 8
#     df = pd.read_csv(vsm_input_dir / f"polar_engineering_{panel_index}.csv")

#     df["alpha"] = np.rad2deg(df["alpha"])

#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     plot_on_ax(
#         axs[0],
#         df["alpha"],
#         df["cl"],
#         label="Corrected",
#         color="red",
#         linestyle="--",
#         # marker=marker,
#         is_with_grid=True,
#         x_label=r"$\alpha$ [$^\circ$]",
#         y_label=r"$C_{\mathrm{l}}$ [-]",
#     )
#     plot_on_ax(
#         axs[1],
#         df["alpha"],
#         df["cd"],
#         label=r"Corrected",
#         color="red",
#         linestyle="--",
#         # marker=marker,
#         is_with_grid=True,
#         x_label=r"$\alpha$ [$^\circ$]",
#         y_label=r"$C_{\mathrm{d}}$ [-]",
#     )
#     plot_on_ax(
#         axs[2],
#         df["alpha"],
#         df["cm"],
#         label=r"Corrected",
#         color="red",
#         linestyle="--",
#         # marker=marker,
#         is_with_grid=True,
#         x_label=r"$\alpha$ [$^\circ$]",
#         y_label=r"$C_{\mathrm{m}}$ [-]",
#     )
#     plot_on_ax(
#         axs[0],
#         df["alpha"],
#         df["cl_breukels"],
#         label=r"Breukels",
#         color="black",
#         linestyle="-",
#         # marker=marker,
#         is_with_grid=True,
#         x_label=r"$\alpha$ [$^\circ$]",
#         y_label=r"$C_{\mathrm{l}}$ [-]",
#     )
#     plot_on_ax(
#         axs[1],
#         df["alpha"],
#         df["cd_breukels"],
#         label=r"Breukels",
#         color="black",
#         linestyle="-",
#         # marker=marker,
#         is_with_grid=True,
#         x_label=r"$\alpha$ [$^\circ$]",
#         y_label=r"$C_{\mathrm{d}}$ [-]",
#     )
#     plot_on_ax(
#         axs[2],
#         df["alpha"],
#         df["cm"],
#         label=r"Breukels",
#         color="black",
#         linestyle="-",
#         # marker=marker,
#         is_with_grid=True,
#         x_label=r"$\alpha$ [$^\circ$]",
#         y_label=r"$C_{\mathrm{m}}$ [-]",
#     )
#     axs[0].legend(loc="best")
#     axs[0].set_xlim([-30, 30])
#     axs[1].set_xlim([-30, 30])
#     axs[2].set_xlim([-30, 30])
#     # Save the plot
#     plt.tight_layout()
#     file_name = "2D_polar_correction_to_breukels"
#     saving_pdf_and_pdf_tex(results_dir, file_name)


def reading_profile_from_airfoil_dat_files(filepath):
    """
    Read main characteristics of an ILE profile from a .dat file.

    Parameters:
    filepath (str): The name of the file containing the profile data. Should be a .dat file

    The .dat file should follow the XFoil norm: points start at the trailing edge (TE),
    go to the leading edge (LE) through the extrado, and come back to the TE through the intrado.

    Returns:
    dict: A dictionary containing the profile name, tube diameter, depth, x_depth, and TE angle.
          The keys are "name", "tube_diameter", "depth", "x_depth", and "TE_angle".
    """
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Initialize variables
    profile_name = lines[0].strip()  # Name of the profile
    tube_diameter = None  # LE tube diameter of the profile, in % of the chord
    depth = -float("inf")  # Depth of the profile, in % of the chord
    x_depth = None  # Position of the maximum depth of the profile, in % of the chord
    TE_angle_deg = None  # Angle of the TE

    # Compute tube diameter of the LE
    # Left empty for now because it is read with the ribs coordinate data in "reading_surfplan_txt"
    # It could also be determined here geometrically
    #
    #

    # Read profile points to find maximum depth and its position
    for line in lines[1:]:
        x, y = map(float, line.split())
        if y > depth:
            depth = y
            x_depth = x

    # Calculate the TE angle
    # TE angle is defined here as the angle between the horizontal and the TE extrado line
    # The TE extrado line is going from the TE to the 3rd point of the extrado from the TE
    points_list = []
    if len(lines) > 4:
        (x1, y1) = map(float, lines[1].split())
        (x2, y2) = map(float, lines[3].split())
        delta_x = x2 - x1
        delta_y = y2 - y1
        TE_angle_rad = math.atan2(delta_y, delta_x)
        TE_angle_deg = 180 - math.degrees(TE_angle_rad)
        # appending all points to the list
        for line in lines[1:]:
            x, y = map(float, line.split())
            points_list.append([x, y])

    else:
        TE_angle_deg = None  # Not enough points to calculate the angle

    return {
        "name": profile_name,
        "tube_diameter": tube_diameter,
        "depth": depth,
        "x_depth": x_depth,
        "TE_angle": TE_angle_deg,
        "points": points_list,
    }


def main(results_dir, project_dir):
    # Airfoil data
    filepath = Path(project_dir) / "data" / "prof_mid_span.dat"
    profile = reading_profile_from_airfoil_dat_files(filepath)
    points = profile["points"]
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    # Load polar data
    vsm_input_dir = Path(project_dir) / "data" / "vsm_input"
    panel_index = 8
    df = pd.read_csv(vsm_input_dir / f"polar_engineering_{panel_index}.csv")

    df["alpha"] = np.rad2deg(df["alpha"])

    # Create figure and gridspec layout
    fig = plt.figure(figsize=(15, 6))
    spec = fig.add_gridspec(
        2, 3, height_ratios=[0.6, 2]
    )  # 2 rows, 2 columns, taller second row

    # Airfoil plot (spanning the entire first row)
    ax0 = fig.add_subplot(spec[0, :])  # Span all columns
    ax0.plot(x, y, label="Profile at mid-span", color="blue")
    ax0.set_xlabel("x/c [m]")
    ax0.set_ylabel("y/c [m]")
    ax0.grid(False)
    ax0.set_aspect("equal", adjustable="box")
    ax0.legend(loc="best")
    # ax0.set_title("Airfoil Profile")

    # CL-alpha plot
    ax1 = fig.add_subplot(spec[1, 0])  # Bottom left
    plot_on_ax(
        ax1,
        df["alpha"],
        df["cl"],
        label="Corrected",
        color="red",
        linestyle="--",
        is_with_grid=True,
        x_label=r"$\alpha$ [$^\circ$]",
        y_label=r"$C_{\mathrm{l}}$ [-]",
    )
    plot_on_ax(
        ax1,
        df["alpha"],
        df["cl_breukels"],
        label="Breukels",
        color="black",
        linestyle="-",
        is_with_grid=True,
        x_label=r"$\alpha$ [$^\circ$]",
        y_label=r"$C_{\mathrm{l}}$ [-]",
    )
    ax1.set_xlim([-30, 30])
    ax1.legend(loc="best")

    # Cd-alpha plot
    ax2 = fig.add_subplot(spec[1, 1])  # Bottom right
    plot_on_ax(
        ax2,
        df["alpha"],
        df["cd"],
        label="Corrected",
        color="red",
        linestyle="--",
        is_with_grid=True,
        x_label=r"$\alpha$ [$^\circ$]",
        y_label=r"$C_{\mathrm{d}}$ [-]",
    )
    plot_on_ax(
        ax2,
        df["alpha"],
        df["cd_breukels"],
        label="Breukels",
        color="black",
        linestyle="-",
        is_with_grid=True,
        x_label=r"$\alpha$ [$^\circ$]",
        y_label=r"$C_{\mathrm{d}}$ [-]",
    )
    ax2.set_xlim([-30, 30])

    # CL/CD-alpha plot
    ax3 = fig.add_subplot(spec[1, 2])  # Bottom right
    plot_on_ax(
        ax3,
        df["alpha"],
        df["cl"] / df["cd"],
        label="Corrected",
        color="red",
        linestyle="--",
        is_with_grid=True,
        x_label=r"$\alpha$ [$^\circ$]",
        y_label=r"$C_{\mathrm{l}}/C_{\mathrm{d}}$ [-]",
    )
    plot_on_ax(
        ax3,
        df["alpha"],
        df["cl_breukels"] / df["cd_breukels"],
        label="Breukels",
        color="black",
        linestyle="-",
        is_with_grid=True,
        x_label=r"$\alpha$ [$^\circ$]",
        y_label=r"$C_{\mathrm{l}}/C_{\mathrm{d}}$ [-]",
    )
    ax3.set_xlim([-30, 30])

    # Adjust layout and save
    plt.tight_layout()
    file_name = "2D_polar_correction_to_breukels"
    saving_pdf_and_pdf_tex(results_dir, file_name)

    # Show the plot
    plt.show()
