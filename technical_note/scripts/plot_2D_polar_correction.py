import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import math
from utils import PROJECT_DIR
from plot_styling import set_plot_style


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


def plot_airfoil_and_polars(
    file_path_airfoil,
    df_corrected,
    df_neuralfoil,
    df_flat_plate=None,
    df_breukels=None,
    output_path="airfoil_and_polars.pdf",
):
    """
    Plots an airfoil on the top row and CL-alpha, CD-alpha, CM-alpha in the bottom row.

    Parameters
    ----------
    airfoil_points : list of (float, float)
        List of (x, y) coordinate pairs defining the airfoil.
    df_corrected : pandas.DataFrame
        Must contain columns 'alpha', 'cl', 'cd', 'cm' (the "corrected" data).
    df_neuralfoil : pandas.DataFrame
        Must contain columns 'alpha', 'cl', 'cd', 'cm' (the "NeuralFoil" data).
    df_flat_plate : pandas.DataFrame, optional
        If provided, must contain columns 'alpha', 'cl', 'cd', 'cm' (the "Flat Plate" data).
    df_breukels : pandas.DataFrame, optional
        If provided, must contain columns 'alpha', 'cl', 'cd', 'cm' (the "Breukels" data).
    output_filename : str
        Filename (including extension) to save the resulting figure.

    Returns
    -------
    None
    """
    set_plot_style()
    # Make sure alphas in degrees for consistent plotting
    # (If they're already in degrees, then remove np.rad2deg)
    # Example: df_corrected["alpha"] = np.rad2deg(df_corrected["alpha"])
    # Do this according to how your data is stored.

    # Create the figure with GridSpec
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[0.8, 2]  # 0.6,2
    )  # top row is a bit smaller, bottom row is larger
    markersize = 3.5
    linewidth = 1.8
    # 1) Airfoil on top (spanning all 3 columns)
    # ax_airfoil = fig.add_subplot(gs[0, 0:3])
    # ax_airfoil = fig.add_subplot(gs[0, 0:2])
    ax_airfoil = fig.add_axes([0.25, 0.8, 0.5, 0.18])  # [left, bottom, width, height]

    profile = reading_profile_from_airfoil_dat_files(file_path_airfoil)
    airfoil_points = profile["points"]
    x_coords = [pt[0] for pt in airfoil_points]
    y_coords = [pt[1] for pt in airfoil_points]

    ax_airfoil.plot(x_coords, y_coords, color="black", linewidth=linewidth)
    ax_airfoil.set_aspect("equal", adjustable="box")
    ax_airfoil.set_xlabel("$x/c$")
    ax_airfoil.set_ylabel("$y/c$")
    ax_airfoil.legend(
        loc="best",
        handles=[plt.Line2D([0], [0], color="black", label="Mid-Span Airfoil")],
    )
    ax_airfoil.grid(False)

    ## Grab z-data from alpha_sweep
    # df = pd.read_csv(
    #     Path(PROJECT_DIR) / "processed_data" / "alpha_sweep" / "gamma_distribution.csv"
    # )
    # # 1) Wing Curvatura on top
    # # ax_airfoil = fig.add_subplot(gs[0, 0:3])
    # # ax_airfoil = fig.add_subplot(gs[0, 2:3])
    # ax_wing_curvature = fig.add_axes([0.6, 0.6, 0.3, 0.4])  # Adjust width and position

    # # profile = reading_profile_from_airfoil_dat_files(file_path_airfoil)
    # airfoil_points = profile["points"]
    # x_coords = np.array(df["y"].values) * 1.278 * 6.5
    # y_coords = np.array(df["z"].values) * 0.462 * 6.5

    # ax_wing_curvature.plot(x_coords, y_coords, color="black", linewidth=linewidth)
    # ax_wing_curvature.set_aspect("equal", adjustable="box")
    # ax_wing_curvature.set_xlabel("$y$")
    # ax_wing_curvature.set_ylabel("$z$")
    # ax_wing_curvature.legend(
    #     loc="best",
    #     handles=[plt.Line2D([0], [0], color="black", label="Wing front view")],
    # )
    # ax_wing_curvature.set_xlim(0, 4.5)
    # ax_wing_curvature.set_ylim(0, 3.5)
    # ax_wing_curvature.grid(False)

    # 2) Bottom row: CL-alpha, CD-alpha, CM-alpha

    # Subplot for CL vs alpha
    height = 0.65
    width = 0.26
    margin = 0.06
    start = 0.07
    bottom = 0.05
    ax_cl = fig.add_axes(
        [start, bottom, width, height]
    )  # [left, bottom, width, height]
    ax_cd = fig.add_axes(
        [start + width + margin, bottom, width, height]
    )  # [left, bottom, width, height]
    ax_cm = fig.add_axes(
        [start + 2 * width + 2 * margin, bottom, width, height]
    )  # [left, bottom, width, height]
    ax_cl.plot(
        df_neuralfoil["alpha"],
        df_neuralfoil["cl"],
        linestyle="dashed",
        label="NeuralFoil",
        color="red",
        markersize=markersize,
        linewidth=linewidth,
    )

    linestyle_flat_plate = "-."
    linestyle_neuralfoil = "solid"
    linestyle_corrected = "dashed"
    linestyle_breukels = "dotted"

    if df_flat_plate is not None:
        ax_cl.plot(
            df_flat_plate["alpha"],
            df_flat_plate["cl"],
            linestyle=linestyle_flat_plate,
            label="Flat Plate",
            color="black",
            markersize=markersize,
            linewidth=linewidth,
        )
    if df_breukels is not None and "cl" in df_breukels.columns:
        ax_cl.plot(
            df_breukels["alpha"],
            df_breukels["cl"],
            linestyle=linestyle_breukels,
            label="Breukels",
            color="blue",
            markersize=markersize,
            linewidth=linewidth,
        )
    ax_cl.plot(
        df_corrected["alpha"],
        df_corrected["cl"],
        linestyle=linestyle_corrected,
        label="Corrected",
        color="blue",
        markersize=markersize,
        linewidth=linewidth,
    )
    ax_cl.set_xlabel(r"$\alpha$ [°]")
    ax_cl.set_ylabel(r"$C_L$ [-]")
    ax_cl.grid(True)
    ax_cl.legend()

    # Subplot for CD vs alpha
    # ax_cd = fig.add_subplot(gs[1, 1])

    ax_cd.plot(
        df_neuralfoil["alpha"],
        df_neuralfoil["cd"],
        linestyle=linestyle_neuralfoil,
        # label="NeuralFoil CD",
        color="red",
        markersize=markersize,
        linewidth=linewidth,
    )
    if df_flat_plate is not None:
        ax_cd.plot(
            df_flat_plate["alpha"],
            df_flat_plate["cd"],
            linestyle=linestyle_flat_plate,
            # label="Flat Plate CD",
            color="black",
            markersize=markersize,
            linewidth=linewidth,
        )
    if df_breukels is not None and "cd" in df_breukels.columns:
        ax_cd.plot(
            df_breukels["alpha"],
            df_breukels["cd"],
            linestyle=linestyle_breukels,
            # label="Breukels CD",
            color="blue",
            markersize=markersize,
            linewidth=linewidth,
        )
    ax_cd.plot(
        df_corrected["alpha"],
        df_corrected["cd"],
        linestyle=linestyle_corrected,
        # label="Corrected CD",
        color="blue",
        markersize=markersize,
        linewidth=linewidth,
    )
    ax_cd.set_xlabel(r"$\alpha$ [°]")
    ax_cd.set_ylabel(r"$C_D$ [-]")
    ax_cd.grid(True)
    # ax_cd.legend()

    # Subplot for CM vs alpha
    # ax_cm = fig.add_subplot(gs[1, 2])

    ax_cm.plot(
        df_neuralfoil["alpha"],
        df_neuralfoil["cm"],
        linestyle=linestyle_neuralfoil,
        # label="NeuralFoil CM",
        color="red",
        markersize=markersize,
        linewidth=linewidth,
    )
    if df_flat_plate is not None:
        ax_cm.plot(
            df_flat_plate["alpha"],
            df_flat_plate["cm"],
            linestyle=linestyle_flat_plate,
            # label="Flat Plate CM",
            color="black",
            markersize=markersize,
            linewidth=linewidth,
        )

    if df_breukels is not None and "cm" in df_breukels.columns:
        ax_cm.plot(
            df_breukels["alpha"],
            df_breukels["cm"],
            linestyle=linestyle_breukels,
            # label="Breukels CM",
            color="blue",
            markersize=markersize,
            linewidth=linewidth,
        )
    ax_cm.plot(
        df_corrected["alpha"],
        df_corrected["cm"],
        linestyle=linestyle_corrected,
        # label="Corrected CM",
        color="blue",
        markersize=markersize,
        linewidth=linewidth,
    )
    ax_cm.set_xlabel(r"$\alpha$ [°]")
    ax_cm.set_ylabel(r"$C_M$ [-]")
    ax_cm.grid(True)
    # ax_cm.legend()
    ax_cm.set_ylim(-2, 1)

    # fig.legend(
    #     loc="upper left",
    #     ncol=1,
    #     # nrow=5,
    #     bbox_to_anchor=(0.05, 0.96),
    #     frameon=True,
    # )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":

    plot_airfoil_and_polars(
        file_path_airfoil=PROJECT_DIR
        / "processed_data"
        / "2D_correction"
        / "prof_mid_span.dat",
        df_corrected=pd.read_csv(
            PROJECT_DIR / "processed_data" / "2D_correction" / "df_corrected_18.csv"
        ),
        df_neuralfoil=pd.read_csv(
            PROJECT_DIR / "processed_data" / "2D_correction" / "df_neuralfoil_18.csv"
        ),
        df_flat_plate=pd.read_csv(
            PROJECT_DIR / "processed_data" / "2D_correction" / "df_flat_plate_18.csv"
        ),
        df_breukels=pd.read_csv(
            PROJECT_DIR / "processed_data" / "2D_correction" / "df_breukels_18.csv"
        ),
        output_path=Path(PROJECT_DIR) / "results" / "airfoil_and_polars.pdf",
    )
