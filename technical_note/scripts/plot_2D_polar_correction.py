import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import math
import itertools
from utils import PROJECT_DIR
from plot_styling import set_plot_style


def find_best_circle_fixed(lines, tol=1e-5):
    """
    Given the lines from a .dat file (each line containing two floats: x and y),
    try to find a circle with center (a, 0) (with a in [0, 0.2]) and radius r in [0.01, 0.3]
    that passes through as many points as possible (within the tolerance tol).

    A point (x,y) is considered to lie on the circle defined by center (a,0) and radius r
    if |sqrt((x-a)^2 + y^2) - r| < tol.

    Parameters:
        lines (list of str): Lines read from the .dat file.
        tol (float): Tolerance for considering a point to be on the circle.

    Returns:
        best_circle (tuple): (a, 0, r) of the best candidate circle.
        best_count (int): Number of points that lie on that candidate circle.
    """
    # Parse the lines into a list of (x, y) points.
    points = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            x, y = map(float, line.split())
            points.append((x, y))
        except Exception:
            continue

    best_circle = None
    best_count = 0

    # Loop over all pairs of distinct points
    for (x1, y1), (x2, y2) in itertools.combinations(points, 2):
        # Avoid division by zero if x1 and x2 are too close
        if abs(x2 - x1) < 1e-12:
            continue

        # Compute candidate a using:
        # (x1 - a)^2 + y1^2 = (x2 - a)^2 + y2^2  ==>  a = (x2^2 - x1^2 + y2^2 - y1^2) / (2*(x2 - x1))
        a = (x2**2 - x1**2 + y2**2 - y1**2) / (2 * (x2 - x1))

        # Enforce a in [0, 0.2]
        if not (0 <= a <= 0.2):
            continue

        # Compute radius r from the first point:
        r = math.sqrt((x1 - a) ** 2 + y1**2)
        if r < 0.01 or r > 0.3:
            continue

        # Count how many points lie on the candidate circle (within tolerance)
        count = 0
        for x, y in points:
            dist = math.sqrt((x - a) ** 2 + y**2)
            if abs(dist - r) < tol:
                count += 1

        # Update the best candidate if this circle fits more points
        if count > best_count:
            best_count = count
            best_circle = (a, 0.0, r)

    return best_circle, best_count


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

    ## Calculating the tube diameter
    best_circle, best_count = find_best_circle_fixed(lines, tol=1e-5)
    if best_circle is not None:
        a, _, r = best_circle
        tube_diameter = 2 * r
    else:
        a = None
        tube_diameter = None

    return {
        "name": profile_name,
        "tube_diameter": tube_diameter,
        "tube_center_x": a,
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
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[0.8, 2.2]  # 0.6,2
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

    # ax_airfoil.plot(x_coords, y_coords, color="black", linewidth=linewidth)
    # ax_airfoil.set_aspect("equal", adjustable="box")
    # ax_airfoil.set_xlabel("$x/c$")
    # ax_airfoil.set_ylabel("$y/c$")
    # ax_airfoil.legend(
    #     loc="best",
    #     handles=[plt.Line2D([0], [0], color="black", label="Mid-Span Airfoil")],
    # )
    # ax_airfoil.grid(False)

    # profile_name = profile.get("name", "Airfoil")
    depth = profile.get("depth", 0.0)
    x_depth = profile.get("x_depth", 0.0)
    TE_angle = profile.get("TE_angle", 0.0)
    tube_diameter = profile.get("tube_diameter", 0.0)
    tube_center_x = profile.get("tube_center_x", 0.0)

    # Plot the airfoil outline
    ax_airfoil.plot(
        x_coords,
        y_coords,
        color="black",
        linewidth=linewidth,
        label=f"Mid-span airfoil",
    )

    # Plot the tube as a circle
    if tube_diameter is not None:
        circle = plt.Circle(
            (tube_center_x, 0),
            tube_diameter / 2,
            color="black",
            fill=False,
            linestyle="--",
            linewidth=linewidth,
            label=f"tube diameter: {tube_diameter:.2f} \ntube center: ({tube_center_x:.2f},0)",
        )
        ax_airfoil.add_artist(circle)

    # Set axis labels and equal aspect ratio
    ax_airfoil.set_xlabel("$x/c$")
    ax_airfoil.set_ylabel("$y/c$")
    ax_airfoil.set_aspect("equal", adjustable="box")

    # Create a legend text with the airfoil's details
    details = (
        # f"{profile_name}\n"
        # f"Depth: {depth:.2f}%\n"
        # f"x_depth: {x_depth:.2f}%\n"
        f"TE Angle: {TE_angle:.2f}°"
    )
    # Optionally, you could add an annotation for the highest (or any significant) point.
    # For example, if you wish to mark the point (x_depth, depth), uncomment below:
    #
    # ax_airfoil.scatter(
    #     x_depth,
    #     depth,
    #     color="black",
    #     marker="x",
    #     zorder=5,
    #     label=f"x: {x_depth:.2f}, y/c: {depth:.2f}",
    # )
    # ax_airfoil.annotate(
    #     f"({x_depth:.2f}, {depth:.2f})",
    #     xy=(x_depth, depth),
    #     xytext=(x_depth, depth + 0.05),
    #     arrowprops=dict(facecolor="black", arrowstyle="->"),
    # )

    # Use a legend that shows both the line label and the profile details as title
    # plot legend on the rightside of the airfoil
    ax_airfoil.legend(
        loc="upper right",
        bbox_to_anchor=(0.95, 0.98),  # Coordinates in the figure's coordinate system
        bbox_transform=fig.transFigure,  # Use the figure's coordinate system for bbox_to_anchor
        # title=details,
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
    height = 0.67
    width = 0.26
    margin = 0.06
    start = 0.07
    bottom = 0.07
    ax_cl = fig.add_axes(
        [start, bottom, width, height]
    )  # [left, bottom, width, height]
    ax_cd = fig.add_axes(
        [start + width + margin, bottom, width, height]
    )  # [left, bottom, width, height]
    ax_cm = fig.add_axes(
        [start + 2 * width + 2 * margin, bottom, width, height]
    )  # [left, bottom, width, height]

    linestyle_flat_plate = "-."
    linestyle_neuralfoil = "solid"
    linestyle_corrected = "dashed"
    linestyle_breukels = "dotted"

    ax_cl.plot(
        df_neuralfoil["alpha"],
        df_neuralfoil["cl"],
        linestyle=linestyle_neuralfoil,
        label="NeuralFoil",
        color="red",
        markersize=markersize,
        linewidth=linewidth,
    )

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
    ax_cl.set_ylabel(r"$C_{\textrm{l}}$ [-]")
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
    ax_cd.set_ylabel(r"$C_{\textrm{d}}$ [-]")
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
    ax_cm.set_ylabel(r"$C_{\textrm{M}}$ [-]")
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

    ax_cl.set_xlim(-35, 35)
    ax_cd.set_xlim(-35, 35)
    ax_cm.set_xlim(-35, 35)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":

    df = pd.read_csv(
        Path(PROJECT_DIR) / "data" / "vsm_input" / "corrected_polar_17.csv"
    )
    df["alpha"] = np.rad2deg(df["alpha"])

    # Select the correct columns for each DataFrame
    df_corrected = df.copy()[["alpha", "cl", "cd", "cm"]]

    df_breukels = df.copy()[["alpha", "cl_breukels", "cd_breukels", "cm_breukels"]]
    df_breukels.columns = ["alpha", "cl", "cd", "cm"]

    df_neuralfoil = df.copy()[
        ["alpha", "cl_neuralfoil", "cd_neuralfoil", "cm_neuralfoil"]
    ]
    df_neuralfoil.columns = ["alpha", "cl", "cd", "cm"]

    plot_airfoil_and_polars(
        file_path_airfoil=PROJECT_DIR
        / "processed_data"
        / "2D_correction"
        / "prof_mid_span.dat",
        df_corrected=df_corrected,
        df_neuralfoil=df_neuralfoil,
        df_flat_plate=pd.read_csv(
            PROJECT_DIR / "processed_data" / "2D_correction" / "df_flat_plate_18.csv"
        ),
        df_breukels=df_breukels,
        output_path=Path(PROJECT_DIR) / "results" / "airfoil_and_polars.pdf",
    )
