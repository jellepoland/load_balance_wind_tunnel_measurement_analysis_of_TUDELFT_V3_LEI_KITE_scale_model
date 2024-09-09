import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import matplotlib.cm as cm


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


def read_and_process_files(
    normal_folder_path: str, suffix: str, aoa_offset: float = 7.25
) -> pd.DataFrame:
    """
    Read and process .lvm files for a given wind speed suffix.

    Args:
    - normal_folder_path (str): Path to the main data folder.
    - suffix (str): Suffix indicating the wind speed and data type.

    Returns:
    - pd.DataFrame: Combined DataFrame of all processed data.
    """
    all_data = []  # Initialize list to store data from multiple files
    for aoa_folder in os.listdir(
        normal_folder_path
    ):  # Iterate through angle of attack folders
        # aoa_folder_path = os.path.join(normal_folder_path, aoa_folder)
        aoa_folder_path = Path(normal_folder_path) / aoa_folder
        if os.path.isdir(aoa_folder_path):  # Check if it is a directory
            for lvm_file in os.listdir(
                aoa_folder_path
            ):  # Iterate through files in the folder
                if lvm_file.startswith("processed_") and lvm_file.endswith(
                    suffix
                ):  # Check file name pattern
                    # lvm_file_path = os.path.join(aoa_folder_path, lvm_file)
                    lvm_file_path = Path(aoa_folder_path) / lvm_file
                    df = pd.read_csv(
                        lvm_file_path, delimiter="\t"
                    )  # Read file into DataFrame
                    df["aoa"] -= aoa_offset  # Adjust angle of attack values
                    # print(f'aoa with offset substracted: {df["aoa"].unique()}')
                    all_data.append(df)  # Append to list
    return pd.concat(all_data)  # Concatenate all DataFrames


def generate_boxplots_vs_aoa(
    combined_df: pd.DataFrame,
    vw: str,
    sideslip: float,
    results_path: str,
    figsize,
    fontsize,
    columns,
    y_labels,
    subplot_titles,
) -> None:
    """
    Generate and save boxplots for different load types versus angle of attack.

    Args:
    - combined_df (pd.DataFrame): DataFrame containing all processed data.
    - vw (str): Wind speed value as a string.
    - sideslip (int): Sideslip angle.
    - titlespeeds (list): List of wind speed titles for labeling plots.
    - j (int): Index of the current wind speed in the list.
    """
    # all
    # columns = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
    # y_labels = ["C_D", "C_S", "C_L", "C_{roll}", "C_{pitch}", "C_{yaw}"]
    # subplot_titles = [
    #     "Drag coefficient",
    #     "Side force coefficient",
    #     "Lift coefficient",
    #     "Rolling moment coefficient",
    #     "Pitching moment coefficient",
    #     "Yawing moment coefficient",
    # ]
    if len(columns) == 6:
        fig, axs = plt.subplots(2, 3, figsize=figsize)
    # # selection
    # columns = ["C_L", "C_D", "C_pitch"]
    # y_labels = ["C_L", "C_D", "C_{pitch}"]
    # subplot_titles = [
    #     "Lift coefficient",
    #     "Drag coefficient",
    #     "Pitch moment coefficient",
    # ]
    elif len(columns) == 3:
        fig, axs = plt.subplots(1, 3, figsize=figsize)

    cmap = plt.get_cmap("rainbow")
    colors = cmap(np.linspace(0, 1, len(columns)))

    axs = axs.flatten()

    for i, load in enumerate(columns):
        unique_aoa = sorted(combined_df["aoa"].unique())
        box_data = [combined_df[combined_df["aoa"] == aoa][load] for aoa in unique_aoa]
        positions = unique_aoa

        bplot = axs[i].boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )
        for patch, color in zip(
            bplot["boxes"], cmap(np.linspace(0, 1, len(unique_aoa)))
        ):
            patch.set_facecolor(color)

        means = [data.mean() for data in box_data]
        axs[i].plot(positions, means, marker="o", linestyle="--", color="black")
        axs[i].set_title(f"{subplot_titles[i]}")
        axs[i].set_ylabel(rf"${y_labels[i]}$ [-]", fontsize=fontsize)
        axs[i].set_xticks(range(-15, 26, 5))
        axs[i].set_xticklabels(range(-15, 26, 5))
        axs[i].set_xlim(-15, 25)
        axs[i].grid()

    for ax in axs:
        ax.set_xlabel(r"$\alpha$ [deg]", fontsize=fontsize)

    plt.tight_layout()
    # output_dir = f"plots_unsteady_final/vw_{vw}/alpha"
    # os.makedirs(output_dir, exist_ok=True)
    plot_filename = (
        Path(results_path)
        / f"boxplot_alpha_sweep_at_fixed_beta_{sideslip:.2f}_vw_{vw}.pdf"
    )
    plt.savefig(plot_filename)
    plt.close()


def generate_boxplots_vs_sideslip(
    combined_df: pd.DataFrame,
    vw: str,
    aoa: float,
    results_path: str,
    figsize,
    fontsize,
    columns,
    y_labels,
    subplot_titles,
) -> None:
    """
    Generate and save boxplots for different load types versus sideslip angle.

    Args:
    - combined_df (pd.DataFrame): DataFrame containing all processed data.
    - vw (str): Wind speed value as a string.
    - aoa (float): Angle of attack.
    - titlespeeds (list): List of wind speed titles for labeling plots.
    - j (int): Index of the current wind speed in the list.
    """
    columns = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
    y_labels = ["C_D", "C_S", "C_L", "C_{roll}", "C_{pitch}", "C_{yaw}"]
    subplot_titles = [
        "Drag coefficient",
        "Side force coefficient",
        "Lift coefficient",
        "Rolling moment coefficient",
        "Pitching moment coefficient",
        "Yawing moment coefficient",
    ]

    unique_sideslip = sorted(combined_df["sideslip"].unique())

    if (
        len(unique_sideslip) > 1
    ):  # Only create plots if there is more than one unique sideslip value

        cmap = plt.get_cmap("rainbow")
        colors = cmap(np.linspace(0, 1, len(columns)))

        fig, axs = plt.subplots(2, 3, figsize=figsize)
        axs = axs.flatten()

        for i, load in enumerate(columns):
            box_data = [
                combined_df[combined_df["sideslip"] == sideslip][load]
                for sideslip in unique_sideslip
            ]
            positions = unique_sideslip

            bplot = axs[i].boxplot(
                box_data,
                positions=positions,
                widths=0.6,
                showfliers=False,
                patch_artist=True,
            )
            for patch, color in zip(
                bplot["boxes"], cmap(np.linspace(0, 1, len(unique_sideslip)))
            ):
                patch.set_facecolor(color)

            means = [data.mean() for data in box_data]
            axs[i].plot(positions, means, marker="o", linestyle="--", color="black")
            axs[i].set_title(f"{subplot_titles[i]}")
            axs[i].set_ylabel(rf"${y_labels[i]}$ [-]", fontsize=fontsize)
            axs[i].set_xticks(
                np.arange(
                    np.floor(min(unique_sideslip)), np.ceil(max(unique_sideslip)) + 1, 4
                )
            )
            axs[i].set_xticklabels(
                np.arange(
                    np.floor(min(unique_sideslip)), np.ceil(max(unique_sideslip)) + 1, 4
                ).astype(int)
            )
            axs[i].grid()

        for ax in axs:
            ax.set_xlabel(r"$\beta$ [deg]", fontsize=fontsize)

        plt.tight_layout()
        # output_dir = f"plots_unsteady_final/vw_{vw}/beta"
        # os.makedirs(output_dir, exist_ok=True)
        plot_filename = (
            Path(results_path)
            / f"boxplot_beta_sweep_at_fixed_aoa_{np.around(aoa,2)}_vw_{vw}.pdf"
        )
        plt.savefig(plot_filename)
        plt.close()


def main(results_path, root_dir):

    # Define folder path, wind speeds, sideslip angles, and titles
    normal_folder_path = Path(root_dir) / "Mark_van_Spronsen" / "normal"
    # all
    plot_speeds = ["05", "10", "15", "20", "25"]
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

    # selection
    plot_speeds = ["15", "20"]
    betas_to_be_plotted = [0]
    alphas_to_be_plotted = [2.35, 4.75, 6.75]

    ### Other figure settings
    plt.rcParams.update({"font.size": 14})
    figsize = (16, 12)
    fontsize = 18

    # Loop through each wind speed
    for j, vw in enumerate(plot_speeds):
        suffix = (
            vw + "_unsteady.lvm"
        )  # Construct file suffix for the current wind speed
        combined_df = read_and_process_files(
            normal_folder_path, suffix
        )  # Read and process files

        print(f"analyzing vw: {plot_speeds[j]}")

        for sideslip in betas_to_be_plotted:
            # columns = ["C_L", "C_D", "C_S", "C_pitch", "C_roll", "C_yaw"]
            # y_labels = ["C_L", "C_D", "C_S", "C_{pitch}", "C_{roll}", "C_{yaw}"]
            # subplot_titles = [
            #     "Lift coefficient",
            #     "Drag Coefficient",
            #     "Side Force coefficient",
            #     "Pitch moment coefficient",
            #     "Roll moment coefficient",
            #     "Yaw moment coefficient",
            # ]
            figsize = (16, 6)
            columns = ["C_L", "C_D", "C_pitch"]
            y_labels = ["C_L", "C_D", "C_{pitch}"]
            subplot_titles = [
                "Lift coefficient",
                "Drag Coefficient",
                "Pitch moment coefficient",
            ]

            generate_boxplots_vs_aoa(
                combined_df[combined_df["sideslip"] == sideslip],
                vw,
                sideslip,
                results_path,
                figsize,
                fontsize,
                columns,
                y_labels,
                subplot_titles,
            )  # Generate and save boxplots vs AoA

        unique_aoas = combined_df["aoa"].unique()
        for aoa in unique_aoas:
            if aoa in alphas_to_be_plotted:
                figsize = (16, 12)
                columns = ["C_L", "C_D", "C_S", "C_pitch", "C_roll", "C_yaw"]
                y_labels = ["C_L", "C_D", "C_S", "C_{pitch}", "C_{roll}", "C_{yaw}"]
                subplot_titles = [
                    "Lift coefficient",
                    "Drag Coefficient",
                    "Side Force coefficient",
                    "Pitch moment coefficient",
                    "Roll moment coefficient",
                    "Yaw moment coefficient",
                ]
                generate_boxplots_vs_sideslip(
                    combined_df[combined_df["aoa"] == aoa],
                    vw,
                    aoa,
                    results_path,
                    figsize,
                    fontsize,
                    columns,
                    y_labels,
                    subplot_titles,
                )  # Generate and save boxplots vs sideslip


if __name__ == "__main__":
    root_dir = defining_root_dir()
    results_path = Path(root_dir) / "results"
    main(results_path, root_dir)
