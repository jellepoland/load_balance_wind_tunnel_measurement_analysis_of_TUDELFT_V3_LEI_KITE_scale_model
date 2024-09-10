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


def loading_data(root_dir: str) -> pd.DataFrame:
    # Read the CSV file into a DataFrame
    path_to_csv = Path(root_dir) / "processed_data" / "stats_all.csv"
    stats_all = pd.read_csv(path_to_csv)
    return stats_all


def plotting_CL_CD_CS_Pitch_Roll_Yaw_vs_alpha_reynolds_sweep(
    results_path: str,
    stats_all: pd.DataFrame,
    betas_to_be_plotted: str,
    plot_speeds: list,
    figsize: tuple,
    fontsize: int,
    columns: list,
    y_labels: list,
    subplot_titles: list,
):

    # # set which wind speeds to plot
    # plot_speeds = [5, 10, 15, 20, 25]

    if len(plot_speeds) < 5:
        subfolder = "vw="
        for i, v in enumerate(plot_speeds):
            if i == 0:
                subfolder = subfolder + f"{v}"
            else:
                subfolder = subfolder + f"+{v}"
    else:
        subfolder = "all_vw"

    # # Create a new folder to save the plots
    # os.makedirs(foldername + "/alpha", exist_ok=True)

    # sort everything for plotting correctly
    stats_plotvsalpha = stats_all.sort_values(by="aoa_kite")

    # Group the data by sideslip
    grouped = stats_plotvsalpha.groupby("sideslip")

    # Loop through each sideslip value
    for sideslip, group in grouped:

        if sideslip in betas_to_be_plotted:
            # Create a subplot with 4 rows and 3 columns
            fig, axs = plt.subplots(2, 3, figsize=figsize)

            # Flatten the subplot array for easier indexing
            axs = axs.flatten()

            # # Loop through each column (F_X, F_Y, ..., M_Z)
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
            for i, column in enumerate(columns):
                # Plot each distinct value in the vw column (excluding vw=0 and vw=5)
                for vw, vw_group in group.groupby("vw"):
                    if vw in plot_speeds:
                        Re = np.around((vw_group["Rey"].mean()) / 1e5, 1)
                        axs[i].plot(
                            vw_group["aoa_kite"],
                            vw_group[column],
                            "o-.",
                            label=rf"$Re = {Re}$ $\cdot$ $10^5$",
                        )

                axs[i].set_title(subplot_titles[i])
                axs[i].set_xlabel(r"$\alpha$ [deg]", fontsize=fontsize)
                axs[i].set_ylabel(rf"${y_labels[i]}$ [-]", fontsize=fontsize)
                if i == 0:
                    axs[i].legend()
                # axs[i].set_xlim([-5,24])
                axs[i].grid()

            # Set the title of the subplot
            # fig.suptitle(rf"Force and moment coefficient plots for sideslip angle: $\beta=${sideslip} deg")#, fontsize=14, fontweight='bold')

            # Save the plot in the plots folder
            # plot_filename = f"plots/sideslip_{sideslip}_plot.png"
            # plot_filename = foldername + '/alpha' + f"/sideslip_{sideslip}_plot.pdf"
            plot_filename = (
                Path(results_path)
                / f"re_variation_alpha_sweep_at_fixed_beta_{sideslip:.2f}.pdf"
            )
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()

            # Print a message when the plot is saved
            print(f"Plot for sideslip {sideslip} saved as {plot_filename}")


def plotting_CL_CD_CS_Pitch_Roll_Yaw_vs_beta_reynolds_sweep(
    results_path: str,
    stats_all: pd.DataFrame,
    alphas_to_be_plotted: list,
    plot_speeds: list,
    figsize: tuple,
    fontsize: int,
    columns: list,
    y_labels: list,
    subplot_titles: list,
):
    # # set which wind speeds to plot
    # plot_speeds = [5, 10, 15, 20, 25]

    if len(plot_speeds) < 5:
        subfolder = "vw="
        for i, v in enumerate(plot_speeds):
            if i == 0:
                subfolder = subfolder + f"{v}"
            else:
                subfolder = subfolder + f"+{v}"
    else:
        subfolder = "all_vw"

    # foldername = Path(root_dir) / "results"

    # sort everything for plotting correctly
    stats_plotvsbeta = stats_all.sort_values(by="sideslip")

    # Group the data by sideslip
    grouped = stats_plotvsbeta.groupby("aoa_kite")

    # Loop through each sideslip value
    for alpha, group in grouped:

        if alpha in alphas_to_be_plotted:
            # only plot if there is more than one entry per wind speed
            entries = len(group["vw"])
            if entries > 5:
                # Create a subplot with 4 rows and 3 columns
                fig, axs = plt.subplots(2, 3, figsize=figsize)

                # Flatten the subplot array for easier indexing
                axs = axs.flatten()

                # Loop through each column (F_X, F_Y, ..., M_Z)
                # columns = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
                # y_labels = ["C_D", "C_S", "C_L", "C_{roll}", "C_{pitch}", "C_{yaw}"]
                # subplot_titles = [
                #     "Drag coefficient",
                #     "Side force",
                #     "Lift coefficient",
                #     "Rolling moment coefficient",
                #     "Pitching moment coefficient",
                #     "Yawing moment coefficient",
                # ]
                for i, column in enumerate(columns):
                    # Plot each distinct value in the vw column (excluding vw=0 and vw=5)
                    for vw, vw_group in group.groupby("vw"):
                        if vw in plot_speeds:
                            Re = np.around((vw_group["Rey"].mean()) / 1e5, 1)
                            axs[i].plot(
                                vw_group["sideslip"],
                                vw_group[column],
                                "o-.",
                                label=rf"$Re = {Re}$ $\cdot$ $10^5$",
                            )

                    axs[i].set_title(subplot_titles[i])
                    axs[i].set_xlabel(r"$\beta$ [$^o$]", fontsize=fontsize)
                    axs[i].set_ylabel(rf"${y_labels[i]}$ [-]", fontsize=fontsize)
                    if i == 1:
                        axs[i].legend()
                    axs[i].set_xlim([-21, 21])
                    axs[i].grid()
                # axs[1].set_ylim([-0.04,0.75])

                # Set the title of the subplot
                # fig.suptitle(rf"Force and moment coefficient plots for angle of attack: $\alpha=${np.around(alpha,2)}$^o$")#, fontsize=14, fontweight='bold')

                # Save the plot in the plots folder
                # plot_filename = f"plots/alpha_{alpha}_plot.png"
                # plot_filename = (
                #     foldername + "/beta" + f"/alpha_{np.around(alpha,2)}_plot.pdf"
                # )
                plot_filename = (
                    Path(results_path)
                    / f"re_variation_beta_sweep_at_fixed_alpha_{alpha:.2f}.pdf"
                )
                plt.tight_layout()
                plt.savefig(plot_filename)
                plt.close()

                # Print a message when the plot is saved
                print(f"Plot for angle of attack {alpha} saved as {plot_filename}")


def main(results_path, root_dir):
    # Load the data from the CSV file
    stats_all = loading_data(root_dir)

    # Plot the data
    plot_speeds = [5, 10, 15, 20, 25]
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
    alphas_to_be_plotted = [2.35, 4.75, 6.75, 11.95, 17.85]
    # selection
    betas_to_be_plotted = [0]
    alphas_to_be_plotted = [2.35, 4.75, 6.75]

    ### Other figure settings
    plt.rcParams.update({"font.size": 14})
    figsize = (16, 12)
    fontsize = 18
    columns = ["C_L", "C_D", "C_S", "C_pitch", "C_roll", "C_yaw"]
    y_labels = ["C_L", "C_D", "C_S", "C_{M,x}", "C_{M,y}", "C_{M,z}"]
    subplot_titles = [
        "Lift coefficient",
        "Drag Coefficient",
        "Side Force coefficient",
        "Pitch moment coefficient",
        "Roll moment coefficient",
        "Yaw moment coefficient",
    ]

    plotting_CL_CD_CS_Pitch_Roll_Yaw_vs_alpha_reynolds_sweep(
        results_path,
        stats_all,
        betas_to_be_plotted,
        plot_speeds,
        figsize,
        fontsize,
        columns,
        y_labels,
        subplot_titles,
    )
    plotting_CL_CD_CS_Pitch_Roll_Yaw_vs_beta_reynolds_sweep(
        results_path,
        stats_all,
        alphas_to_be_plotted,
        plot_speeds,
        figsize,
        fontsize,
        columns,
        y_labels,
        subplot_titles,
    )


if __name__ == "__main__":
    root_dir = defining_root_dir()
    results_path = Path(root_dir) / "results"
    main(results_path, root_dir)
