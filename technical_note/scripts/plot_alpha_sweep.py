from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_styling import (
    plot_on_ax,
    set_plot_style,
)
import matplotlib.gridspec as gridspec
from utils import (
    PROJECT_DIR,
    create_wing_aero,
    saving_pdf_and_pdf_tex,
    x_axis_labels,
    y_axis_labels,
)


def plotting_polars_alpha_correction_comparison(
    PROJECT_dir: str,
    results_dir: str,
    confidence_interval: float = 99,
):
    set_plot_style()
    polar_dir = Path(PROJECT_dir) / "processed_data" / "alpha_sweep"

    # Read in data
    data_vsm_breukels = pd.read_csv(polar_dir / "polar_data_vsm_breukels.csv")
    data_vsm_breukels_stall = pd.read_csv(
        polar_dir / "polar_data_vsm_breukels_stall.csv"
    )
    data_vsm_corrected = pd.read_csv(polar_dir / "polar_data_vsm_polar.csv")
    data_vsm_corrected_stall = pd.read_csv(polar_dir / "polar_data_vsm_polar_stall.csv")
    data_CFD_Vire2020_5e5 = pd.read_csv(
        polar_dir / "V3_CL_CD_RANS_Vire2020_Rey_50e4.csv"
    )
    data_windtunnel = pd.read_csv(
        polar_dir
        / "V3_CL_CD_CS_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
    )

    # Read in the gamma distribution data (including cd distribution)
    data_gamma = pd.read_csv(polar_dir / "gamma_distribution.csv")

    # Prepare data & styling
    data_frame_list = [
        data_vsm_breukels,
        data_vsm_breukels_stall,
        data_vsm_corrected,
        data_vsm_corrected_stall,
        data_CFD_Vire2020_5e5,
        data_windtunnel,
    ]
    labels = [
        r"VSM Breukels Re = $5.6\times10^5$",
        r"VSM Breukels Stall Re = $5.6\times10^5$",
        r"VSM Corrected Re = $5.6\times10^5$",
        r"VSM Corrected Stall Re = $5.6\times10^5$",
        r"CFD Re = $5\times10^5$",
        r"WT Re = $5.6\times10^5$",
    ]
    labels = [
        r"VSM Breukels ",
        r"VSM Breukels + Smoothening",
        r"VSM Corrected",
        r"VSM Corrected + Smoothening",
        r"CFD Re = $5\times10^5$",
        r"WT",
    ]
    colors = ["blue", "blue", "green", "green", "black", "red"]
    linestyles = ["dotted", "dashdot", "dashed", "solid", "-", "-"]
    markers = ["", "", "", "", "*", "o"]

    # ## changing the order of these lists
    # # first do CFD, then WT, then the all the VSM results
    # data_frame_list = [
    #     data_CFD_Vire2020_5e5,
    #     data_windtunnel,
    #     data_vsm_breukels,
    #     data_vsm_breukels_stall,
    #     data_vsm_corrected,
    #     data_vsm_corrected_stall,
    # ]
    # labels = [
    #     r"CFD Re = $5\times10^5$",
    #     r"WT",
    #     r"VSM Breukels ",
    #     r"VSM Breukels + Smoothening",
    #     r"VSM Corrected",
    #     r"VSM Corrected + Smoothening",
    # ]
    # colors = ["black", "red", "blue", "blue", "green", "green"]
    # linestyles = ["-", "-", "dotted", "dashdot", "dashed", "solid"]
    # markers = ["*", "o", "", "", "", ""]

    # Mapping columns in data_gamma
    gamma_col_map = {
        0: "gamma_breukels",
        1: "gamma_breukels_stall",
        2: "gamma_polar",
        3: "gamma_polar_stall",
    }
    cd_dist_col_map = {
        0: "cd_dist_breukels",
        1: "cd_dist_breukels_stall",
        2: "cd_dist_polar",
        3: "cd_dist_polar_stall",
    }

    aoa_col = "aoa"
    cl_col = "CL"
    cd_col = "CD"

    fig = plt.figure(figsize=(15, 9))
    gs_main = gridspec.GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=[0.8, 1.2],
        bottom=0.15,
        left=0.07,
        top=0.98,
        right=0.98,
    )
    # fig.subplots_adjust(bottom=0.2)
    # plt.tight_layout()

    linewidth = 1.5

    # Left column: Two vertical plots (CL vs AOA and CD vs AOA)
    gs_left = gridspec.GridSpecFromSubplotSpec(
        nrows=2,
        ncols=1,
        subplot_spec=gs_main[0],
        height_ratios=[1, 1],
        hspace=0.1,  # Space between subplots in the left column
    )
    ax_cl = fig.add_subplot(gs_left[0, 0])  # Top-left: CL vs AOA
    ax_cd = fig.add_subplot(gs_left[1, 0])  # Bottom-left: CD vs AOA

    # Right column: Three vertical plots spanning full height
    gs_right = gridspec.GridSpecFromSubplotSpec(
        nrows=3,
        ncols=1,
        subplot_spec=gs_main[1],
        height_ratios=[1, 1, 1],
        hspace=0.1,
    )
    ax_geometry = fig.add_subplot(gs_right[0, 0])  # Top-right: Geometry
    ax_gamma = fig.add_subplot(gs_right[1, 0])  # Middle-right: Gamma
    ax_cd_dist = fig.add_subplot(gs_right[2, 0])  # Bottom-right: CD distribution

    # --------------------------
    # Plotting
    # --------------------------
    for i, (df, label, color, ls, marker) in enumerate(
        zip(data_frame_list, labels, colors, linestyles, markers)
    ):
        # 1) CL vs AOA on ax_cl
        plot_on_ax(
            ax_cl,
            df[aoa_col],
            df[cl_col],
            label=label,
            color=color,
            linestyle=ls,
            marker=marker,
            is_with_grid=True,
            linewidth=linewidth,
            is_with_x_ticks=False,
            is_with_x_label=False,
        )

        # 2) CD vs AOA on ax_cd
        plot_on_ax(
            ax_cd,
            df[aoa_col],
            df[cd_col],
            label=None,
            color=color,
            linestyle=ls,
            marker=marker,
            is_with_grid=True,
            linewidth=linewidth,
        )

        # Only for the first 4 data sets, plot gamma and cd distribution
        if i < 4:
            # Gamma distribution on ax_gamma
            plot_on_ax(
                ax_gamma,
                data_gamma["y"],
                data_gamma[gamma_col_map[i]],
                label=None,
                color=color,
                linestyle=ls,
                marker=None,  # or remove if you want markers
                is_with_grid=True,
                is_with_x_ticks=False,
                is_with_x_label=False,
                linewidth=linewidth,
            )

            # CD distribution on ax_cd_dist
            plot_on_ax(
                ax_cd_dist,
                data_gamma["y"],
                data_gamma[cd_dist_col_map[i]],
                label=None,
                color=color,
                linestyle=ls,
                marker=None,
                is_with_grid=True,
                linewidth=linewidth,
            )
            if i == 0:
                plot_on_ax(
                    ax_geometry,
                    data_gamma["y"],
                    data_gamma["z"] * 6.5 * 1.287,
                    y_label="$z$ [m]",
                    # label="Geometry",
                    label=None,
                    color="black",
                    linestyle="-",
                    marker=None,
                    is_with_grid=True,
                    linewidth=linewidth,
                    is_with_x_label=False,
                    is_with_x_ticks=False,
                )
                ax_geometry.legend(
                    loc="best",
                    handles=[
                        plt.Line2D([0], [0], color="black", label="Modelled geometry")
                    ],
                ),

        # If Wind Tunnel data: add fill_between for confidence intervals
        if "WT" in label:
            alpha_fill = 0.2
            ax_cl.fill_between(
                df[aoa_col],
                df[cl_col] - df["CL_ci"],
                df[cl_col] + df["CL_ci"],
                color=color,
                alpha=alpha_fill,
                label=f"WT CI of {confidence_interval}\%",
            )
            ax_cd.fill_between(
                df[aoa_col],
                df[cd_col] - df["CD_ci"],
                df[cd_col] + df["CD_ci"],
                color=color,
                alpha=alpha_fill,
                # label=f"WT CI of {confidence_interval}\%",
            )

    # --------------------------
    # Axes formatting
    # --------------------------
    ax_cl.set_xlim(-13, 24)
    # ax_cl.set_xlabel(x_axis_labels["alpha"])
    ax_cl.set_ylabel(y_axis_labels["CL"])

    ax_cd.set_xlim(-13, 24)
    ax_cd.set_ylim(0, 0.5)
    ax_cd.set_xlabel(x_axis_labels["alpha"])
    ax_cd.set_ylabel(y_axis_labels["CD"])
    # ax_cd.legend(loc="upper left")

    # ax_gamma.set_xlabel(x_axis_labels["y/b"])
    ax_gamma.set_ylabel(y_axis_labels["gamma"] + r" ($\alpha=20^\circ$)")
    ax_gamma.set_xlim(0, 0.5)

    ax_cd_dist.set_xlabel(x_axis_labels["y/b"])
    ax_cd_dist.set_ylabel(y_axis_labels["CD"] + r" ($\alpha=20^\circ$)")
    ax_cd_dist.set_xlim(0, 0.5)
    ax_cd_dist.set_ylim(-0.3, 0.8)

    ax_geometry.set_xlim(0, 0.5)

    # fig.legend(
    #     loc="lower center",
    #     ncol=1,
    #     bbox_to_anchor=(0.3, 0.36),
    #     frameon=True,
    # )
    fig.legend(
        loc="lower center",
        ncol=4,
        # nrow=5,
        # bbox_to_anchor=(0.05, 0.96),
        frameon=True,
    )
    # plt.subplots_adjust(bottom=0.2)
    # plt.subplots_adjust(bottom=0.22)

    # Tight layout & save
    # plt.tight_layout()
    file_name = "alpha_sweep"
    saving_pdf_and_pdf_tex(results_dir, file_name)


def main():
    results_dir = Path(PROJECT_DIR) / "results"
    plotting_polars_alpha_correction_comparison(PROJECT_DIR, results_dir)


if __name__ == "__main__":
    main()
