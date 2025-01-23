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


def average_beta_values(df):
    """
    Averages the values of rows with positive and negative beta values.

    Args:
        df (pd.DataFrame): Input DataFrame with columns including beta values.

    Returns:
        pd.DataFrame: A new DataFrame containing averaged values for each absolute beta.
    """
    # Group by the absolute value of beta
    df["abs_beta"] = df["beta"].abs()
    grouped = df.groupby("abs_beta")

    # Average all columns except for `beta` and `abs_beta`
    averaged_df = grouped.mean().reset_index()

    # Drop the intermediate column `abs_beta` and rename `abs_beta` back to `beta`
    averaged_df = averaged_df.drop(columns=["beta"], errors="ignore")
    averaged_df = averaged_df.rename(columns={"abs_beta": "beta"})

    return averaged_df


def plotting_polars_beta_correction_comparison(
    PROJECT_dir: str,
    results_dir: str,
    confidence_interval: float = 99,
):
    set_plot_style()
    polar_dir = Path(PROJECT_dir) / "processed_data" / "beta_sweep"

    # Read in data
    data_vsm_breukels = pd.read_csv(polar_dir / "polar_data_vsm_breukels.csv")
    data_vsm_breukels_stall = pd.read_csv(
        polar_dir / "polar_data_vsm_breukels_stall.csv"
    )
    data_vsm_corrected = pd.read_csv(polar_dir / "polar_data_vsm_polar.csv")
    data_vsm_corrected_stall = pd.read_csv(polar_dir / "polar_data_vsm_polar_stall.csv")

    data_windtunnel_load = pd.read_csv(
        polar_dir
        / "V3_CL_CD_CS_beta_sweep_alpha_6.8_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_windtunnel_moment = pd.read_csv(
        polar_dir
        / "V3_CMx_CMy_CMz_beta_sweep_alpha_6.8_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_windtunnel = pd.merge(
        data_windtunnel_load, data_windtunnel_moment, on=["beta"], how="inner"
    )
    data_windtunnel = average_beta_values(data_windtunnel)

    # print(f"breukels: {data_vsm_breukels.columns}")
    # print(f"windtunnel: {data_windtunnel.columns}")

    # # Read in the gamma distribution data (including cd distribution)
    beta_dist = 20
    data_gamma = pd.read_csv(polar_dir / f"gamma_distribution_{int(beta_dist)}.csv")

    # Prepare data & styling
    data_frame_list = [
        data_vsm_breukels,
        data_vsm_breukels_stall,
        data_vsm_corrected,
        data_vsm_corrected_stall,
        data_windtunnel,
    ]
    labels = [
        r"VSM Breukels Re = $5.6\times10^5$",
        r"VSM Breukels Stall Re = $5.6\times10^5$",
        r"VSM Corrected Re = $5.6\times10^5$",
        r"VSM Corrected Stall Re = $5.6\times10^5$",
        r"WT Re = $5.6\times10^5$",
    ]
    labels = [
        r"VSM Breukels ",
        r"VSM Breukels Stall",
        r"VSM Corrected",
        r"VSM Corrected Stall",
        r"WT",
    ]
    colors = ["blue", "blue", "green", "green", "red"]
    linestyles = ["dotted", "dashdot", "dashed", "solid", "-"]
    markers = ["", "", "", "", "o"]

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
    cl_dist_col_map = {
        0: "cl_dist_breukels",
        1: "cl_dist_breukels_stall",
        2: "cl_dist_polar",
        3: "cl_dist_polar_stall",
    }

    aoa_col = "beta"
    # --------------------------
    #  1) Create the main figure
    #     1 row, 3 columns total
    # --------------------------
    # fig = plt.figure(figsize=(15, 6))
    # gs_main = gridspec.GridSpec(
    #     nrows=2, ncols=4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1]
    # )  # , wspace=0.4)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), width_ratios=[1, 1, 1, 1.6])
    # gs_main = axes
    linewidth = 1.5
    # # Left subplot for CL vs AOA
    # ax_cl = fig.add_subplot(gs_main[0, 0])
    # ax_cd = fig.add_subplot(gs_main[0, 1])
    # ax_cs = fig.add_subplot(gs_main[0, 2])
    # ax_cmx = fig.add_subplot(gs_main[1, 0])
    # ax_cmy = fig.add_subplot(gs_main[1, 1])
    # ax_cmz = fig.add_subplot(gs_main[1, 2])

    # # # ----------------------------------
    # # #  2) Nested GridSpec for last column
    # # #     Splits it into two vertical plots
    # # # ----------------------------------
    # # gs_right = gridspec.GridSpecFromSubplotSpec(
    # #     nrows=2, ncols=1, subplot_spec=gs_main[0:1, 3], height_ratios=[1, 1], hspace=0.3
    # # )
    # ax_gamma = fig.add_subplot(gs_main[0, 3])  # top half for Gamma
    # ax_cd_dist = fig.add_subplot(gs_main[1, 3])  # bottom half for CD distribution

    ax_cl = axes[0, 0]
    ax_cd = axes[0, 1]
    ax_cs = axes[0, 2]
    ax_cmx = axes[1, 0]
    ax_cmy = axes[1, 1]
    ax_cmz = axes[1, 2]
    ax_gamma = axes[0, 3]
    ax_cd_dist = axes[1, 3]

    # --------------------------
    # Plotting
    # --------------------------
    for i, (df, label, color, ls, marker) in enumerate(
        zip(data_frame_list, labels, colors, linestyles, markers)
    ):

        plot_on_ax(
            ax_cl,
            df[aoa_col],
            df["CL"],
            label=label,
            color=color,
            linestyle=ls,
            marker=marker,
            is_with_grid=True,
            linewidth=linewidth,
            is_with_x_label=False,
            is_with_x_ticks=False,
        )
        plot_on_ax(
            ax_cd,
            df[aoa_col],
            df["CD"],
            label=None,
            color=color,
            linestyle=ls,
            marker=marker,
            is_with_grid=True,
            linewidth=linewidth,
            is_with_x_label=False,
            is_with_x_ticks=False,
        )
        plot_on_ax(
            ax_cs,
            df[aoa_col],
            df["CS"],
            label=None,
            color=color,
            linestyle=ls,
            marker=marker,
            is_with_grid=True,
            linewidth=linewidth,
            is_with_x_label=False,
            is_with_x_ticks=False,
        )
        plot_on_ax(
            ax_cmx,
            df[aoa_col],
            df["CMx"],
            label=None,
            color=color,
            linestyle=ls,
            marker=marker,
            is_with_grid=True,
            linewidth=linewidth,
        )
        plot_on_ax(
            ax_cmy,
            df[aoa_col],
            df["CMy"],
            label=None,
            color=color,
            linestyle=ls,
            marker=marker,
            is_with_grid=True,
            linewidth=linewidth,
        )
        plot_on_ax(
            ax_cmz,
            df[aoa_col],
            df["CMz"],
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
            # plot_on_ax(
            #     ax_gamma,
            #     data_gamma["y"],
            #     data_gamma[gamma_col_map[i]],
            #     label=None,
            #     color=color,
            #     linestyle=ls,
            #     marker=None,  # or remove if you want markers
            #     is_with_grid=True,
            #     is_with_x_ticks=False,
            #     is_with_x_label=False,
            #     linewidth=linewidth,
            # )
            plot_on_ax(
                ax_gamma,
                data_gamma["y"],
                data_gamma[cl_dist_col_map[i]],
                is_with_x_label=False,
                is_with_x_ticks=False,
                label=None,
                color=color,
                linestyle=ls,
                marker=None,  # or remove if you want markers
                is_with_grid=True,
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

        # If Wind Tunnel data: add fill_between for confidence intervals
        if "WT" in label:
            alpha_fill = 0.2
            ax_cl.fill_between(
                df[aoa_col],
                df["CL"] - df["CL_ci"],
                df["CL"] + df["CL_ci"],
                color=color,
                alpha=alpha_fill,
                label=f"WT CI of {confidence_interval}\%",
            )
            ax_cd.fill_between(
                df[aoa_col],
                df["CD"] - df["CD_ci"],
                df["CD"] + df["CD_ci"],
                color=color,
                alpha=alpha_fill,
                # label=f"WT CI of {confidence_interval}\%",
            )
            ax_cs.fill_between(
                df[aoa_col],
                df["CS"] - df["CS_ci"],
                df["CS"] + df["CS_ci"],
                color=color,
                alpha=alpha_fill,
                # label=f"WT CI of {confidence_interval}\%",
            )
            ax_cmx.fill_between(
                df[aoa_col],
                df["CMx"] - df["CMx_ci"],
                df["CMx"] + df["CMx_ci"],
                color=color,
                alpha=alpha_fill,
                # label=f"WT CI of {confidence_interval}\%",
            )
            ax_cmy.fill_between(
                df[aoa_col],
                df["CMy"] - df["CMy_ci"],
                df["CMy"] + df["CMy_ci"],
                color=color,
                alpha=alpha_fill,
                # label=f"WT CI of {confidence_interval}\%",
            )
            ax_cmz.fill_between(
                df[aoa_col],
                df["CMz"] - df["CMz_ci"],
                df["CMz"] + df["CMz_ci"],
                color=color,
                alpha=alpha_fill,
                # label=f"WT CI of {confidence_interval}\%",
            )

    # --------------------------
    # Axes formatting
    # --------------------------
    ax_cl.set_xlim(0, 20)
    ax_cl.set_ylabel(y_axis_labels["CL"])
    ax_cd.set_xlim(0, 20)
    # ax_cd.set_ylim(0, 0.5)
    ax_cd.set_ylabel(y_axis_labels["CD"])
    ax_cs.set_xlim(0, 20)
    ax_cs.set_ylabel(y_axis_labels["CS"])
    ax_cmx.set_xlim(0, 20)
    ax_cmx.set_ylabel(y_axis_labels["CMx"])
    ax_cmx.set_xlabel(x_axis_labels["beta"])
    ax_cmy.set_xlim(0, 20)
    ax_cmy.set_ylabel(y_axis_labels["CMy"])
    ax_cmy.set_xlabel(x_axis_labels["beta"])
    ax_cmz.set_xlim(0, 20)
    ax_cmz.set_ylabel(y_axis_labels["CMz"])
    ax_cmz.set_xlabel(x_axis_labels["beta"])

    # ax_gamma.set_xlabel(x_axis_labels["y/b"])
    ax_gamma.set_ylabel(y_axis_labels["CL"] + rf" ($\beta={beta_dist}^\circ$)")
    ax_gamma.set_xlim(-0.5, 0.5)

    ax_cd_dist.set_xlabel(x_axis_labels["y/b"])
    ax_cd_dist.set_ylabel(y_axis_labels["CD"] + rf" ($\beta={beta_dist}^\circ$)")
    ax_cd_dist.set_xlim(-0.5, 0.5)

    ### Ensuring that a value does not ruin the naturally zooomed in ylim
    for i, ax in enumerate(
        [ax_cl, ax_cd, ax_cs, ax_cmx, ax_cmy, ax_cmz, ax_gamma, ax_cd_dist]
    ):
        y_min_allowed, y_max_allowed = -1.5, 1.5

        if i == 2:
            y_min_allowed, y_max_allowed = -0.35, 0.05
        elif i == 3:
            y_min_allowed, y_max_allowed = -1.5, 0.1
        elif i == 5:
            y_min_allowed, y_max_allowed = -0.25, 0.1
        elif i == 6:
            y_min_allowed, y_max_allowed = -0.4, 1.6  # 0,6 for gamma
        elif i == 7:
            y_min_allowed, y_max_allowed = -0.05, 1.2
        # Collect all y-data from the lines in the current axis
        y_data = np.concatenate([line.get_ydata() for line in ax.get_lines()])

        # Identify data within the allowed range
        in_range = y_data[(y_data >= y_min_allowed) & (y_data <= y_max_allowed)]

        if in_range.size > 0:
            # Optionally add some padding to the y-limits
            padding = 0.05 * (in_range.max() - in_range.min())
            ax.set_ylim(in_range.min() - padding, in_range.max() + padding)
        else:
            # If no data is within the range, you might choose to set default limits or skip
            pass  # Or set default limits, e.g., ax.set_ylim(y_min_allowed, y_max_allowed)

    fig.legend(
        loc="lower center",
        ncol=3,
        # nrow=5,
        # bbox_to_anchor=(0.05, 0.96),
        frameon=True,
    )
    # adjusting the vspace between row 0 and row 1
    # plt.subplots_adjust(hspace=-0.1, wspace=0.05)
    # Tight layout & save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    file_name = f"beta_sweep_{beta_dist}"
    saving_pdf_and_pdf_tex(results_dir, file_name)


def main():
    results_dir = Path(PROJECT_DIR) / "results"
    plotting_polars_beta_correction_comparison(PROJECT_DIR, results_dir)


if __name__ == "__main__":
    main()
