import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import (
    project_dir,
    saving_pdf_and_pdf_tex,
    x_axis_labels,
    y_axis_labels,
    reduce_df_by_parameter_mean_and_std,
)
from load_balance_analysis.functions_statistics import (
    hac_newey_west_confidence_interval,
)
from matplotlib.patches import Patch
from plot_styling import plot_on_ax


def raw_df_to_mean_and_ci(
    df: pd.DataFrame,
    variable_col: str,
    coefficients: list,
    confidence_interval: float,
    max_lag: int,
) -> pd.DataFrame:

    # Initialize an empty list to hold each unique group's results
    all_data_calculated = []

    # Loop through each unique value in variable_col
    for col in df[variable_col].unique():
        data = df[df[variable_col] == col]

        # Initialize a dictionary for the current group's results
        data_calculated = {}

        # HAC Newey-West Confidence Interval calculation for each coefficient
        for coeff in coefficients:
            mean_value = data[coeff].mean()
            ci = hac_newey_west_confidence_interval(
                data[coeff], confidence_interval=confidence_interval, max_lag=max_lag
            )

            # Create entries in the dictionary with dynamic column names
            data_calculated[f"{coeff}_mean"] = mean_value
            data_calculated[f"{coeff}_ci"] = ci

        # Add the variable_col value to the dictionary
        data_calculated[variable_col] = col

        # Append the current group's results to the list
        all_data_calculated.append(data_calculated)

    # Transform the list of dictionaries into a DataFrame
    data_calculated_df = pd.DataFrame(all_data_calculated)

    # sorting the df by the variable col
    data_calculated_df = data_calculated_df.sort_values(by=variable_col)

    return data_calculated_df


def saving_alpha_and_beta_sweeps(
    project_dir: str,
    confidence_interval: float,
    max_lag: int,
    name_appendix: str = "",
):
    """
    Reads wind tunnel CSV data for alpha sweeps (beta=0, multiple velocities)
    and beta sweeps (alpha ~ 6.8 or 11.9 deg), computes mean + confidence intervals,
    and saves out separate files:
      - A file with CL, CD, (CS) vs alpha or beta.
      - A file with CMx, CMy, CMz vs alpha or beta.
    """

    ### Part 1) ALPHA sweeps
    vw_values = [5, 20]
    beta_value = 0
    folders_and_files = [(f"beta_{beta_value}", f"vw_{vw}.csv") for vw in vw_values]
    data_windtunnel = []

    for folder_name, file_name in folders_and_files:
        path_to_csv = (
            Path(project_dir)
            / "processed_data"
            / "normal_csv"
            / folder_name
            / file_name
        )
        df_all_values = pd.read_csv(path_to_csv)

        # A) Forces: C_L, C_D
        df_forces = raw_df_to_mean_and_ci(
            df_all_values,
            variable_col="aoa_kite",
            coefficients=["C_L", "C_D"],
            confidence_interval=confidence_interval,
            max_lag=max_lag,
        )
        # df_forces has columns: ["C_L_mean","C_L_ci","C_D_mean","C_D_ci","aoa_kite"]

        # B) Moments: C_roll, C_pitch, C_yaw
        df_moments = raw_df_to_mean_and_ci(
            df_all_values,
            variable_col="aoa_kite",
            coefficients=["C_roll", "C_pitch", "C_yaw"],
            confidence_interval=confidence_interval,
            max_lag=max_lag,
        )
        # df_moments has columns: ["C_roll_mean","C_roll_ci","C_pitch_mean","C_pitch_ci","C_yaw_mean","C_yaw_ci","aoa_kite"]

        # Rename them so the user sees CM_x_mean, etc.
        df_moments.columns = [
            "CM_x_mean",
            "CM_x_ci",
            "CM_y_mean",
            "CM_y_ci",
            "CM_z_mean",
            "CM_z_ci",
            "aoa_kite",
        ]

        # Store this tuple (forces, moments)
        data_windtunnel.append((df_forces, df_moments))

    # We only use the second one (index=1) in your code
    df_forces_alpha = data_windtunnel[1][0]
    df_moments_alpha = data_windtunnel[1][1]

    # df_forces_alpha => rename columns to [CL, CL_ci, CD, CD_ci, aoa]
    # ( currently = ["C_L_mean","C_L_ci","C_D_mean","C_D_ci","aoa_kite"] )
    col_names_forces = ["CL", "CL_ci", "CD", "CD_ci", "aoa"]
    df_forces_alpha.columns = col_names_forces

    # df_moments_alpha => rename columns to [CMx, CMx_ci, CMy, CMy_ci, CMz, CMz_ci, aoa]
    # ( currently = ["CM_x_mean","CM_x_ci","CM_y_mean","CM_y_ci","CM_z_mean","CM_z_ci","aoa_kite"] )
    col_names_moments = ["CMx", "CMx_ci", "CMy", "CMy_ci", "CMz", "CMz_ci", "aoa"]
    df_moments_alpha.columns = col_names_moments

    # Save them out
    df_forces_alpha.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CL_CD_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )
    df_moments_alpha.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CMx_CMy_CMz_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )

    ### Part 2) BETA sweeps
    vw = 20
    alpha_high = 11.9
    alpha_low = 6.8

    # -- alpha_high
    folder_name = f"alpha_{alpha_high}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(project_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)

    # A) Force coefficients: [C_L, C_D, C_S]
    data_WT_beta_high_forces = raw_df_to_mean_and_ci(
        df_all_values,
        variable_col="sideslip",
        coefficients=["C_L", "C_D", "C_S"],
        confidence_interval=confidence_interval,
        max_lag=max_lag,
    )
    # B) Moment coefficients: [C_Mx, C_My, C_Mz]
    data_WT_beta_high_moments = raw_df_to_mean_and_ci(
        df_all_values,
        variable_col="sideslip",
        coefficients=["C_roll", "C_pitch", "C_yaw"],
        confidence_interval=confidence_interval,
        max_lag=max_lag,
    )

    # data_WT_beta_high_forces => rename columns
    col_names_forces_beta = ["CL", "CL_ci", "CD", "CD_ci", "CS", "CS_ci", "beta"]
    data_WT_beta_high_forces.columns = col_names_forces_beta

    # data_WT_beta_high_moments => rename columns
    col_names_moments_beta = ["CMx", "CMx_ci", "CMy", "CMy_ci", "CMz", "CMz_ci", "beta"]
    data_WT_beta_high_moments.columns = col_names_moments_beta

    # Save them
    data_WT_beta_high_forces.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CL_CD_CS_beta_sweep_alpha_{alpha_high}_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )
    data_WT_beta_high_moments.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CMx_CMy_CMz_beta_sweep_alpha_{alpha_high}_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )

    # -- alpha_low
    folder_name = f"alpha_{alpha_low}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(project_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)

    data_WT_beta_low_forces = raw_df_to_mean_and_ci(
        df_all_values,
        variable_col="sideslip",
        coefficients=["C_L", "C_D", "C_S"],
        confidence_interval=confidence_interval,
        max_lag=max_lag,
    )
    data_WT_beta_low_moments = raw_df_to_mean_and_ci(
        df_all_values,
        variable_col="sideslip",
        coefficients=["C_roll", "C_pitch", "C_yaw"],
        confidence_interval=confidence_interval,
        max_lag=max_lag,
    )

    data_WT_beta_low_forces.columns = col_names_forces_beta
    data_WT_beta_low_forces.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CL_CD_CS_beta_sweep_alpha_{alpha_low}_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )

    data_WT_beta_low_moments.columns = col_names_moments_beta
    data_WT_beta_low_moments.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CMx_CMy_CMz_beta_sweep_alpha_{alpha_low}_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )


def plotting_polars_alpha(
    project_dir: str,
    results_dir: str,
    confidence_interval: float,
):

    # VSM and Lebesque data
    path_to_csv_VSM_alpha_re_56e4 = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"VSM_results_alpha_sweep_Rey_5.6_corrected_stall.csv"
    )
    data_VSM_alpha_re_56e4 = pd.read_csv(path_to_csv_VSM_alpha_re_56e4)

    data_CFD_Vire2020_5e5 = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_RANS_Vire2020_Rey_50e4.csv"
    )
    data_CFD_Vire2021_10e5 = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv"
    )

    # Get wind tunnel data
    path_to_csv = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_windtunnel = pd.read_csv(path_to_csv)
    # Data frames, labels, colors, and linestyles
    data_frame_list = [
        data_CFD_Vire2020_5e5,
        data_CFD_Vire2021_10e5,
        data_VSM_alpha_re_56e4,
        data_windtunnel,
    ]
    labels = [
        rf"CFD Re = $5\times10^5$ (No struts)",
        rf"CFD Re = $10\times10^5$",
        rf"VSM Re = $5.6\times10^5$",
        rf"WT Re = $5.6\times10^5$",
        # rf"Polars Uri",
    ]
    colors = ["black", "black", "blue", "red"]
    linestyles = ["--", "-", "-", "-"]
    markers = ["*", "*", "s", "o"]
    fmt_wt = "o"

    # Plot CL, CD, and CL/CD in subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    aoa_col = f"aoa"
    cl_col = f"CL"
    cd_col = f"CD"

    for i, (data_frame, label, color, linestyle, marker) in enumerate(
        zip(data_frame_list, labels, colors, linestyles, markers)
    ):
        plot_on_ax(
            axs[0],
            data_frame[aoa_col],
            data_frame[cl_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        plot_on_ax(
            axs[1],
            data_frame[aoa_col],
            data_frame[cd_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        plot_on_ax(
            axs[2],
            data_frame[aoa_col],
            data_frame[cl_col] / data_frame[cd_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        # adding the confidence interval
        if i == 3:
            alpha = 0.2
            axs[0].fill_between(
                data_frame[aoa_col],
                data_frame[cl_col] - data_frame["CL_ci"],
                data_frame[cl_col] + data_frame["CL_ci"],
                color=color,
                alpha=alpha,
                label=f"WT CI of {confidence_interval}\%",
            )
            axs[1].fill_between(
                data_frame[aoa_col],
                data_frame[cd_col] - data_frame["CD_ci"],
                data_frame[cd_col] + data_frame["CD_ci"],
                color=color,
                alpha=alpha,
                label=f"WT CI of {confidence_interval}\%",
            )
            axs[2].fill_between(
                data_frame[aoa_col],
                data_frame[cl_col] / data_frame[cd_col]
                - data_frame["CL_ci"] / data_frame["CD_ci"],
                data_frame[cl_col] / data_frame[cd_col]
                + data_frame["CL_ci"] / data_frame["CD_ci"],
                color=color,
                alpha=alpha,
                label=f"WT CI of {confidence_interval}\%",
            )

    axs[0].set_xlim(-13, 24)
    axs[1].set_xlim(-13, 24)
    axs[2].set_xlim(-13, 24)

    # Format axes
    axs[0].set_xlabel(x_axis_labels["alpha"])
    axs[0].set_ylabel(y_axis_labels["CL"])
    # axs[0].grid()

    axs[1].set_xlabel(x_axis_labels["alpha"])
    axs[1].set_ylabel(y_axis_labels["CD"])
    axs[1].legend(loc="upper left")
    # axs[1].grid()

    axs[2].set_xlabel(x_axis_labels["alpha"])
    axs[2].set_ylabel(y_axis_labels["L/D"])
    # axs[2].grid()

    # Save the plot
    plt.tight_layout()
    file_name = f"literature_polars_alpha"
    saving_pdf_and_pdf_tex(results_dir, file_name)


def plotting_polars_alpha_correction_comparison(
    project_dir: str,
    results_dir: str,
    confidence_interval: float,
):
    polar_dir = Path(project_dir) / "processed_data" / "polar_data"
    # VSM data
    data_VSM_alpha_re_56e4_breukels = pd.read_csv(
        Path(polar_dir) / f"VSM_results_alpha_sweep_Rey_5.6_breukels.csv"
    )
    data_VSM_alpha_re_56e4_breukels_stall = pd.read_csv(
        Path(polar_dir) / f"VSM_results_alpha_sweep_Rey_5.6_breukels_stall.csv"
    )
    data_VSM_alpha_re_56e4_corrected = pd.read_csv(
        Path(polar_dir) / f"VSM_results_alpha_sweep_Rey_5.6_corrected.csv"
    )
    data_VSM_alpha_re_56e4_corrected_stall = pd.read_csv(
        Path(polar_dir) / f"VSM_results_alpha_sweep_Rey_5.6_corrected_stall.csv"
    )

    # Get wind tunnel data
    path_to_csv = (
        Path(polar_dir)
        / "V3_CL_CD_CS_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_windtunnel = pd.read_csv(path_to_csv)

    data_CFD_Vire2020_5e5 = pd.read_csv(
        Path(polar_dir) / "V3_CL_CD_RANS_Vire2020_Rey_50e4.csv"
    )
    data_CFD_Vire2021_10e5 = pd.read_csv(
        Path(polar_dir) / "V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv"
    )

    # Data frames, labels, colors, and linestyles
    data_frame_list = [
        data_CFD_Vire2020_5e5,
        data_CFD_Vire2021_10e5,
        data_VSM_alpha_re_56e4_breukels,
        data_VSM_alpha_re_56e4_breukels_stall,
        data_VSM_alpha_re_56e4_corrected,
        data_VSM_alpha_re_56e4_corrected_stall,
        data_windtunnel,
    ]
    labels = [
        rf"CFD Re = $5\times10^5$ (No struts)",
        rf"CFD Re = $10\times10^5$",
        rf"VSM Breukels Re = $5.6\times10^5$",
        rf"VSM Breukels Stall Re = $5.6\times10^5$",
        rf"VSM Corrected Re = $5.6\times10^5$",
        rf"VSM Corrected Stall Re = $5.6\times10^5$",
        rf"WT Re = $5.6\times10^5$",
    ]
    colors = ["black", "black", "blue", "blue", "green", "green", "red"]
    linestyles = ["--", "-", "--", "-", "--", "-", "-"]
    markers = ["*", "*", "s", "s", "+", "+", "o"]
    fmt_wt = "o"

    # Plot CL, CD, and CL/CD in subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    aoa_col = f"aoa"
    cl_col = f"CL"
    cd_col = f"CD"

    for i, (data_frame, label, color, linestyle, marker) in enumerate(
        zip(data_frame_list, labels, colors, linestyles, markers)
    ):
        plot_on_ax(
            axs[0],
            data_frame[aoa_col],
            data_frame[cl_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        plot_on_ax(
            axs[1],
            data_frame[aoa_col],
            data_frame[cd_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        plot_on_ax(
            axs[2],
            data_frame[aoa_col],
            data_frame[cl_col] / data_frame[cd_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        # adding the confidence interval
        if "WT" in label:
            alpha = 0.2
            axs[0].fill_between(
                data_frame[aoa_col],
                data_frame[cl_col] - data_frame["CL_ci"],
                data_frame[cl_col] + data_frame["CL_ci"],
                color=color,
                alpha=alpha,
                label=f"WT CI of {confidence_interval}\%",
            )
            axs[1].fill_between(
                data_frame[aoa_col],
                data_frame[cd_col] - data_frame["CD_ci"],
                data_frame[cd_col] + data_frame["CD_ci"],
                color=color,
                alpha=alpha,
                label=f"WT CI of {confidence_interval}\%",
            )
            axs[2].fill_between(
                data_frame[aoa_col],
                data_frame[cl_col] / data_frame[cd_col]
                - data_frame["CL_ci"] / data_frame["CD_ci"],
                data_frame[cl_col] / data_frame[cd_col]
                + data_frame["CL_ci"] / data_frame["CD_ci"],
                color=color,
                alpha=alpha,
                label=f"WT CI of {confidence_interval}\%",
            )

    axs[0].set_xlim(-13, 24)
    axs[1].set_xlim(-13, 24)
    axs[2].set_xlim(-13, 24)

    # Format axes
    axs[0].set_xlabel(x_axis_labels["alpha"])
    axs[0].set_ylabel(y_axis_labels["CL"])
    # axs[0].grid()

    axs[1].set_xlabel(x_axis_labels["alpha"])
    axs[1].set_ylabel(y_axis_labels["CD"])
    axs[1].legend(loc="upper left")
    # axs[1].grid()
    axs[1].set_ylim(0, 0.6)

    axs[2].set_xlabel(x_axis_labels["alpha"])
    axs[2].set_ylabel(y_axis_labels["L/D"])
    # axs[2].grid()

    # Save the plot
    plt.tight_layout()
    file_name = f"literature_polars_alpha_correction_and_stall_effects"
    saving_pdf_and_pdf_tex(results_dir, file_name)


def plotting_polars_alpha_moments_correction_comparison(
    project_dir: str,
    results_dir: str,
    confidence_interval: float,
):
    """
    Plots CMx, CMy, CMz as a function of alpha, comparing:
      - VSM no-correction
      - VSM corrected
      - Wind tunnel data
    for Re ~ 5.6e5, alpha sweep at beta=0.

    Filenames assume you have:
      VSM_results_alpha_sweep_Re_5.6_no_correction_moment.csv
      VSM_results_alpha_sweep_Re_5.6_moment.csv
      V3_CMx_CMy_CMz_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv
    or similar in your 'polar_data' folder.
    """
    polar_dir = Path(project_dir) / "processed_data" / "polar_data"

    # 1) Load VSM
    data_VSM_alpha_re_56e4_breukels_moment = pd.read_csv(
        Path(polar_dir) / "VSM_results_alpha_sweep_Rey_5.6_breukels_moment.csv"
    )
    data_VSM_alpha_re_56e4_breukels_stall_moment = pd.read_csv(
        Path(polar_dir) / "VSM_results_alpha_sweep_Rey_5.6_breukels_stall_moment.csv"
    )
    data_VSM_alpha_re_56e4_corrected_moment = pd.read_csv(
        Path(polar_dir) / "VSM_results_alpha_sweep_Rey_5.6_corrected_moment.csv"
    )
    data_VSM_alpha_re_56e4_corrected_stall_moment = pd.read_csv(
        Path(polar_dir) / "VSM_results_alpha_sweep_Rey_5.6_corrected_stall_moment.csv"
    )

    # 3) Load wind tunnel (WT) moment data
    path_to_csv_WT_moment = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CMx_CMy_CMz_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_WT_alpha_moment = pd.read_csv(path_to_csv_WT_moment)

    # Put them in a list for easy iteration
    data_frame_list = [
        data_VSM_alpha_re_56e4_breukels_moment,
        data_VSM_alpha_re_56e4_breukels_stall_moment,
        data_VSM_alpha_re_56e4_corrected_moment,
        data_VSM_alpha_re_56e4_corrected_stall_moment,
        data_WT_alpha_moment,
    ]
    labels = [
        r"VSM Breukels $\mathrm{Re} = 5.6\times10^5$",
        r"VSM Breukels Stall $\mathrm{Re} = 5.6\times10^5$",
        r"VSM Corrected $\mathrm{Re} = 5.6\times10^5$",
        r"VSM Corrected Stall$\mathrm{Re} = 5.6\times10^5$",
        r"WT $\mathrm{Re} = 5.6\times10^5$",
    ]
    colors = ["blue", "blue", "green", "green", "red"]
    linestyles = ["--", "-", "--", "-", "-"]
    markers = ["s", "s", "+", "+", "o"]

    # 4) Create subplots: CMx, CMy, CMz
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Columns used in each DataFrame
    alpha_col = "aoa"
    cmx_col = "CMx"
    cmy_col = "CMy"
    cmz_col = "CMz"

    # Also, the columns that store the confidence intervals (for the WT data)
    # e.g. "CMx_ci", "CMy_ci", "CMz_ci"
    cmx_ci_col = "CMx_ci"
    cmy_ci_col = "CMy_ci"
    cmz_ci_col = "CMz_ci"

    # 5) Plot each dataset
    for i, (data_frame, label, color, linestyle, marker) in enumerate(
        zip(data_frame_list, labels, colors, linestyles, markers)
    ):
        # Plot CMx vs alpha
        plot_on_ax(
            axs[0],
            data_frame[alpha_col],
            data_frame[cmx_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        # Plot CMy vs alpha
        plot_on_ax(
            axs[1],
            data_frame[alpha_col],
            data_frame[cmy_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        # Plot CMz vs alpha
        plot_on_ax(
            axs[2],
            data_frame[alpha_col],
            data_frame[cmz_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )

        # If this is the WT dataset (i==2), fill confidence intervals
        if "WT" in label and cmx_ci_col in data_frame.columns:
            alpha_shade = 0.2
            # CMx
            axs[0].fill_between(
                data_frame[alpha_col],
                data_frame[cmx_col] - data_frame[cmx_ci_col],
                data_frame[cmx_col] + data_frame[cmx_ci_col],
                color=color,
                alpha=alpha_shade,
                label=f"WT CI of {confidence_interval}%",
            )
            # CMy
            axs[1].fill_between(
                data_frame[alpha_col],
                data_frame[cmy_col] - data_frame[cmy_ci_col],
                data_frame[cmy_col] + data_frame[cmy_ci_col],
                color=color,
                alpha=alpha_shade,
                label=f"WT CI of {confidence_interval}%",
            )
            # CMz
            axs[2].fill_between(
                data_frame[alpha_col],
                data_frame[cmz_col] - data_frame[cmz_ci_col],
                data_frame[cmz_col] + data_frame[cmz_ci_col],
                color=color,
                alpha=alpha_shade,
                label=f"WT CI of {confidence_interval}%",
            )

    # 6) Format axes, add labels
    # X-limits (optional)
    axs[0].set_xlim(-13, 24)
    axs[1].set_xlim(-13, 24)
    axs[2].set_xlim(-13, 24)

    # X-axis
    axs[0].set_xlabel(x_axis_labels["alpha"])
    axs[1].set_xlabel(x_axis_labels["alpha"])
    axs[2].set_xlabel(x_axis_labels["alpha"])

    # Y-axis
    axs[0].set_ylabel("CMx")
    axs[1].set_ylabel("CMy")
    axs[2].set_ylabel("CMz")

    # Legend on the second subplot (or whichever you like)
    axs[0].legend(loc="upper left")

    # 7) Save the figure
    plt.tight_layout()
    file_name = "literature_polars_alpha_moments_correction_and_stall_effects"
    saving_pdf_and_pdf_tex(results_dir, file_name)


def plotting_polars_alpha_moments_vsm_wt(
    project_dir: str,
    results_dir: str,
    confidence_interval: float,
):
    """
    Plots CMx, CMy, CMz as a function of alpha, comparing:
      - VSM no-correction
      - Wind tunnel data
    for Re ~ 5.6e5, alpha sweep at beta=0.

    Filenames assume you have:
      VSM_results_alpha_sweep_Re_5.6_no_correction_moment.csv
      V3_CMx_CMy_CMz_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv
    or similar in your 'polar_data' folder.
    """

    # 1) Load VSM (no correction) moment data
    path_to_csv_VSM_alpha_re_56e4_no_correction_moment = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "VSM_results_alpha_sweep_Rey_5.6_no_correction_moment.csv"
    )
    data_VSM_alpha_re_56e4_no_correction_moment = pd.read_csv(
        path_to_csv_VSM_alpha_re_56e4_no_correction_moment
    )

    # 2) Load wind tunnel (WT) moment data
    path_to_csv_WT_moment = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CMx_CMy_CMz_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_WT_alpha_moment = pd.read_csv(path_to_csv_WT_moment)

    # Put them in a list for easy iteration
    data_frame_list = [
        data_VSM_alpha_re_56e4_no_correction_moment,
        data_WT_alpha_moment,
    ]
    labels = [
        r"VSM Re = $5.6\times10^5$ no correction",
        r"WT Re = $5.6\times10^5$",
    ]
    colors = ["blue", "red"]
    linestyles = ["-", "-"]
    markers = ["s", "o"]

    # 3) Create subplots: CMx, CMy, CMz
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Columns used in each DataFrame
    alpha_col = "aoa"
    cmx_col = "CMx"
    cmy_col = "CMy"
    cmz_col = "CMz"

    # Also, the columns that store the confidence intervals (for the WT data)
    # e.g. "CMx_ci", "CMy_ci", "CMz_ci"
    cmx_ci_col = "CMx_ci"
    cmy_ci_col = "CMy_ci"
    cmz_ci_col = "CMz_ci"

    # 4) Plot each dataset
    for i, (data_frame, label, color, linestyle, marker) in enumerate(
        zip(data_frame_list, labels, colors, linestyles, markers)
    ):
        # Plot CMx vs alpha
        axs[0].plot(
            data_frame[alpha_col],
            data_frame[cmx_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            linewidth=1.5,
        )
        # Plot CMy vs alpha
        axs[1].plot(
            data_frame[alpha_col],
            data_frame[cmy_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            linewidth=1.5,
        )
        # Plot CMz vs alpha
        axs[2].plot(
            data_frame[alpha_col],
            data_frame[cmz_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            linewidth=1.5,
        )

        # If this is the WT dataset (i==1), fill confidence intervals
        if i == 1 and all(
            col in data_frame.columns for col in [cmx_ci_col, cmy_ci_col, cmz_ci_col]
        ):
            alpha_shade = 0.2
            # CMx
            axs[0].fill_between(
                data_frame[alpha_col],
                data_frame[cmx_col] - data_frame[cmx_ci_col],
                data_frame[cmx_col] + data_frame[cmx_ci_col],
                color=color,
                alpha=alpha_shade,
                label=f"WT CI of {confidence_interval}%",
            )
            # CMy
            axs[1].fill_between(
                data_frame[alpha_col],
                data_frame[cmy_col] - data_frame[cmy_ci_col],
                data_frame[cmy_col] + data_frame[cmy_ci_col],
                color=color,
                alpha=alpha_shade,
                label=f"WT CI of {confidence_interval}%",
            )
            # CMz
            axs[2].fill_between(
                data_frame[alpha_col],
                data_frame[cmz_col] - data_frame[cmz_ci_col],
                data_frame[cmz_col] + data_frame[cmz_ci_col],
                color=color,
                alpha=alpha_shade,
                label=f"WT CI of {confidence_interval}%",
            )

    # 5) Format axes, add labels
    # X-limits (optional)
    for ax in axs:
        ax.set_xlim(-13, 24)

    # Y-limits
    # axs[0].set_ylim(-0.05, 0.05)
    # axs[1].set_ylim(-1, 1)
    # axs[2].set_ylim(-0.05, 0.05)

    # X-axis labels
    for ax in axs:
        ax.set_xlabel(r"$\alpha$ [°]")

    # Y-axis labels
    axs[0].set_ylabel("CMx")
    axs[1].set_ylabel("CMy")
    axs[2].set_ylabel("CMz")

    # Legend on the second subplot (or whichever you prefer)
    axs[1].legend(loc="upper left")

    # 6) Save the figure
    plt.tight_layout()
    file_name = "moment_literature_polar_alphas"
    saving_pdf_and_pdf_tex(results_dir, file_name)

    # Optionally, display the plot
    plt.show()


def plot_single_row(
    results_dir,
    data_frames,
    labels,
    colors,
    linestyles,
    markers,
    file_name,
    confidence_interval,
    axs_titles,
    legend_location_index=0,
    legend_location="lower left",
    variables_to_plot=["CL", "CD", "CS"],  # <--- NEW: which columns to plot
    show_ci=True,  # <--- NEW: toggle confidence intervals
    xlim=None,
    ylim=None,
):
    # Helper to plot lines/markers on an axis
    def plot_on_ax(ax, x, y, label, color, linestyle, marker, markersize):
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
        )

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Three subplots in one row

    for i, (data_frame, label, color, linestyle, marker) in enumerate(
        zip(data_frames, labels, colors, linestyles, markers)
    ):
        if data_frame is None:
            continue

        # Some runs might have a ratio for side-force 'CS'
        # e.g. 3D geometry difference vs. 2D reference area
        if "CFD" in label:
            ratio_projected_area_to_side_area = -3.7
        else:
            ratio_projected_area_to_side_area = 1.0

        # Marker size only for the first dataset
        markersize_i = 8 if i == 0 else None

        # Plot each chosen variable on its corresponding subplot
        for j, var in enumerate(variables_to_plot):
            # X-values are always "beta" in your setup
            x_data = data_frame["beta"]
            # Y-values may need special scaling if var == "CS"
            if var == "CS":
                y_data = data_frame[var] / ratio_projected_area_to_side_area
            else:
                y_data = data_frame[var]

            # Plot main data
            plot_on_ax(
                axs[j],
                x_data,
                y_data,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize_i,
            )

        # If this is the last dataset (i == 2) AND we want confidence intervals:
        if "WT" in label and show_ci:
            # For each variable, fill between upper/lower bounds if *_ci is in DataFrame
            for j, var in enumerate(variables_to_plot):
                # Build main Y data
                if var == "CS":
                    main_y = data_frame[var] / ratio_projected_area_to_side_area
                    ci_y = data_frame[var + "_ci"] / ratio_projected_area_to_side_area
                else:
                    main_y = data_frame[var]
                    ci_y = data_frame[var + "_ci"] if (var + "_ci") in data_frame else 0

                lower_bound = main_y - ci_y
                upper_bound = main_y + ci_y

                # Fill the CI region on positive beta
                axs[j].fill_between(
                    data_frame["beta"],
                    lower_bound,
                    upper_bound,
                    color=color,
                    alpha=0.15,
                    label=f"WT CI of {confidence_interval}%",
                )

            ### Also plot the negative-beta mirror curves:
            # You show negative-beta versions of CL,CD,CS with dashed lines
            for j, var in enumerate(variables_to_plot):
                if var == "CS" or var == "CMx" or var == "CMz":
                    main_y = data_frame[var] / ratio_projected_area_to_side_area
                    ci_y = data_frame[var + "_ci"] / ratio_projected_area_to_side_area
                    # Negative side: sign flips for CS
                    main_y_neg = -main_y
                else:
                    main_y = data_frame[var]
                    ci_y = data_frame[var + "_ci"] if (var + "_ci") in data_frame else 0
                    # Negative side: sign does NOT flip for CL, CD, etc.
                    main_y_neg = main_y

                axs[j].plot(
                    -data_frame["beta"],
                    main_y_neg,
                    color=color,
                    linestyle=linestyle.replace("-", "--"),
                    marker=marker,
                    markersize=markersize_i,
                    label=label + r" (-$\beta$)",
                )

                # If also filling CI for negative beta:
                lower_bound = main_y - ci_y
                upper_bound = main_y + ci_y
                # Sign flip only if var=="CS"
                if var == "CS" or var == "CMx" or var == "CMz":
                    lower_bound_neg = -lower_bound
                    upper_bound_neg = -upper_bound
                else:
                    lower_bound_neg = lower_bound
                    upper_bound_neg = upper_bound

                axs[j].fill_between(
                    -data_frame["beta"],
                    lower_bound_neg,
                    upper_bound_neg,
                    color="white",
                    facecolor=color,
                    alpha=0.3,
                    label=rf"WT CI of {confidence_interval}\% (-$\beta$)",
                    hatch="||",
                )

    # Final axis formatting
    # X-limits, labels, titles, etc.
    for j in range(3):
        axs[j].set_xlim(0, 20)
        axs[j].set_xlabel(r"$\beta$ [°]")
        # Optional: if you have a dictionary for custom axis titles:
        # axs[j].set_title(axs_titles[j])

    # For demonstration, label Y-axes based on variables_to_plot
    # (You could also pass in a separate param of y-axis labels if you prefer.)
    axs[0].set_ylabel(variables_to_plot[0])
    axs[1].set_ylabel(variables_to_plot[1])
    axs[2].set_ylabel(variables_to_plot[2])

    # Example special Y-limits if "CFD" in the first label
    if len(labels) > 0 and "CFD" in labels[0]:
        axs[2].set_ylim(-0.4, 0.05)
    elif "CS" in variables_to_plot:
        axs[2].set_ylim(-0.5, 0.05)

    if xlim is not None:
        axs[0].set_xlim(xlim[0])
        axs[1].set_xlim(xlim[1])
        axs[2].set_xlim(xlim[2])
    if ylim is not None:
        axs[0].set_ylim(ylim[0])
        axs[1].set_ylim(ylim[1])
        axs[2].set_ylim(ylim[2])

    # Adjust layout for subplots
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)

    # Create custom legend elements for final figure
    legend_elements = []
    last_i = len(data_frames) - 1  # Default last index for confidence intervals
    for i in range(last_i):
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=markers[i],
                color=colors[i],
                label=labels[i],
                linestyle=linestyles[i],
            )
        )
    legend_elements_WT = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            label=r"WT Re = $5.6 \times 10^5$",
            linestyle="-",
        ),
        Patch(facecolor="red", edgecolor="none", alpha=0.15, label=r"WT CI of 99%"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            linestyle="--",
            label=r"WT Re = $5.6 \times 10^5$ $(-\beta)$",
        ),
        Patch(
            facecolor="red",
            edgecolor="white",
            alpha=0.3,
            hatch="||",
            label=r"WT CI of 99% $(-\beta)$",
        ),
    ]
    legend_elements.extend(legend_elements_WT)

    # Final legend at bottom
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=True,
    )

    # Finally save
    saving_pdf_and_pdf_tex(results_dir, file_name)


def plotting_polars_beta(
    project_dir: str,
    results_dir: str,
    confidence_interval: float,
):
    polar_dir = Path(project_dir) / "processed_data" / "polar_data"
    # Load VSM data
    data_VSM_beta_re_56e4_alpha_675_breukels = pd.read_csv(
        Path(polar_dir) / f"VSM_results_beta_sweep_Rey_5.6_alpha_675_breukels.csv"
    )
    data_VSM_beta_re_56e4_alpha_675_breukels_stall = pd.read_csv(
        Path(polar_dir) / f"VSM_results_beta_sweep_Rey_5.6_alpha_675_breukels_stall.csv"
    )
    data_VSM_beta_re_56e4_alpha_675_corrected = pd.read_csv(
        Path(polar_dir) / f"VSM_results_beta_sweep_Rey_5.6_alpha_675_corrected.csv"
    )
    data_VSM_beta_re_56e4_alpha_675_corrected_stall = pd.read_csv(
        Path(polar_dir)
        / f"VSM_results_beta_sweep_Rey_5.6_alpha_675_corrected_stall.csv"
    )
    data_VSM_beta_re_56e4_alpha_1195_breukels = pd.read_csv(
        Path(polar_dir) / f"VSM_results_beta_sweep_Rey_5.6_alpha_1195_breukels.csv"
    )
    data_VSM_beta_re_56e4_alpha_1195_breukels_stall = pd.read_csv(
        Path(polar_dir)
        / f"VSM_results_beta_sweep_Rey_5.6_alpha_1195_breukels_stall.csv"
    )
    data_VSM_beta_re_56e4_alpha_1195_corrected = pd.read_csv(
        Path(polar_dir) / f"VSM_results_beta_sweep_Rey_5.6_alpha_1195_corrected.csv"
    )
    data_VSM_beta_re_56e4_alpha_1195_corrected_stall = pd.read_csv(
        Path(polar_dir)
        / f"VSM_results_beta_sweep_Rey_5.6_alpha_1195_corrected_stall.csv"
    )

    # Load Lebesque data
    path_to_csv_lebesque_re_100e4_alpha_1195 = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_RANS_Lebesque_2024_Rey_100e4_beta_sweep.csv"
    )
    data_lebesque_re_100e4_alpha_1195 = pd.read_csv(
        path_to_csv_lebesque_re_100e4_alpha_1195
    )
    # Load Wind Tunnel data
    data_WT_beta_re_56e4_alpha_6_8 = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_beta_sweep_alpha_6.8_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_WT_beta_re_56e4_alpha_11_9 = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_beta_sweep_alpha_11.9_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    # ## correcting the beta
    # data_WT_beta_re_56e4_alpha_11_9["beta"] = -data_WT_beta_re_56e4_alpha_11_9["beta"]
    # data_WT_beta_re_56e4_alpha_6_8["beta"] = -data_WT_beta_re_56e4_alpha_6_8["beta"]

    ### 1. literature_polars_beta_high_alpha
    plot_single_row(
        results_dir,
        data_frames=[
            data_lebesque_re_100e4_alpha_1195,
            data_VSM_beta_re_56e4_alpha_1195_corrected_stall,
            data_WT_beta_re_56e4_alpha_11_9,
        ],
        labels=[
            rf"CFD Re = $10\times10^5$",
            rf"VSM Re = $5.6\times10^5$",
            rf"WT Re = $5.6\times10^5$",
        ],
        colors=["black", "blue", "red"],
        linestyles=["-", "-", "-"],
        markers=["*", "s", "o"],
        file_name=f"literature_polars_beta_high_alpha",
        confidence_interval=confidence_interval,
        axs_titles=["CL (High Alpha)", "CD (High Alpha)", "CS (High Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
    )
    ### 2. literature_polars_beta_low_alpha
    plot_single_row(
        results_dir,
        data_frames=[
            data_VSM_beta_re_56e4_alpha_675_corrected_stall,
            data_WT_beta_re_56e4_alpha_6_8,
        ],
        labels=[
            # rf"VSM Re = $5.6\times10^5$",
            rf"VSM Re = $5.6\times10^5$",
            rf"WT Re = $5.6\times10^5$",
        ],
        colors=["blue", "red"],
        linestyles=["-", "-"],
        markers=["s", "o"],
        confidence_interval=confidence_interval,
        file_name=f"literature_polars_beta_low_alpha",
        axs_titles=["CL (Low Alpha)", "CD (Low Alpha)", "CS (Low Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
    )

    ### 3. literature_polars_beta_high_alpha_correction_vs_no_correction
    plot_single_row(
        results_dir,
        data_frames=[
            data_VSM_beta_re_56e4_alpha_1195_breukels,
            data_VSM_beta_re_56e4_alpha_1195_breukels_stall,
            data_VSM_beta_re_56e4_alpha_1195_corrected,
            data_VSM_beta_re_56e4_alpha_1195_corrected_stall,
            data_WT_beta_re_56e4_alpha_11_9,
        ],
        labels=[
            r"VSM Breukels",
            r"VSM Breukels Stall",
            r"VSM Corrected",
            r"VSM Corrected Stall",
            r"WT",
        ],
        colors=["blue", "blue", "green", "green", "red"],
        linestyles=["--", "-", "--", "-", "-"],
        markers=["s", "s", "+", "+", "o"],
        confidence_interval=confidence_interval,
        file_name=f"literature_polars_beta_alpha_1195_correction_and_stall_effects",
        axs_titles=["CL (High Alpha)", "CD (High Alpha)", "CS (High Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
    )
    ### 4. literature_polars_beta_low_alpha_correction_vs_no_correction
    plot_single_row(
        results_dir,
        data_frames=[
            data_VSM_beta_re_56e4_alpha_675_breukels,
            data_VSM_beta_re_56e4_alpha_675_breukels_stall,
            data_VSM_beta_re_56e4_alpha_675_corrected,
            data_VSM_beta_re_56e4_alpha_675_corrected_stall,
            data_WT_beta_re_56e4_alpha_6_8,
        ],
        labels=[
            r"VSM Breukels",
            r"VSM Breukels Stall",
            r"VSM Corrected",
            r"VSM Corrected Stall",
            r"WT",
        ],
        colors=["blue", "blue", "green", "green", "red"],
        linestyles=["--", "-", "--", "-", "-"],
        markers=["s", "s", "+", "+", "o"],
        confidence_interval=confidence_interval,
        file_name=f"literature_polars_beta_alpha_675_correction_and_stall_effects",
        axs_titles=["CL (Low Alpha)", "CD (Low Alpha)", "CS (Low Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
    )


def plotting_polars_beta_moments(
    project_dir: str,
    results_dir: str,
    confidence_interval: float,
):
    """
    Plots CMx, CMy, CMz vs beta for high-alpha and low-alpha cases, comparing
    no-correction vs correction for VSM, plus wind tunnel data.

    The file names are assumed to have '_moment' appended for the VSM data,
    e.g. 'VSM_results_beta_sweep_Rey_5.6_alpha_675_no_correction_moment.csv'.
    """
    polar_dir = Path(project_dir) / "processed_data" / "polar_data"
    # 1) Load VSM (High alpha = 11.95 deg) moment data
    data_VSM_beta_re_56e4_alpha_1195_breukels_moment = pd.read_csv(
        Path(polar_dir)
        / "VSM_results_beta_sweep_Rey_5.6_alpha_1195_breukels_moment.csv"
    )
    data_VSM_beta_re_56e4_alpha_1195_breukels_stall_moment = pd.read_csv(
        Path(polar_dir)
        / "VSM_results_beta_sweep_Rey_5.6_alpha_1195_breukels_stall_moment.csv"
    )
    data_VSM_beta_re_56e4_alpha_1195_corrected_moment = pd.read_csv(
        Path(polar_dir)
        / "VSM_results_beta_sweep_Rey_5.6_alpha_1195_corrected_moment.csv"
    )
    data_VSM_beta_re_56e4_alpha_1195_corrected_stall_moment = pd.read_csv(
        Path(polar_dir)
        / "VSM_results_beta_sweep_Rey_5.6_alpha_1195_corrected_stall_moment.csv"
    )

    # 2) Load VSM (Low alpha = 6.75 deg) moment data
    data_VSM_beta_re_56e4_alpha_675_breukels_moment = pd.read_csv(
        Path(polar_dir) / "VSM_results_beta_sweep_Rey_5.6_alpha_675_breukels_moment.csv"
    )
    data_VSM_beta_re_56e4_alpha_675_breukels_stall_moment = pd.read_csv(
        Path(polar_dir)
        / "VSM_results_beta_sweep_Rey_5.6_alpha_675_breukels_stall_moment.csv"
    )
    data_VSM_beta_re_56e4_alpha_675_corrected_moment = pd.read_csv(
        Path(polar_dir)
        / "VSM_results_beta_sweep_Rey_5.6_alpha_675_corrected_moment.csv"
    )
    data_VSM_beta_re_56e4_alpha_675_corrected_stall_moment = pd.read_csv(
        Path(polar_dir)
        / "VSM_results_beta_sweep_Rey_5.6_alpha_675_corrected_stall_moment.csv"
    )

    # 3) Load Wind Tunnel moment data
    #    (Assuming you saved them as "V3_CMx_CMy_CMz_beta_sweep_alpha_6_8_..." etc.)
    data_WT_beta_re_56e4_alpha_6_8_moment = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CMx_CMy_CMz_beta_sweep_alpha_6.8_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_WT_beta_re_56e4_alpha_11_9_moment = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CMx_CMy_CMz_beta_sweep_alpha_11.9_WindTunnel_Poland_2025_Rey_560e4.csv"
    )

    # ## correcting the beta
    # data_WT_beta_re_56e4_alpha_11_9_moment["beta"] = (
    #     -data_WT_beta_re_56e4_alpha_11_9_moment["beta"]
    # )
    # data_WT_beta_re_56e4_alpha_6_8_moment["beta"] = (
    #     -data_WT_beta_re_56e4_alpha_6_8_moment["beta"]
    # )

    # 1. moment_literature_polars_beta_high_alpha_correction_vs_no_correction
    plot_single_row(
        results_dir=results_dir,
        data_frames=[
            data_VSM_beta_re_56e4_alpha_675_breukels_moment,
            data_VSM_beta_re_56e4_alpha_675_breukels_stall_moment,
            data_VSM_beta_re_56e4_alpha_675_corrected_moment,
            data_VSM_beta_re_56e4_alpha_675_corrected_stall_moment,
            data_WT_beta_re_56e4_alpha_6_8_moment,
        ],
        labels=[
            r"VSM Breukels",
            r"VSM Breukels Stall",
            r"VSM Corrected",
            r"VSM Corrected Stall",
            r"WT",
        ],
        colors=["blue", "blue", "green", "green", "red"],
        linestyles=["--", "-", "--", "-", "-"],
        markers=["s", "s", "+", "+", "o"],
        file_name="moment_literature_polars_beta_alpha_6_8_correction_and_stall_effect",
        confidence_interval=confidence_interval,
        axs_titles=["CMx (High Alpha)", "CMy (High Alpha)", "CMz (High Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
        variables_to_plot=["CMx", "CMy", "CMz"],  # <--- Key difference
        # ylim=[[-0.2, 0.5], [-0.2, 0.5], [-0.3, 0.6]],
    )

    # 2. moment_literature_polars_beta_low_alpha_correction_vs_no_correction
    plot_single_row(
        results_dir=results_dir,
        data_frames=[
            data_VSM_beta_re_56e4_alpha_1195_breukels_moment,
            data_VSM_beta_re_56e4_alpha_1195_breukels_stall_moment,
            data_VSM_beta_re_56e4_alpha_1195_corrected_moment,
            data_VSM_beta_re_56e4_alpha_1195_corrected_stall_moment,
            data_WT_beta_re_56e4_alpha_11_9_moment,
        ],
        labels=[
            r"VSM Breukels",
            r"VSM Breukels Stall",
            r"VSM Corrected",
            r"VSM Corrected Stall",
            r"WT",
        ],
        colors=["blue", "blue", "green", "green", "red"],
        linestyles=["--", "-", "--", "-", "-"],
        markers=["s", "s", "+", "+", "o"],
        file_name="moment_literature_polars_beta_alpha_11_95_correction_and_stall_effect",
        confidence_interval=confidence_interval,
        axs_titles=["CMx (Low Alpha)", "CMy (Low Alpha)", "CMz (Low Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
        variables_to_plot=["CMx", "CMy", "CMz"],  # <--- Key difference
        # ylim=[[-0.2, 0.5], [-0.2, 0.5], [-0.3, 0.6]],
    )

    # 3. moment_literature_polars_beta_high_alpha
    plot_single_row(
        results_dir=results_dir,
        data_frames=[
            data_VSM_beta_re_56e4_alpha_1195_corrected_stall_moment,
            data_WT_beta_re_56e4_alpha_11_9_moment,
        ],
        labels=[
            r"VSM $\mathrm{Re} = 5.6\times10^5$",
            r"WT $\mathrm{Re} = $5.6\times10^5$",
        ],
        colors=["blue", "red"],
        linestyles=["-", "-"],
        markers=["s", "o"],
        file_name="moment_literature_polars_beta_high_alpha",
        confidence_interval=confidence_interval,
        axs_titles=["CMx (High Alpha) moment", "CMy (High Alpha)", "CMz (High Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
        variables_to_plot=["CMx", "CMy", "CMz"],  # <--- Key difference
        # ylim=[[-0.2, 0.5], [-0.2, 0.5], [-0.3, 0.6]],
    )

    # 4. moment_literature_polars_beta_low_alpha
    plot_single_row(
        results_dir=results_dir,
        data_frames=[
            data_VSM_beta_re_56e4_alpha_675_corrected_stall_moment,
            data_WT_beta_re_56e4_alpha_6_8_moment,
        ],
        labels=[
            r"VSM $\mathrm{Re} = 5.6\times10^5$",
            r"WT $\mathrm{Re}e = 5.6\times10^5$",
        ],
        colors=["blue", "red"],
        linestyles=["-", "-"],
        markers=["s", "o"],
        file_name="moment_literature_polars_beta_low_alpha",
        confidence_interval=confidence_interval,
        axs_titles=["CMx (Low Alpha) moment", "CMy (Low Alpha)", "CMz (Low Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
        variables_to_plot=["CMx", "CMy", "CMz"],  # <--- Key difference
        # ylim=[[-0.2, 0.5], [-0.2, 0.5], [-0.3, 0.6]],
    )


def main(results_dir, project_dir):

    confidence_interval = 99
    saving_alpha_and_beta_sweeps(
        project_dir,
        confidence_interval=confidence_interval,
        max_lag=11,
    )

    # plotting_polars_alpha(
    #     project_dir,
    #     results_dir,
    #     confidence_interval=confidence_interval,
    # )
    plotting_polars_alpha_correction_comparison(
        project_dir,
        results_dir,
        confidence_interval=confidence_interval,
    )
    plotting_polars_alpha_moments_correction_comparison(
        project_dir,
        results_dir,
        confidence_interval=confidence_interval,
    )
    # plotting_polars_alpha_moments_vsm_wt(
    #     project_dir,
    #     results_dir,
    #     confidence_interval=confidence_interval,
    # )
    plotting_polars_beta(
        project_dir,
        results_dir,
        confidence_interval=confidence_interval,
    )

    plotting_polars_beta_moments(
        project_dir,
        results_dir,
        confidence_interval=confidence_interval,
    )
