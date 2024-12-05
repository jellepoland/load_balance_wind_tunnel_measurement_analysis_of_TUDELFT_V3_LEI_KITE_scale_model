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
from plot_styling import plot_on_ax


def raw_df_to_mean_and_ci(
    df: pd.DataFrame,
    variable_col: str,
    coefficients: list,
    confidence_interval: float,
    max_lag: int,
) -> pd.DataFrame:

    ## Standard Deviation
    # data_calculated = [
    #     [data[coeff].mean(), data[coeff].std()] for coeff in coefficients
    # ]
    # ## Confidence Interval
    # data_calculated = [
    #     [data[coeff].mean(), calculate_confidence_interval(data[coeff])] for coeff in coefficients
    # ]
    ## Block Bootstrap Confidence Interval
    # data_calculated = [
    #     [data[coeff].mean(), block_bootstrap_confidence_interval(data[coeff])]
    #     for coeff in coefficients
    # ]
    # ## HAC Newey-West Confidence Interval
    # data_calculated = [
    #     [
    #         data[coeff].mean(),
    #         hac_newey_west_confidence_interval(
    #             data[coeff],confidence_interval = 99.99, max_lag=11, )
    #         ),
    #     ]
    #     for coeff in coefficients
    # ]
    # # transform back into df
    # for coeff in coefficients:
    #     for data in data_calculated:
    #         mean = data[0]
    #         ci = data[1]

    #         data = pd.DataFrame(data, columns=["mean", "ci"])

    # data_calculated = pd.DataFrame(data_calculated, columns=["mean", "ci"])

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
):

    ### Alpha part
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

        df = raw_df_to_mean_and_ci(
            df_all_values,
            variable_col="aoa_kite",
            coefficients=["C_L", "C_D"],
            confidence_interval=confidence_interval,
            max_lag=max_lag,
        )
        data_windtunnel.append(df)

    # changing names to be consistent
    df_alpha = data_windtunnel[1]
    col_names = [
        "CL",
        "CL_ci",
        "CD",
        "CD_ci",
        "aoa",
    ]
    df_alpha.columns = col_names

    # Saving
    df_alpha.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv",
        columns=col_names,
        index=False,
    )

    ### Beta sweep part
    vw = 20
    alpha_high = 11.9
    alpha_low = 6.8
    folder_name = f"alpha_{alpha_high}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(project_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)

    data_WT_beta_re_56e4_alpha_high = raw_df_to_mean_and_ci(
        df_all_values,
        variable_col="sideslip",
        coefficients=["C_L", "C_D", "C_S"],
        confidence_interval=confidence_interval,
        max_lag=max_lag,
    )
    # data_WT_beta_re_56e4_alpha_high = reduce_df_by_parameter_mean_and_std(
    #     df_all_values, "sideslip"
    # )

    folder_name = f"alpha_{alpha_low}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(project_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)
    data_WT_beta_re_56e4_alpha_low = raw_df_to_mean_and_ci(
        df_all_values,
        variable_col="sideslip",
        coefficients=["C_L", "C_D", "C_S"],
        confidence_interval=confidence_interval,
        max_lag=max_lag,
    )

    # Saving
    col_names = ["CL", "CL_ci", "CD", "CD_ci", "CS", "CS_ci", "beta"]
    data_WT_beta_re_56e4_alpha_high.columns = col_names
    data_WT_beta_re_56e4_alpha_high.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_beta_sweep_alpha_11_9_WindTunnel_Poland_2025_Rey_560e4.csv",
        # columns=col_names,
        index=False,
    )

    df_alpha = data_WT_beta_re_56e4_alpha_low
    col_names = ["CL", "CL_ci", "CD", "CD_ci", "CS", "CS_ci", "beta"]
    data_WT_beta_re_56e4_alpha_low.columns = col_names
    data_WT_beta_re_56e4_alpha_low.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_beta_sweep_alpha_6_8_WindTunnel_Poland_2025_Rey_560e4.csv",
        # columns=col_names,
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
        / "VSM_results_alpha_sweep_Rey_5.6.csv"
    )
    data_VSM_alpha_re_56e4 = pd.read_csv(path_to_csv_VSM_alpha_re_56e4)

    data_lebesque_paths = [
        "V3_CL_CD_RANS_Lebesque_2024_Rey_10e4.csv",
        # "V3_CL_CD_RANS_Vire2020_Rey_50e4.csv",
        "V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv",
        "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv",
    ]
    data_lebesque = [
        pd.read_csv(Path(project_dir) / "data" / "CFD_polar_data" / file)
        for file in data_lebesque_paths
    ]

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
        data_lebesque[0],
        data_lebesque[1],
        data_VSM_alpha_re_56e4,
        data_windtunnel,
        # pd.read_csv(
        #     Path(project_dir)
        #     / "processed_data"
        #     / "polar_data"
        #     / "polar_2019-10-08.csv"
        # ),
    ]
    labels = [
        rf"CFD Re = $1\times10^5$",
        rf"CFD Re = $10\times10^5$",
        rf"VSM Re = $5.6\times10^5$",
        rf"WT Re = $5.6\times10^5$",
        # rf"Polars Uri",
    ]
    colors = ["black", "black", "blue", "red", "green"]
    linestyles = ["--", "-", "-", "-", "-"]
    markers = ["*", "s", "o", "s"]
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
    axs[0].legend(loc="lower right")

    axs[1].set_xlabel(x_axis_labels["alpha"])
    axs[1].set_ylabel(y_axis_labels["CD"])
    # axs[1].grid()

    axs[2].set_xlabel(x_axis_labels["alpha"])
    axs[2].set_ylabel(y_axis_labels["L/D"])
    # axs[2].grid()

    # Save the plot
    plt.tight_layout()
    file_name = "literature_polars_alpha"
    saving_pdf_and_pdf_tex(results_dir, file_name)


def plotting_polars_beta(
    project_dir: str,
    results_dir: str,
    ratio_projected_area_to_side_area: float,
    confidence_interval: float,
):

    # Load VSM data
    path_to_csv_VSM_beta_re_56e4_alpha_1195 = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "VSM_results_beta_sweep_Rey_5.6_alpha_1195.csv"
    )
    data_VSM_beta_re_56e4_alpha_1195 = pd.read_csv(
        path_to_csv_VSM_beta_re_56e4_alpha_1195
    )

    # Load VSM data
    path_to_csv_VSM_beta_re_56e4_alpha_675 = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "VSM_results_beta_sweep_Rey_5.6_alpha_675.csv"
    )
    data_VSM_beta_re_56e4_alpha_675 = pd.read_csv(
        path_to_csv_VSM_beta_re_56e4_alpha_675
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
        / "V3_CL_CD_CS_beta_sweep_alpha_6_8_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_WT_beta_re_56e4_alpha_11_9 = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_beta_sweep_alpha_11_9_WindTunnel_Poland_2025_Rey_560e4.csv"
    )

    def plot_single_row(
        data_frames,
        labels,
        file_name,
        axs_titles,
        legend_location_index=0,
        legend_location="lower left",
    ):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Maintain original aspect ratio
        colors = ["black", "blue", "red"]
        linestyles = ["-", "-", "-"]
        markers = ["*", "s", "o"]
        fmt_wt = "x"

        for i, (data_frame, label, color, linestyle, marker) in enumerate(
            zip(data_frames, labels, colors, linestyles, markers)
        ):
            if data_frame is None:
                continue

            if i == 0:
                markersize = 8
            else:
                markersize = None

            if "CFD" in label:
                axs[0].set_ylim(0.5, 1.1)

            plot_on_ax(
                axs[0],
                data_frame["beta"],
                data_frame["CL"],
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
            )
            plot_on_ax(
                axs[1],
                data_frame["beta"],
                data_frame["CD"],
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
            )
            plot_on_ax(
                axs[2],
                data_frame["beta"],
                data_frame["CS"] / ratio_projected_area_to_side_area,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
            )
            if i == 2:
                # Create shaded regions for confidence intervals
                axs[0].fill_between(
                    data_frame["beta"],
                    data_frame["CL"] - data_frame["CL_ci"],  # Lower bound
                    data_frame["CL"] + data_frame["CL_ci"],  # Upper bound
                    color=color,
                    alpha=0.15,  # Adjust transparency
                    label=f"WT CI of {confidence_interval}\%",
                )
                axs[1].fill_between(
                    data_frame["beta"],
                    data_frame["CD"] - data_frame["CD_ci"],  # Lower bound
                    data_frame["CD"] + data_frame["CD_ci"],  # Upper bound
                    color=color,
                    alpha=0.15,  # Adjust transparency
                    label=f"WT CI of {confidence_interval}\%",
                )
                low_bound = (
                    data_frame["CS"] / ratio_projected_area_to_side_area
                    - data_frame["CS_ci"] / ratio_projected_area_to_side_area
                )
                upper_bound = (
                    data_frame["CS"] / ratio_projected_area_to_side_area
                    + data_frame["CS_ci"] / ratio_projected_area_to_side_area
                )

                axs[2].fill_between(
                    data_frame["beta"],
                    low_bound,  # Lower bound
                    upper_bound,  # Upper bound
                    color=color,
                    alpha=0.15,  # Adjust transparency
                    label=f"WT CI of {confidence_interval}\%",
                )

                plot_on_ax(
                    axs[0],
                    -data_frame["beta"],
                    data_frame["CL"],
                    label=label + rf"(-$\beta$)",
                    color=color,
                    linestyle=linestyle.replace("-", "--"),
                    marker=marker,
                    markersize=markersize,
                )
                plot_on_ax(
                    axs[1],
                    -data_frame["beta"],
                    data_frame["CD"],
                    label=label + rf"(-$\beta$)",
                    color=color,
                    linestyle=linestyle.replace("-", "--"),
                    marker=marker,
                    markersize=markersize,
                )
                plot_on_ax(
                    axs[2],
                    -data_frame["beta"],
                    -data_frame["CS"] / ratio_projected_area_to_side_area,
                    label=label + rf"(-$\beta$)",
                    color=color,
                    linestyle=linestyle.replace("-", "--"),
                    marker=marker,
                    markersize=markersize,
                )
                # Create shaded regions for confidence intervals
                axs[0].fill_between(
                    -data_frame["beta"],
                    data_frame["CL"] - data_frame["CL_ci"],  # Lower bound
                    data_frame["CL"] + data_frame["CL_ci"],  # Upper bound
                    color="white",
                    facecolor=color,
                    alpha=0.3,  # Adjust transparency
                    label=rf"WT CI of {confidence_interval}\% (-$\beta$)",
                    hatch="||",
                )
                axs[1].fill_between(
                    -data_frame["beta"],
                    data_frame["CD"] - data_frame["CD_ci"],  # Lower bound
                    data_frame["CD"] + data_frame["CD_ci"],  # Upper bound
                    color="white",
                    facecolor=color,
                    alpha=0.3,  # Adjust transparency
                    label=rf"WT CI of {confidence_interval}\% (-$\beta$)",
                    hatch="||",
                )
                axs[2].fill_between(
                    -data_frame["beta"],
                    -(
                        data_frame["CS"] / ratio_projected_area_to_side_area
                        - data_frame["CS_ci"] / ratio_projected_area_to_side_area
                    ),  # Lower bound
                    -(
                        data_frame["CS"] / ratio_projected_area_to_side_area
                        + data_frame["CS_ci"] / ratio_projected_area_to_side_area
                    ),  # Upper bound
                    color="white",
                    facecolor=color,
                    alpha=0.3,  # Adjust transparency
                    label=rf"WT CI of {confidence_interval}\% (-$\beta$)",
                    hatch="||",
                )

        # Set common formatting
        for col in range(3):
            axs[col].set_xlim(0, 20)
            # axs[col].grid()
            axs[col].set_xlabel(x_axis_labels["beta"])
            # axs[col].set_title(axs_titles[col])

        # Only add legend to the first plot (CL)
        axs[0].set_ylabel(y_axis_labels["CL"])
        axs[legend_location_index].legend(loc=legend_location)

        axs[1].set_ylabel(y_axis_labels["CD"])
        axs[2].set_ylabel(y_axis_labels["CS"])
        # axs[2].set_ylim(-0.05, 0.6)

        # Adjust layout and save
        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.15)  # Leave space for legend
        saving_pdf_and_pdf_tex(results_dir, file_name)

    # Split data into high and low alpha groups
    high_alpha_data = [
        data_lebesque_re_100e4_alpha_1195,
        data_VSM_beta_re_56e4_alpha_1195,
        data_WT_beta_re_56e4_alpha_11_9,
    ]
    low_alpha_data = [
        None,  # No Lebesque data for low alpha
        data_VSM_beta_re_56e4_alpha_675,
        data_WT_beta_re_56e4_alpha_6_8,
    ]

    # Split labels, colors and linestyles accordingly
    high_alpha_labels = [
        rf"CFD Re = $10\times10^5$",
        rf"VSM Re = $5.6\times10^5$",
        rf"WT Re = $5.6\times10^5$",
    ]
    low_alpha_labels = [
        "",  # Empty label for missing Lebesque data
        rf"VSM Re = $5.6\times10^5$",
        rf"WT Re = $5.6\times10^5$",
    ]

    # First plot: High Alpha
    plot_single_row(
        high_alpha_data,
        high_alpha_labels,
        file_name="literature_polars_beta_high_alpha",
        axs_titles=["CL (High Alpha)", "CD (High Alpha)", "CS (High Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
    )

    # Second plot: Low Alpha
    plot_single_row(
        low_alpha_data,
        low_alpha_labels,
        file_name="literature_polars_beta_low_alpha",
        axs_titles=["CL (Low Alpha)", "CD (Low Alpha)", "CS (Low Alpha)"],
        legend_location_index=1,
        legend_location="upper left",
    )


def main(results_dir, project_dir):

    confidence_interval = 99
    # fontsize = 18
    # figsize = (20, 6)

    saving_alpha_and_beta_sweeps(
        project_dir,
        confidence_interval=confidence_interval,
        max_lag=11,
    )

    plotting_polars_alpha(
        project_dir,
        results_dir,
        confidence_interval=confidence_interval,
    )

    # figsize = (20, 6)
    plotting_polars_beta(
        project_dir,
        results_dir,
        ratio_projected_area_to_side_area=3.7,
        confidence_interval=confidence_interval,
    )


if __name__ == "__main__":

    results_dir = Path(project_dir) / "results"
    main(results_dir, project_dir)
