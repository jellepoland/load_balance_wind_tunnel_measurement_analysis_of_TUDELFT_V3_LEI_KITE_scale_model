import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from utils import (
    project_dir,
    saving_pdf_and_pdf_tex,
    x_axis_labels,
    y_axis_labels,
    reduce_df_by_parameter_mean_and_std,
)


def plotting_polars_alpha(
    project_dir: str,
    results_dir: str,
    figsize: tuple,
    fontsize: int,
):
    ## 1.4e5
    vw = 5
    beta_value = 0
    folder_name = f"beta_{beta_value}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(project_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)
    data_windtunnel_alpha_re_14e4 = reduce_df_by_parameter_mean_and_std(
        df_all_values, "aoa_kite"
    )

    ## 5.6e5
    vw = 20
    beta_value = 0
    folder_name = f"beta_{beta_value}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(project_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)
    data_windtunnel_alpha_re_56e4 = reduce_df_by_parameter_mean_and_std(
        df_all_values, "aoa_kite"
    )

    # for i, _ in enumerate(data_windtunnel_alpha_re_56e4.iterrows()):
    #     row = data_windtunnel_alpha_re_56e4.iloc[i]
    #     print(f"aoa: {row['aoa_kite']}, C_L: {row['C_L']}")

    # VSM
    path_to_csv_VSM_alpha_re_56e4 = (
        Path(project_dir)
        / "processed_data"
        / "literature_comparison"
        / "VSM_results_alpha_sweep_Rey_5.6.csv"
    )
    data_VSM_alpha_re_56e4 = pd.read_csv(path_to_csv_VSM_alpha_re_56e4)
    # Lebesque
    path_to_csv_lebesque_Rey_10e4 = (
        Path(project_dir)
        / "processed_data"
        / "literature_comparison"
        / "V3_CL_CD_RANS_Lebesque_2024_Rey_10e4.csv"
    )
    data_lebesque_alpha_re_10e4 = pd.read_csv(path_to_csv_lebesque_Rey_10e4)

    path_to_csv_lebesque_Rey_100e4 = (
        Path(project_dir)
        / "processed_data"
        / "literature_comparison"
        / "V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv"
    )
    data_lebesque_alpha_re_100e4 = pd.read_csv(path_to_csv_lebesque_Rey_100e4)

    path_to_csv_lebesque_Rey_300e4 = (
        Path(project_dir)
        / "processed_data"
        / "literature_comparison"
        / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
    )
    data_lebesque_alpha_re_300e4 = pd.read_csv(path_to_csv_lebesque_Rey_300e4)

    # data_frame_list = [
    #     data_lebesque_alpha_re_100e4,
    #     data_VSM_alpha_re_56e4,
    #     data_windtunnel_alpha_re_56e4,
    # ]
    # labels = [
    #     rf"CFD Re = $10\times10^5$",
    #     rf"VSM Re = $5.6\times10^5$",
    #     rf"WT Re = $5.6\times10^5$",
    # ]
    # colors = ["black", "blue", "red"]
    # linestyles = ["s-", "s-", "o-"]
    data_frame_list = [
        # data_windtunnel_alpha_re_42e4,
        # data_lebesque_alpha_re_300e4,
        data_lebesque_alpha_re_10e4,
        data_lebesque_alpha_re_100e4,
        data_VSM_alpha_re_56e4,
        data_windtunnel_alpha_re_56e4,
        # data_windtunnel_alpha_re_14e4,
    ]
    labels = [
        # rf"Re = $4.2\times10^5$ Wind Tunnel",
        # rf"Re = $30\times10^5$ CFD (Lebesque, 2022)",
        rf"CFD Re = $1\times10^5$",
        rf"CFD Re = $10\times10^5$",
        rf"VSM Re = $5.6\times10^5$",
        rf"WT Re = $5.6\times10^5$",
        # rf"WT Re = $1.4\times10^5$",
    ]

    colors = ["black", "black", "blue", "red", "red"]
    linestyles = ["o--", "o-", "o-", "s-"]
    markersizes = [4, 4, 4, 6]

    # Plot CL, CD, and CS curves in subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    for i, (data_frame, label, color, linestyle, markersize) in enumerate(
        zip(data_frame_list, labels, colors, linestyles, markersizes)
    ):
        # if CFD
        if i == 0 or i == 1:
            axs[0].plot(
                data_frame["aoa"],
                data_frame["CL"],
                linestyle,
                label=label,
                color=color,
                markersize=markersize,
            )
            axs[1].plot(
                data_frame["aoa"],
                data_frame["CD"],
                linestyle,
                label=label,
                color=color,
                markersize=markersize,
            )
            axs[2].plot(
                data_frame["aoa"],
                data_frame["CL"] / data_frame["CD"],
                linestyle,
                label=label,
                color=color,
                markersize=markersize,
            )
        # if VSM
        elif i == 2:

            # # No stall model
            # linestyle = "s--"
            # axs[0].plot(
            #     data_frame["aoa"],
            #     data_frame["CL"],
            #     linestyle,
            #     label=label + " no stall model",
            # )
            # axs[1].plot(
            #     data_frame["aoa"],
            #     data_frame["CD"],
            #     linestyle,
            #     label=label + " no stall model",
            # )
            # axs[2].plot(
            #     data_frame["aoa"],
            #     data_frame["CL"] / data_frame["CD"],
            #     linestyle,
            #     label=label + " no stall model",
            # )

            # Adding stall-corrected values
            axs[0].plot(
                data_frame["aoa"],
                data_frame["CL_stall"],
                linestyle,
                label=label,
                color=color,
                markersize=markersize,
            )
            axs[1].plot(
                data_frame["aoa"],
                data_frame["CD_stall"],
                linestyle,
                label=label,
                color=color,
                markersize=markersize,
            )
            axs[2].plot(
                data_frame["aoa"],
                data_frame["CL_stall"] / data_frame["CD_stall"],
                linestyle,
                label=label,
                color=color,
                markersize=markersize,
            )

        # if wind tunnel
        else:
            axs[0].plot(
                data_frame["aoa_kite"],
                data_frame["C_L"],
                linestyle,
                label=label,
                color=color,
                markersize=markersize,
            )
            axs[1].plot(
                data_frame["aoa_kite"],
                data_frame["C_D"],
                linestyle,
                label=label,
                color=color,
                markersize=markersize,
            )
            axs[2].plot(
                data_frame["aoa_kite"],
                data_frame["C_L"] / data_frame["C_D"],
                linestyle,
                label=label,
                color=color,
                markersize=markersize,
            )

    # # adding the older data
    # (
    #     cay_alpha_cd,
    #     cay_cd_aoa,
    #     cay_alpha_cl,
    #     cay_cl_aoa,
    #     cay_alpha_clcd,
    #     cay_clcd_aoa,
    #     cay_cdcl_cd,
    #     cay_cdcl_cl,
    # ) = loading_data_vsm_old_alpha()
    # label_cay = r"Re = $3\times10^6$ (Cayon, 2022)"
    # axs[0, 0].plot(cay_alpha_cl, cay_cl_aoa, "o--", label=label_cay)
    # axs[0, 1].plot(cay_alpha_cd, cay_cd_aoa, "o--", label=label_cay)
    # axs[1, 0].plot(cay_alpha_clcd, cay_clcd_aoa, "o--", label=label_cay)
    # axs[1, 1].plot(cay_cdcl_cd, cay_cdcl_cl, "o--", label=label_cay)

    # formatting axis
    axs[0].set_xlabel(x_axis_labels["alpha"])
    axs[0].set_ylabel(y_axis_labels["CL"])
    axs[0].grid()
    axs[0].legend(loc="lower right")

    axs[1].set_xlabel(x_axis_labels["alpha"])
    axs[1].set_ylabel(y_axis_labels["CD"])
    axs[1].grid()

    axs[2].set_xlabel(x_axis_labels["alpha"])
    axs[2].set_ylabel(y_axis_labels["L/D"])
    axs[2].grid()

    # axs[0].set_xlabel(r"$\alpha$ [$^o$]")  # , fontsize=fontsize)
    # axs[0].set_ylabel(r"$C_L$ [-]")  # , fontsize=fontsize)
    # # axs[0].set_title("Lift Coefficient")
    # axs[0].set_xlim(-12.65, 24)
    # axs[0].grid()
    # # axs[0].set_xlim([-5, 24])
    # axs[0].set_ylim(-1.0, 1.5)
    # axs[0].legend(loc="lower right")

    # axs[1].set_xlabel(r"$\alpha$ [$^o$]")  # , fontsize=fontsize)
    # axs[1].set_ylabel(r"$C_D$ [-]")  # , fontsize=fontsize)
    # # axs[1].set_title("Drag Coefficient")
    # axs[1].grid()
    # axs[1].set_xlim(-12.65, 24)
    # # axs[1].set_xlim([-5, 24])
    # axs[1].set_ylim(0, 0.5)
    # # axs[1].legend(loc="upper left")

    # axs[2].set_xlabel(r"$\alpha$ [$^o$]")  # , fontsize=fontsize)
    # axs[2].set_ylabel(r"$L/D$ [-]")  # , fontsize=fontsize)
    # # axs[2].set_title("Lift/drag ratio")
    # axs[2].grid()
    # axs[2].set_xlim(-12.65, 24)
    # # axs[2].set_xlim([-5, 24])
    # axs[2].set_ylim(-10, 11)

    # plotting and saving
    plt.tight_layout()
    file_name = "literature_polars_alpha"
    saving_pdf_and_pdf_tex(results_dir, file_name)


def plotting_polars_beta(
    project_dir: str,
    results_dir: str,
    figsize: tuple,
    fontsize: int,
    ratio_projected_area_to_side_area: float = 3.7,
):

    ## NEW
    vw = 20
    alpha_high = 11.9
    alpha_low = 6.8
    folder_name = f"alpha_{alpha_high}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(project_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)
    data_WT_beta_re_56e4_alpha_high = reduce_df_by_parameter_mean_and_std(
        df_all_values, "sideslip"
    )

    folder_name = f"alpha_{alpha_low}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(project_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)
    data_WT_beta_re_56e4_alpha_low = reduce_df_by_parameter_mean_and_std(
        df_all_values, "sideslip"
    )

    def averaging_pos_and_neg_sideslip(df: pd.DataFrame) -> pd.DataFrame:

        # Define the columns of interest
        columns_to_average = ["C_L", "C_D", "C_S"]

        # Loop through the relevant sideslip angles
        for slip in [2, 4, 6, 8, 10, 12, 14, 20]:
            # Get the rows for positive and negative sideslip values
            df_positive = df[df["sideslip"] == slip].copy()
            df_negative = df[df["sideslip"] == -slip].copy()

            # Loop through the specified columns
            for col in columns_to_average:
                if col == "C_S":
                    # Take the absolute value of C_S before averaging
                    avg_values = (df_positive[col].values - df_negative[col].values) / 2
                else:
                    # Average C_L and C_D normally
                    avg_values = (df_positive[col].values + df_negative[col].values) / 2

                # Replace the values for the positive sideslip row with the averaged values
                df.loc[df["sideslip"] == slip, col] = avg_values
        return df

    data_WT_beta_re_56e4_alpha_high = averaging_pos_and_neg_sideslip(
        data_WT_beta_re_56e4_alpha_high
    )
    data_WT_beta_re_56e4_alpha_low = averaging_pos_and_neg_sideslip(
        data_WT_beta_re_56e4_alpha_low
    )

    # Load VSM data
    path_to_csv_VSM_beta_re_56e4_alpha_1195 = (
        Path(project_dir)
        / "processed_data"
        / "literature_comparison"
        / "VSM_results_beta_sweep_Rey_5.6_alpha_1195.csv"
    )
    data_VSM_beta_re_56e4_alpha_1195 = pd.read_csv(
        path_to_csv_VSM_beta_re_56e4_alpha_1195
    )

    # Load VSM data
    path_to_csv_VSM_beta_re_56e4_alpha_675 = (
        Path(project_dir)
        / "processed_data"
        / "literature_comparison"
        / "VSM_results_beta_sweep_Rey_5.6_alpha_675.csv"
    )
    data_VSM_beta_re_56e4_alpha_675 = pd.read_csv(
        path_to_csv_VSM_beta_re_56e4_alpha_675
    )

    # Load Lebesque data
    path_to_csv_lebesque_re_100e4_alpha_1195 = (
        Path(project_dir)
        / "processed_data"
        / "literature_comparison"
        / "V3_CL_CD_CS_RANS_Lebesque_2024_Rey_100e4_beta_sweep.csv"
    )
    data_lebesque_re_100e4_alpha_1195 = pd.read_csv(
        path_to_csv_lebesque_re_100e4_alpha_1195
    )
    # path_to_csv_lebesque_re_300e4 = (
    #     Path(project_dir)
    #     / "processed_data"
    #     / "V3_CL_CD_CS_RANS_Lebesque_2024_Rey_300e4_beta_sweep.csv"
    # )
    # data_lebesque_re_300e4 = pd.read_csv(path_to_csv_lebesque_re_300e4)

    data_frame_list = [
        # data_lebesque_re_300e4,
        data_lebesque_re_100e4_alpha_1195,
        data_VSM_beta_re_56e4_alpha_1195,
        data_VSM_beta_re_56e4_alpha_675,
        # data_windtunnel_beta_re_42e4,
        data_WT_beta_re_56e4_alpha_high,
        data_WT_beta_re_56e4_alpha_low,
        # data_windtunnel_beta_re_56e4_alpha_475,
    ]
    labels = [
        # rf"Re = $30e\times10^5$ CFD (Lebesque, 2022)",
        rf"CFD $\alpha$ = 12.0$^\circ$ Re = $10\times10^5$",
        rf"VSM $\alpha$ = 11.9$^\circ$ Re = $5.6\times10^5$",
        rf"VSM $\alpha$ = 6.8$^\circ$ Re = $5.6\times10^5$",
        # rf"Re = $4.2\times10^5$ Wind Tunnel",
        rf"WT $\alpha$ = {alpha_high}$^\circ$ Re = $5.6\times10^5$",
        rf"WT $\alpha$ = {alpha_low}$^\circ$ Re = $5.6\times10^5$",
        # rf"Wind Tunnel $\alpha$ = 4.75$^\circ$, Re = $5.6\times10^5$",
    ]
    colors = ["black", "blue", "blue", "red", "red"]
    linestyles = ["s-", "s-", "s--", "o-", "o--"]

    # ## Adding two more lines for corrected for side slip data
    # def correcting_for_sideslip(df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     [F_X_new]   [cos(β)  -sin(β)  0] [F_X_old]
    #     [F_Y_new] = [sin(β)   cos(β)  0] [F_Y_old]
    #     [F_Z_new]   [  0       0      1] [F_Z_old]
    #     """

    #     ## Grabbing sideslip array and converting to radians
    #     beta = np.deg2rad(df["sideslip"])
    #     # beta = np.deg2rad(10) * np.ones_like(df["sideslip"])

    #     ## Defining rotation matrix for each row
    #     def create_rotation_matrix(beta_angle):
    #         return np.array(
    #             [
    #                 [np.cos(beta_angle), np.sin(beta_angle), 0],
    #                 [-np.sin(beta_angle), np.cos(beta_angle), 0],
    #                 [0, 0, 1],
    #             ]
    #         )

    #     # Create arrays for forces and moments
    #     forces = np.array([df["C_D"], df["C_S"], df["C_L"]]).T
    #     moments = np.array([df["C_pitch"], df["C_yaw"], df["C_roll"]]).T

    #     # Initialize arrays for corrected forces and moments
    #     corrected_forces = np.zeros_like(forces)
    #     corrected_moments = np.zeros_like(moments)

    #     # Apply rotation to each row
    #     for i in range(len(df)):
    #         R = create_rotation_matrix(beta[i])
    #         corrected_forces[i] = R @ forces[i]
    #         corrected_moments[i] = R @ moments[i]

    #     # Update dataframe with corrected values
    #     df["C_D"], df["C_S"], df["C_L"] = corrected_forces.T
    #     df["C_pitch"], df["C_yaw"], df["C_roll"] = corrected_moments.T

    #     return df

    # data_WT_beta_re_56e4_alpha_high_corrected = correcting_for_sideslip(
    #     data_WT_beta_re_56e4_alpha_high.copy()
    # )
    # data_WT_beta_re_56e4_alpha_low_corrected = correcting_for_sideslip(
    #     data_WT_beta_re_56e4_alpha_low.copy()
    # )
    # data_frame_list.append(data_WT_beta_re_56e4_alpha_high_corrected)
    # data_frame_list.append(data_WT_beta_re_56e4_alpha_low_corrected)
    # labels.append(rf"WT CORRECTED $\alpha$ = {alpha_high}$^\circ$ Re = $5.6\times10^5$")
    # labels.append(rf"WT CORRECTED $\alpha$ = {alpha_low}$^\circ$ Re = $5.6\times10^5$")
    # colors.append("green")
    # colors.append("green")
    # linestyles.append("o-")
    # linestyles.append("o--")

    # Plot CL, CD, and CS curves in subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    for i, (data_frame, label, color, linestyle) in enumerate(
        zip(data_frame_list, labels, colors, linestyles)
    ):
        if i == 0:  # if Lebesque
            axs[0].plot(
                data_frame["beta"],
                data_frame["CL"],
                linestyle,
                label=label,
                color=color,
            )
            axs[1].plot(
                data_frame["beta"],
                data_frame["CD"],
                linestyle,
                label=label,
                color=color,
            )
            axs[2].plot(
                data_frame["beta"],
                data_frame["CS"] / ratio_projected_area_to_side_area,
                linestyle,
                label=label,
                color=color,
            )
        elif i == 1 or i == 2:  # if VSM
            # axs[0].plot(data_frame["beta"], data_frame["CL"], linestyle, label=label)
            # axs[1].plot(data_frame["beta"], data_frame["CD"], linestyle, label=label)
            # axs[2].plot(data_frame["beta"], data_frame["CS"], linestyle, label=label)

            # Adding stall-corrected values
            axs[0].plot(
                data_frame["beta"],
                data_frame["CL_stall"],
                linestyle,
                label=label,
                color=color,
            )
            axs[1].plot(
                data_frame["beta"],
                data_frame["CD_stall"],
                linestyle,
                label=label,
                color=color,
            )
            axs[2].plot(
                data_frame["beta"],
                data_frame["CS_stall"],
                linestyle,
                label=label,
                color=color,
            )
        else:  # if windtunnel

            axs[0].plot(
                data_frame["sideslip"],
                data_frame["C_L"],
                linestyle,
                label=label,
                color=color,
            )
            axs[1].plot(
                data_frame["sideslip"],
                data_frame["C_D"],
                linestyle,
                label=label,
                color=color,
            )
            axs[2].plot(
                data_frame["sideslip"],
                data_frame["C_S"],
                linestyle,
                label=label,
                color=color,
            )

    # Formatting the axis
    axs[0].set_xlabel(x_axis_labels["beta"])
    axs[0].set_ylabel(y_axis_labels["CL"])
    axs[0].grid()
    axs[0].set_xlim(0, 20)
    axs[0].set_ylim(0.35, 1.1)
    axs[0].legend(loc="lower left")

    axs[1].set_xlabel(x_axis_labels["beta"])
    axs[1].set_ylabel(y_axis_labels["CD"])
    axs[1].grid()
    axs[1].set_xlim(0, 20)
    axs[1].set_ylim(0.0, 0.25)

    axs[2].set_xlabel(x_axis_labels["beta"])
    axs[2].set_ylabel(y_axis_labels["CS"])
    axs[2].grid()
    axs[2].set_xlim(0, 20)
    axs[2].set_ylim(-0.05, 0.6)

    # Plotting and saving
    plt.tight_layout()
    file_name = "literature_polars_beta"
    saving_pdf_and_pdf_tex(results_dir, file_name)


def main(results_dir, project_dir):

    fontsize = 18
    figsize = (20, 6)
    plotting_polars_alpha(
        project_dir,
        results_dir,
        figsize,
        fontsize,
    )

    figsize = (20, 6)
    plotting_polars_beta(
        project_dir,
        results_dir,
        figsize,
        fontsize,
        ratio_projected_area_to_side_area=3.7,
    )


if __name__ == "__main__":
    from utils import project_dir

    results_dir = Path(project_dir) / "results"
    main(results_dir, project_dir)
