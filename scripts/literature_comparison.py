import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from settings import (
    root_dir,
    saving_pdf_and_pdf_tex,
    x_axis_labels,
    y_axis_labels,
    reduce_df_by_parameter_mean_and_std,
)


# def loading_data_vsm_old_alpha() -> tuple:
#     # cayon data
#     cay_alpha_cd = [
#         -5,
#         -2.972027972027972,
#         -1.013986013986014,
#         0.9440559440559442,
#         3.041958041958042,
#         5.06993006993007,
#         6.993006993006993,
#         8.986013986013987,
#         10.97902097902098,
#         13.041958041958043,
#         15.034965034965037,
#         17.02797202797203,
#         19.020979020979023,
#         20.944055944055947,
#         22.972027972027973,
#         25.034965034965037,
#         26.993006993006993,
#         28.986013986013987,
#     ]
#     cay_cd_aoa = [
#         0.051436265709156194,
#         0.046768402154398564,
#         0.04640933572710952,
#         0.04820466786355476,
#         0.053949730700179534,
#         0.06149012567324955,
#         0.07333931777378816,
#         0.08770197486535009,
#         0.10350089766606824,
#         0.12217235188509876,
#         0.13545780969479354,
#         0.14946140035906644,
#         0.16885098743267507,
#         0.18752244165170556,
#         0.20439856373429088,
#         0.2151705565529623,
#         0.2209156193895871,
#         0.22594254937163377,
#     ]

#     cay_alpha_cl = [
#         -5.042321644498187,
#         -3.0834340991535676,
#         -1.0519951632406288,
#         0.9794437726723096,
#         2.902055622732769,
#         4.969770253929868,
#         6.964933494558647,
#         8.996372430471585,
#         10.991535671100364,
#         12.914147521160823,
#         15.054413542926241,
#         16.94074969770254,
#         18.93591293833132,
#         21.00362356952842,
#         22.962515114873035,
#         25.030229746070134,
#         26.989117291414754,
#         28.984280532043535,
#     ]
#     cay_cl_aoa = [
#         -0.27040816326530615,
#         -0.08418367346938777,
#         0.09948979591836732,
#         0.28316326530612246,
#         0.4515306122448979,
#         0.6173469387755102,
#         0.7806122448979591,
#         0.9234693877551019,
#         1.0637755102040816,
#         1.193877551020408,
#         1.204081632653061,
#         1.107142857142857,
#         1.0612244897959182,
#         1.0153061224489794,
#         0.982142857142857,
#         0.9642857142857142,
#         0.9489795918367345,
#         0.9413265306122447,
#     ]

#     cay_alpha_clcd = [
#         -5.063583815028902,
#         -3.0173410404624272,
#         -1.0404624277456647,
#         1.0404624277456647,
#         2.982658959537572,
#         4.99421965317919,
#         7.005780346820808,
#         8.982658959537572,
#         10.959537572254334,
#         13.005780346820808,
#         14.947976878612716,
#         16.99421965317919,
#         19.005780346820806,
#         20.947976878612714,
#         22.959537572254334,
#         24.97109826589595,
#         26.98265895953757,
#         28.959537572254334,
#     ]
#     cay_clcd_aoa = [
#         -5.198675496688741,
#         -1.7963576158940397,
#         2.152317880794701,
#         5.778145695364238,
#         8.410596026490065,
#         9.975165562913906,
#         10.596026490066224,
#         10.571192052980132,
#         10.298013245033111,
#         9.776490066225165,
#         8.857615894039734,
#         7.367549668874171,
#         6.299668874172184,
#         5.455298013245033,
#         4.784768211920529,
#         4.461920529801324,
#         4.312913907284768,
#         4.139072847682119,
#     ]

#     cay_cdcl_cd = [
#         0.05204545454545455,
#         0.0475,
#         0.04613636363636364,
#         0.04840909090909091,
#         0.05363636363636364,
#         0.06227272727272727,
#         0.07386363636363637,
#         0.08727272727272728,
#         0.10386363636363637,
#         0.12204545454545455,
#         0.13636363636363635,
#         0.15000000000000002,
#         0.1684090909090909,
#         0.18818181818181817,
#         0.20500000000000002,
#         0.21454545454545454,
#         0.22136363636363637,
#         0.22613636363636364,
#     ]
#     cay_cdcl_cl = [
#         -0.27163461538461536,
#         -0.08173076923076922,
#         0.09855769230769235,
#         0.27884615384615385,
#         0.451923076923077,
#         0.6201923076923077,
#         0.778846153846154,
#         0.9254807692307694,
#         1.0649038461538463,
#         1.1947115384615385,
#         1.2115384615384617,
#         1.1129807692307694,
#         1.0625,
#         1.0192307692307694,
#         0.9783653846153846,
#         0.96875,
#         0.9519230769230771,
#         0.9399038461538463,
#     ]
#     return [
#         cay_alpha_cd,
#         cay_cd_aoa,
#         cay_alpha_cl,
#         cay_cl_aoa,
#         cay_alpha_clcd,
#         cay_clcd_aoa,
#         cay_cdcl_cd,
#         cay_cdcl_cl,
#     ]


# def loading_data_vsm_old_beta():

#     # cayon
#     cay_beta_cl = [
#         -0.0262582056892779,
#         1.9693654266958425,
#         3.9912472647702404,
#         6,
#         8.00875273522976,
#         10.017505470459518,
#         11.986870897155361,
#     ]
#     cay_cl = [
#         1.1303687635574837,
#         1.129067245119306,
#         1.120824295010846,
#         1.1078091106290673,
#         1.0704989154013016,
#         0.9837310195227766,
#         0.872234273318872,
#     ]

#     cay_beta_cd = [
#         -0.012578616352201259,
#         1.9874213836477987,
#         3.9874213836477987,
#         6,
#         7.987421383647799,
#         10.012578616352203,
#         12.012578616352203,
#     ]
#     cay_cd = [
#         0.1127077747989276,
#         0.11324396782841822,
#         0.11506702412868632,
#         0.11764075067024128,
#         0.11978552278820374,
#         0.12053619302949062,
#         0.12525469168900805,
#     ]

#     cay_beta_cs = [
#         -0.0121212121212122,
#         1.9878787878787878,
#         4,
#         5.975757575757576,
#         7.963636363636364,
#         9.975757575757576,
#         11.975757575757576,
#     ]
#     cay_cs = (
#         np.array(
#             [
#                 -0.002366863905325444,
#                 0.19053254437869824,
#                 0.3810650887573965,
#                 0.5633136094674557,
#                 0.7242603550295859,
#                 0.7597633136094675,
#                 0.7100591715976331,
#             ]
#         )
#         / 3.7
#     )

#     cay_beta_clcd = [
#         0.011730205278592375,
#         2.005865102639296,
#         4,
#         6.029325513196481,
#         8,
#         10.005865102639296,
#         12,
#     ]
#     cay_clcd = [
#         10.028520499108733,
#         9.964349376114082,
#         9.74331550802139,
#         9.401069518716577,
#         8.93048128342246,
#         8.153297682709447,
#         6.948306595365419,
#     ]
#     return [
#         cay_beta_cl,
#         cay_cl,
#         cay_beta_cd,
#         cay_cd,
#         cay_beta_cs,
#         cay_cs,
#         cay_beta_clcd,
#         cay_clcd,
#     ]


# def reduce_df_by_parameter_mean_and_std(
#     df: pd.DataFrame, parameter: str
# ) -> pd.DataFrame:
#     """
#     Reduces a dataframe to unique values of a parameter, averaging specified columns
#     and adding standard deviations for coefficients.

#     Parameters:
#     df (pandas.DataFrame): The input dataframe
#     parameter (str): Either 'aoa_kite' or 'sideslip'

#     Returns:
#     pandas.DataFrame: Reduced dataframe with averages and coefficient standard deviations
#     """
#     # All columns to average
#     columns_to_average = [
#         "C_L",
#         "C_S",
#         "C_D",
#         "C_roll",
#         "C_pitch",
#         "C_yaw",
#         "F_X_raw",
#         "F_Y_raw",
#         "F_Z_raw",
#         "M_X_raw",
#         "M_Y_raw",
#         "M_Z_raw",
#         "Rey",
#     ]

#     if parameter == "aoa_kite":
#         columns_to_average += ["sideslip"]
#     elif parameter == "sideslip":
#         columns_to_average += ["aoa_kite"]
#     else:
#         raise ValueError("Invalid parameter")

#     # Calculate means
#     mean_df = df.groupby(parameter)[columns_to_average].mean()

#     # Coefficient columns that also need standard deviation
#     coef_columns = ["C_L", "C_S", "C_D", "C_roll", "C_pitch", "C_yaw"]

#     # Calculate standard deviations for coefficients
#     std_df = df.groupby(parameter)[coef_columns].std()

#     # Rename standard deviation columns
#     std_df.columns = [f"{col}_std" for col in std_df.columns]

#     # Combine mean and standard deviation dataframes
#     result_df = pd.concat([mean_df, std_df], axis=1).reset_index()

#     return result_df


def plotting_polars_alpha(
    root_dir: str,
    results_dir: str,
    figsize: tuple,
    fontsize: int,
):

    vw = 20
    beta_value = 0
    folder_name = f"beta_{beta_value}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(root_dir) / "processed_data" / "normal_csv" / folder_name / file_name
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
        Path(root_dir)
        / "processed_data"
        / "literature_comparison"
        / "VSM_results_alpha_sweep_Rey_5.6.csv"
    )
    data_VSM_alpha_re_56e4 = pd.read_csv(path_to_csv_VSM_alpha_re_56e4)
    # Lebesque
    path_to_csv_lebesque_Rey_100e4 = (
        Path(root_dir)
        / "processed_data"
        / "literature_comparison"
        / "V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv"
    )
    data_lebesque_alpha_re_100e4 = pd.read_csv(path_to_csv_lebesque_Rey_100e4)
    path_to_csv_lebesque_Rey_300e4 = (
        Path(root_dir)
        / "processed_data"
        / "literature_comparison"
        / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
    )
    data_lebesque_alpha_re_300e4 = pd.read_csv(path_to_csv_lebesque_Rey_300e4)

    data_frame_list = [
        # data_windtunnel_alpha_re_42e4,
        # data_lebesque_alpha_re_300e4,
        data_lebesque_alpha_re_100e4,
        data_VSM_alpha_re_56e4,
        data_windtunnel_alpha_re_56e4,
    ]
    labels = [
        # rf"Re = $4.2\times10^5$ Wind Tunnel",
        # rf"Re = $30\times10^5$ CFD (Lebesque, 2022)",
        rf"CFD Re = $10\times10^5$",
        rf"VSM Re = $5.6\times10^5$",
        rf"WT Re = $5.6\times10^5$",
    ]

    colors = ["black", "blue", "red"]
    linestyles = ["s-", "s-", "o-"]

    # Plot CL, CD, and CS curves in subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    for i, (data_frame, label, color, linestyle) in enumerate(
        zip(data_frame_list, labels, colors, linestyles)
    ):
        if i == 0:  # if Lebesque
            linestyle = "s--"
            axs[0].plot(
                data_frame["aoa"],
                data_frame["CL"],
                linestyle,
                label=label,
                color=color,
            )
            axs[1].plot(
                data_frame["aoa"],
                data_frame["CD"],
                linestyle,
                label=label,
                color=color,
            )
            axs[2].plot(
                data_frame["aoa"],
                data_frame["CL"] / data_frame["CD"],
                linestyle,
                label=label,
                color=color,
            )

        elif i == 1:

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
            )
            axs[1].plot(
                data_frame["aoa"],
                data_frame["CD_stall"],
                linestyle,
                label=label,
                color=color,
            )
            axs[2].plot(
                data_frame["aoa"],
                data_frame["CL_stall"] / data_frame["CD_stall"],
                linestyle,
                label=label,
                color=color,
            )

        if i == 2:
            axs[0].plot(
                data_frame["aoa_kite"],
                data_frame["C_L"],
                linestyle,
                label=label,
                color=color,
            )
            axs[1].plot(
                data_frame["aoa_kite"],
                data_frame["C_D"],
                linestyle,
                label=label,
                color=color,
            )
            axs[2].plot(
                data_frame["aoa_kite"],
                data_frame["C_L"] / data_frame["C_D"],
                linestyle,
                label=label,
                color=color,
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
    root_dir: str,
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
        Path(root_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)
    data_WT_beta_re_56e4_alpha_high = reduce_df_by_parameter_mean_and_std(
        df_all_values, "sideslip"
    )

    folder_name = f"alpha_{alpha_low}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(root_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)
    data_WT_beta_re_56e4_alpha_low = reduce_df_by_parameter_mean_and_std(
        df_all_values, "sideslip"
    )

    # Load VSM data
    path_to_csv_VSM_beta_re_56e4_alpha_1195 = (
        Path(root_dir)
        / "processed_data"
        / "literature_comparison"
        / "VSM_results_beta_sweep_Rey_5.6_alpha_1195.csv"
    )
    data_VSM_beta_re_56e4_alpha_1195 = pd.read_csv(
        path_to_csv_VSM_beta_re_56e4_alpha_1195
    )

    # Load VSM data
    path_to_csv_VSM_beta_re_56e4_alpha_675 = (
        Path(root_dir)
        / "processed_data"
        / "literature_comparison"
        / "VSM_results_beta_sweep_Rey_5.6_alpha_675.csv"
    )
    data_VSM_beta_re_56e4_alpha_675 = pd.read_csv(
        path_to_csv_VSM_beta_re_56e4_alpha_675
    )

    # Load Lebesque data
    path_to_csv_lebesque_re_100e4_alpha_1195 = (
        Path(root_dir)
        / "processed_data"
        / "literature_comparison"
        / "V3_CL_CD_CS_RANS_Lebesque_2024_Rey_100e4_beta_sweep.csv"
    )
    data_lebesque_re_100e4_alpha_1195 = pd.read_csv(
        path_to_csv_lebesque_re_100e4_alpha_1195
    )
    # path_to_csv_lebesque_re_300e4 = (
    #     Path(root_dir)
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


def main(results_dir, root_dir):

    fontsize = 18
    figsize = (20, 6)
    plotting_polars_alpha(
        root_dir,
        results_dir,
        figsize,
        fontsize,
    )

    figsize = (20, 6)
    plotting_polars_beta(
        root_dir,
        results_dir,
        figsize,
        fontsize,
        ratio_projected_area_to_side_area=3.7,
    )


if __name__ == "__main__":
    from settings import root_dir

    results_dir = Path(root_dir) / "results"
    main(results_dir, root_dir)
