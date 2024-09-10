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


def loading_data_vsm_old_alpha() -> tuple:
    # cayon data
    cay_alpha_cd = [
        -5,
        -2.972027972027972,
        -1.013986013986014,
        0.9440559440559442,
        3.041958041958042,
        5.06993006993007,
        6.993006993006993,
        8.986013986013987,
        10.97902097902098,
        13.041958041958043,
        15.034965034965037,
        17.02797202797203,
        19.020979020979023,
        20.944055944055947,
        22.972027972027973,
        25.034965034965037,
        26.993006993006993,
        28.986013986013987,
    ]
    cay_cd_aoa = [
        0.051436265709156194,
        0.046768402154398564,
        0.04640933572710952,
        0.04820466786355476,
        0.053949730700179534,
        0.06149012567324955,
        0.07333931777378816,
        0.08770197486535009,
        0.10350089766606824,
        0.12217235188509876,
        0.13545780969479354,
        0.14946140035906644,
        0.16885098743267507,
        0.18752244165170556,
        0.20439856373429088,
        0.2151705565529623,
        0.2209156193895871,
        0.22594254937163377,
    ]

    cay_alpha_cl = [
        -5.042321644498187,
        -3.0834340991535676,
        -1.0519951632406288,
        0.9794437726723096,
        2.902055622732769,
        4.969770253929868,
        6.964933494558647,
        8.996372430471585,
        10.991535671100364,
        12.914147521160823,
        15.054413542926241,
        16.94074969770254,
        18.93591293833132,
        21.00362756952842,
        22.962515114873035,
        25.030229746070134,
        26.989117291414754,
        28.984280532043535,
    ]
    cay_cl_aoa = [
        -0.27040816326530615,
        -0.08418367346938777,
        0.09948979591836732,
        0.28316326530612246,
        0.4515306122448979,
        0.6173469387755102,
        0.7806122448979591,
        0.9234693877551019,
        1.0637755102040816,
        1.193877551020408,
        1.204081632653061,
        1.107142857142857,
        1.0612244897959182,
        1.0153061224489794,
        0.982142857142857,
        0.9642857142857142,
        0.9489795918367345,
        0.9413265306122447,
    ]

    cay_alpha_clcd = [
        -5.063583815028902,
        -3.0173410404624272,
        -1.0404624277456647,
        1.0404624277456647,
        2.982658959537572,
        4.99421965317919,
        7.005780346820808,
        8.982658959537572,
        10.959537572254334,
        13.005780346820808,
        14.947976878612716,
        16.99421965317919,
        19.005780346820806,
        20.947976878612714,
        22.959537572254334,
        24.97109826589595,
        26.98265895953757,
        28.959537572254334,
    ]
    cay_clcd_aoa = [
        -5.198675496688741,
        -1.7963576158940397,
        2.152317880794701,
        5.778145695364238,
        8.410596026490065,
        9.975165562913906,
        10.596026490066224,
        10.571192052980132,
        10.298013245033111,
        9.776490066225165,
        8.857615894039734,
        7.367549668874171,
        6.299668874172184,
        5.455298013245033,
        4.784768211920529,
        4.461920529801324,
        4.312913907284768,
        4.139072847682119,
    ]

    cay_cdcl_cd = [
        0.05204545454545455,
        0.0475,
        0.04613636363636364,
        0.04840909090909091,
        0.05363636363636364,
        0.06227272727272727,
        0.07386363636363637,
        0.08727272727272728,
        0.10386363636363637,
        0.12204545454545455,
        0.13636363636363635,
        0.15000000000000002,
        0.1684090909090909,
        0.18818181818181817,
        0.20500000000000002,
        0.21454545454545454,
        0.22136363636363637,
        0.22613636363636364,
    ]
    cay_cdcl_cl = [
        -0.27163461538461536,
        -0.08173076923076922,
        0.09855769230769235,
        0.27884615384615385,
        0.451923076923077,
        0.6201923076923077,
        0.778846153846154,
        0.9254807692307694,
        1.0649038461538463,
        1.1947115384615385,
        1.2115384615384617,
        1.1129807692307694,
        1.0625,
        1.0192307692307694,
        0.9783653846153846,
        0.96875,
        0.9519230769230771,
        0.9399038461538463,
    ]
    return [
        cay_alpha_cd,
        cay_cd_aoa,
        cay_alpha_cl,
        cay_cl_aoa,
        cay_alpha_clcd,
        cay_clcd_aoa,
        cay_cdcl_cd,
        cay_cdcl_cl,
    ]


def loading_data_lebesque_beta():
    # compare non zero sideslip, lebesque
    leb_beta_cl = [0, 4.021482277121375, 8.004296455424276, 12.012889366272825]
    leb_cl = [
        1.0480851063829788,
        1.0340425531914894,
        0.994468085106383,
        0.7948936170212766,
    ]

    leb_beta_cd = [
        0.0032206119162641045,
        4.009661835748792,
        8.003220611916264,
        11.98389694041868,
    ]
    leb_cd = [
        0.11070208728652751,
        0.11886148007590132,
        0.14106261859582542,
        0.22910815939278936,
    ]

    leb_beta_cs = [0.012499999999999956, 4, 8, 11.9875]
    leb_cs = (
        np.array([0, 0.2528089887640449, 0.49438202247191004, 0.447191011235955]) / 3.7
    )

    leb_beta_clcd = [
        -0.002945508100147265,
        3.991163475699558,
        7.985272459499264,
        11.991163475699558,
    ]
    leb_clcd = [
        9.46641791044776,
        8.692164179104477,
        7.059701492537313,
        3.486940298507463,
    ]

    return [
        leb_beta_cl,
        leb_cl,
        leb_beta_cd,
        leb_cd,
        leb_beta_cs,
        leb_cs,
        leb_beta_clcd,
        leb_clcd,
    ]


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


def plotting_polars_alpha(
    root_dir: str,
    results_path: str,
    figsize: tuple,
    fontsize: int,
):

    # Wind tunnel
    path_to_csv = Path(root_dir) / "processed_data" / "stats_all.csv"
    df_windtunnel = pd.read_csv(path_to_csv)
    filtered_df = df_windtunnel.loc[df_windtunnel["sideslip"] == 0]
    filtered_df_grouped = filtered_df.groupby("vw")
    # data_windtunnel_alpha_re_42e4 = filtered_df_grouped.get_group(15)
    data_windtunnel_alpha_re_56e4 = filtered_df_grouped.get_group(20)
    # VSM
    path_to_csv_VSM_alpha_re_56e4 = (
        Path(root_dir) / "processed_data" / "VSM_results_alpha_sweep_Rey_5.6.csv"
    )
    data_VSM_alpha_re_56e4 = pd.read_csv(path_to_csv_VSM_alpha_re_56e4)
    # Lebesque
    path_to_csv_lebesque_Rey_100e4 = (
        Path(root_dir) / "processed_data" / "V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv"
    )
    data_lebesque_alpha_re_100e4 = pd.read_csv(path_to_csv_lebesque_Rey_100e4)
    path_to_csv_lebesque_Rey_300e4 = (
        Path(root_dir) / "processed_data" / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
    )
    data_lebesque_alpha_re_300e4 = pd.read_csv(path_to_csv_lebesque_Rey_300e4)

    data_frame_list = [
        # data_windtunnel_alpha_re_42e4,
        data_lebesque_alpha_re_300e4,
        data_lebesque_alpha_re_100e4,
        data_VSM_alpha_re_56e4,
        data_windtunnel_alpha_re_56e4,
    ]
    labels = [
        # rf"Re = $4.2\cdot10^5$ Wind Tunnel",
        rf"Re = $30\cdot10^5$ CFD (Lebesque, 2022)",
        rf"Re = $10\cdot10^5$ CFD (Lebesque, 2022)",
        rf"Re = $5.6\cdot10^5$ VSM",
        rf"Re = $5.6\cdot10^5$ Wind Tunnel",
    ]

    # Plot cl and cd curves in subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    for i, (data_frame, label) in enumerate(zip(data_frame_list, labels)):
        if i == 3:
            linestyle = "o--"
            axs[0].plot(
                data_frame["aoa_kite"], data_frame["C_L"], linestyle, label=label
            )
            axs[1].plot(
                data_frame["aoa_kite"], data_frame["C_D"], linestyle, label=label
            )
            axs[2].plot(
                data_frame["aoa_kite"],
                data_frame["C_L"] / data_frame["C_D"],
                linestyle,
                label=label,
            )

        elif i == 2:
            linestyle = "s--"
            axs[0].plot(data_frame["aoa"], data_frame["CL"], linestyle, label=label)
            axs[1].plot(data_frame["aoa"], data_frame["CD"], linestyle, label=label)
            axs[2].plot(
                data_frame["aoa"],
                data_frame["CL"] / data_frame["CD"],
                linestyle,
                label=label,
            )
            linestyle = "s--"
            axs[0].plot(
                data_frame["aoa"],
                data_frame["CL_stall"],
                linestyle,
                label=label + " STALL",
            )
            axs[1].plot(
                data_frame["aoa"],
                data_frame["CD_stall"],
                linestyle,
                label=label + " STALL",
            )
            axs[2].plot(
                data_frame["aoa"],
                data_frame["CL_stall"] / data_frame["CD_stall"],
                linestyle,
                label=label,
            )
        else:  # if Lebesque
            linestyle = "s--"
            axs[0].plot(data_frame["aoa"], data_frame["CL"], linestyle, label=label)
            axs[1].plot(data_frame["aoa"], data_frame["CD"], linestyle, label=label)
            axs[2].plot(
                data_frame["aoa"],
                data_frame["CL"] / data_frame["CD"],
                linestyle,
                label=label,
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
    # label_cay = r"Re = $3\cdot10^6$ (Cayon, 2022)"
    # axs[0, 0].plot(cay_alpha_cl, cay_cl_aoa, "o--", label=label_cay)
    # axs[0, 1].plot(cay_alpha_cd, cay_cd_aoa, "o--", label=label_cay)
    # axs[1, 0].plot(cay_alpha_clcd, cay_clcd_aoa, "o--", label=label_cay)
    # axs[1, 1].plot(cay_cdcl_cd, cay_cdcl_cl, "o--", label=label_cay)

    # formatting axis
    axs[0].set_xlabel(r"$\alpha$ [$^o$]", fontsize=fontsize)
    axs[0].set_ylabel(r"$C_L$ [-]", fontsize=fontsize)
    axs[0].set_title("Lift Coefficient")
    axs[0].set_xlim(-12.65, 24)
    axs[0].grid()
    # axs[0].set_xlim([-5, 24])
    axs[0].set_ylim(-1.2, 1.5)
    axs[0].legend(loc="lower right")

    axs[1].set_xlabel(r"$\alpha$ [$^o$]", fontsize=fontsize)
    axs[1].set_ylabel(r"$C_D$ [-]", fontsize=fontsize)
    axs[1].set_title("Drag Coefficient")
    axs[1].grid()
    axs[1].set_xlim(-12.65, 24)
    # axs[1].set_xlim([-5, 24])
    axs[1].set_ylim(0, 0.5)
    # axs[1].legend(loc="upper left")

    axs[2].set_xlabel(r"$\alpha$ [$^o$]", fontsize=fontsize)
    axs[2].set_ylabel(r"$L/D$ [-]", fontsize=fontsize)
    axs[2].set_title("Lift/drag ratio")
    axs[2].grid()
    axs[2].set_xlim(-12.65, 24)
    # axs[2].set_xlim([-5, 24])
    axs[2].set_ylim(-11, 11)

    # plotting and saving
    plt.tight_layout()
    plot_filepath = Path(results_path) / f"literature_polars_alpha.pdf"
    plt.savefig(plot_filepath)


def plotting_polars_beta(
    root_dir: str,
    results_path: str,
    figsize: tuple,
    fontsize: int,
):

    # Load Wind tunnel data
    path_to_csv = Path(root_dir) / "processed_data" / "stats_all.csv"
    df_windtunnel = pd.read_csv(path_to_csv)
    filtered_df = df_windtunnel.loc[df_windtunnel["aoa_kite"] == 11.95]
    filtered_df_grouped = filtered_df.groupby("vw")
    data_windtunnel_beta_re_42e4 = filtered_df_grouped.get_group(15)
    data_windtunnel_beta_re_56e4 = filtered_df_grouped.get_group(20)

    # Load VSM data
    path_to_csv_VSM_beta_re_56e4 = (
        Path(root_dir) / "processed_data" / "VSM_results_beta_sweep_Rey_5.6.csv"
    )
    data_VSM_beta_re_56e4 = pd.read_csv(path_to_csv_VSM_beta_re_56e4)

    # Load Lebesque's data
    (
        leb_beta_cl,
        leb_cl,
        leb_beta_cd,
        leb_cd,
        leb_beta_cs,
        leb_cs,
        leb_beta_clcd,
        leb_clcd,
    ) = loading_data_lebesque_beta()

    data_frame_list = [
        data_VSM_beta_re_56e4,
        # data_windtunnel_beta_re_42e4,
        data_windtunnel_beta_re_56e4,
    ]
    labels = [
        rf"Re = $5.6\cdot10^5$ VSM",
        # rf"Re = $4.2\cdot10^5$ Wind Tunnel",
        rf"Re = $5.6\cdot10^5$ Wind Tunnel",
    ]

    # Plot CL, CD, and CS curves in subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Adding Lebesque's data first
    label_lebesque = [rf"Re = $10\cdot10^5$ CFD (Lebesque, 2022)"]
    axs[0].plot(leb_beta_cl, leb_cl, "s--", label=label_lebesque[0])
    axs[1].plot(leb_beta_cd, leb_cd, "s--", label=label_lebesque[0])
    axs[2].plot(leb_beta_cs, leb_cs, "s--", label=label_lebesque[0])

    for i, (data_frame, label) in enumerate(zip(data_frame_list, labels)):
        if i == 1:  # if windtunnel
            linestyle = "o--"
            axs[0].plot(
                data_frame["sideslip"], data_frame["C_L"], linestyle, label=label
            )
            axs[1].plot(
                data_frame["sideslip"], data_frame["C_D"], linestyle, label=label
            )
            axs[2].plot(
                data_frame["sideslip"], data_frame["C_S"], linestyle, label=label
            )
        elif i == 0:  # if VSM
            linestyle = "s--"
            axs[0].plot(data_frame["beta"], data_frame["CL"], linestyle, label=label)
            axs[1].plot(data_frame["beta"], data_frame["CD"], linestyle, label=label)
            axs[2].plot(data_frame["beta"], data_frame["CS"], linestyle, label=label)

            # Adding stall-corrected values
            axs[0].plot(
                data_frame["beta"],
                data_frame["CL_stall"],
                linestyle,
                label=label + " STALL",
            )
            axs[1].plot(
                data_frame["beta"],
                data_frame["CD_stall"],
                linestyle,
                label=label + " STALL",
            )
            axs[2].plot(
                data_frame["beta"],
                data_frame["CS_stall"],
                linestyle,
                label=label + " STALL",
            )

    # Formatting the axis
    axs[0].set_xlabel(r"$\beta$ [$^o$]", fontsize=fontsize)
    axs[0].set_ylabel(r"$C_L$ [-]", fontsize=fontsize)
    axs[0].set_title("Lift Coefficient")
    axs[0].grid()
    axs[0].set_xlim(0, 20)
    axs[0].set_ylim(0.3, 1.2)
    axs[0].legend()

    axs[1].set_xlabel(r"$\beta$ [$^o$]", fontsize=fontsize)
    axs[1].set_ylabel(r"$C_D$ [-]", fontsize=fontsize)
    axs[1].set_title("Drag Coefficient")
    axs[1].grid()
    axs[1].set_xlim(0, 20)
    axs[1].set_ylim(0.05, 0.25)

    axs[2].set_xlabel(r"$\beta$ [$^o$]", fontsize=fontsize)
    axs[2].set_ylabel(r"$C_S$ [-]", fontsize=fontsize)
    axs[2].set_title("Side Force Coefficient")
    axs[2].grid()
    axs[2].set_xlim(0, 20)
    axs[2].set_ylim(-0.05, 0.4)

    # Plotting and saving
    plt.tight_layout()
    plot_filepath = Path(results_path) / "literature_polars_beta.pdf"
    plt.savefig(plot_filepath)


def main(results_path, root_dir):

    # defining some plot specifics
    plt.rcParams.update({"font.size": 14})
    fontsize = 18

    figsize = (20, 6)
    plotting_polars_alpha(
        root_dir,
        results_path,
        figsize,
        fontsize,
    )

    figsize = (20, 6)
    plotting_polars_beta(
        root_dir,
        results_path,
        figsize,
        fontsize,
    )


if __name__ == "__main__":
    root_dir = defining_root_dir()
    results_path = Path(root_dir) / "results"
    main(results_path, root_dir)
