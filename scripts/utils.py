from pathlib import Path
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm


def reduce_df_by_parameter_mean_and_std(
    df: pd.DataFrame, parameter: str
) -> pd.DataFrame:
    """
    Reduces a dataframe to unique values of a parameter, averaging specified columns
    and adding standard deviations for coefficients.

    Parameters:
    df (pandas.DataFrame): The input dataframe
    parameter (str): Either 'aoa_kite' or 'sideslip'

    Returns:
    pandas.DataFrame: Reduced dataframe with averages and coefficient standard deviations
    """
    # All columns to average
    columns_to_average = [
        "C_L",
        "C_S",
        "C_D",
        "C_roll",
        "C_pitch",
        "C_yaw",
        # "F_X_raw",
        # "F_Y_raw",
        # "F_Z_raw",
        # "M_X_raw",
        # "M_Y_raw",
        # "M_Z_raw",
        "Rey",
        "vw",
    ]

    if parameter == "aoa_kite":
        columns_to_average += ["sideslip"]
    elif parameter == "sideslip":
        columns_to_average += ["aoa_kite"]
    else:
        raise ValueError("Invalid parameter")

    # Calculate means
    mean_df = df.groupby(parameter)[columns_to_average].mean()

    # Coefficient columns that also need standard deviation
    coef_columns = ["C_L", "C_S", "C_D", "C_roll", "C_pitch", "C_yaw"]

    # Calculate standard deviations for coefficients
    std_df = df.groupby(parameter)[coef_columns].std()

    # Rename standard deviation columns
    std_df.columns = [f"{col}_std" for col in std_df.columns]

    # Combine mean and standard deviation dataframes
    result_df = pd.concat([mean_df, std_df], axis=1).reset_index()

    # Round the velocities to 0 decimal places
    result_df["vw"] = result_df["vw"].round(0)

    return result_df


mpl.rcParams["font.family"] = "Open Sans"
mpl.rcParams.update({"font.size": 14})
mpl.rcParams["figure.figsize"] = 10, 5.625
mpl.rc("xtick", labelsize=13)
mpl.rc("ytick", labelsize=13)
mpl.rcParams["pdf.fonttype"] = 42  # Output Type 3 (Type3) or Type 42(TrueType)

# disable outline paths for inkscape > PDF+Latex
# important: comment out all other local font settings
mpl.rcParams["svg.fonttype"] = "none"

# Path to the directory where your fonts are installed
font_dir = "/home/jellepoland/.local/share/fonts/"

# Add each font in the directory
for font_file in os.listdir(font_dir):
    if font_file.endswith(".ttf") or font_file.endswith(".otf"):
        fm.fontManager.addfont(os.path.join(font_dir, font_file))


def saving_pdf_and_pdf_tex(results_dir: str, filename: str):
    plt.savefig(Path(results_dir) / f"{filename}.pdf")
    plt.savefig(Path(results_dir) / "pdf_tex" / f"{filename}.pdf")
    plt.savefig(Path(results_dir) / "pdf_tex" / f"{filename}.pdf_tex", format="pgf")
    plt.close()


x_axis_labels = {
    "alpha": r"$\alpha$ [$^\circ$]",
    "beta": r"$\beta$ [$^\circ$]",
    "Re": r"Re $\times 10^5$ [-]",
}

y_axis_labels = {
    "CL": r"$C_{\mathrm{L}}$ [-]",
    "CD": r"$C_{\mathrm{D}}$ [-]",
    "CS": r"$C_{\mathrm{S}}$ [-]",
    "CMx": r"$C_{\mathrm{M,x}}$ [-]",
    "CMy": r"$C_{\mathrm{M,y}}$ [-]",
    "CMz": r"$C_{\mathrm{M,z}}$ [-]",
    "L/D": r"$L/D$ [-]",
}

project_dir = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    print(f"project_dir: {project_dir}")
    print(f"\nlabel_x:")
    for label_x, label_x_item in zip(x_axis_labels, x_axis_labels.items()):
        print(f"{label_x}, {label_x_item}")
    print(f"\nlabel_y:")
    for label_y, label_y_item in zip(y_axis_labels, y_axis_labels.items()):
        print(f"{label_y}, {label_y_item}")