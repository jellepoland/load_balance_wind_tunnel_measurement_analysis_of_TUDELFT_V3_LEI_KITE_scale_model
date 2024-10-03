import matplotlib as mpl
import os
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm


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


root_dir = defining_root_dir()
print(f"Root directory: {root_dir}")

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
