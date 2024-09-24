import matplotlib.pyplot as plt
import subprocess
import os


# Function to export figure as pdf_tex using Inkscape
def savepdf_tex(fig, name, **kwargs):
    fig.savefig("temp.pdf", format="pdf", **kwargs)

    incmd = [
        "inkscape",
        "temp.pdf",
        "--export-pdf={}.pdf".format(name),
        "--export-latex",
    ]

    try:
        subprocess.check_output(incmd)
    except subprocess.CalledProcessError as e:
        print(f"Error during Inkscape conversion: {e}")
        return
    finally:
        os.remove("temp.pdf")  # Ensure the temp file is always removed


if __name__ == "__main__":
    # Sample Plot
    plt.bar(x=[1, 2], height=[3, 4], label="Bar")
    plt.legend()
    plt.xlabel("Label")
    plt.title("title")

    # Save the current figure using Inkscape
    savepdf_tex(plt.gcf(), "test_bar_plot")
