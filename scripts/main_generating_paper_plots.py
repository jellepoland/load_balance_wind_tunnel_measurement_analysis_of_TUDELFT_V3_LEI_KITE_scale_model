from pathlib import Path
import matplotlib.pyplot as plt
import reynolds_variation_plots
import literature_comparison
import uncertainty_table
import zigzag


def main():
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        }
    )

    root_dir = reynolds_variation_plots.defining_root_dir()
    results_path = Path(root_dir) / "results"
    results_path = Path(
        "/home/jellepoland/ownCloud/phd/latex_documents/WES24_KITE_WindTunnel/Images"
    )
    reynolds_variation_plots.main(results_path, root_dir)
    literature_comparison.main(results_path, root_dir)
    zigzag.main(results_path, root_dir)
    uncertainty_table.main(root_dir)

    print(
        f"\n--> New plots generated in saved inside the latex-folder, located at: \n {results_path}"
    )


if __name__ == "__main__":
    main()
