from reynolds_variation_plots import main as reynolds_variation_plots_main
from uncertainty_boxplots import main as uncertainty_boxplots_main
from literature_comparison import main as literature_comparison_main
from reynolds_variation_plots import defining_root_dir
from pathlib import Path


def main():
    root_dir = defining_root_dir()
    results_path = Path(root_dir) / "results"
    results_path = Path(
        "/home/jellepoland/ownCloud/phd/latex_documents/WES24_KITE_WindTunnel/Images"
    )

    reynolds_variation_plots_main(results_path, root_dir)
    uncertainty_boxplots_main(results_path, root_dir)
    literature_comparison_main(results_path, root_dir)

    print(
        f"\n--> New plots generated in saved inside the latex-folder, located at: \n {results_path}"
    )


if __name__ == "__main__":
    main()
