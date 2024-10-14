from pathlib import Path
import matplotlib.pyplot as plt
import plot_and_process_zigzag
import plot_literature_comparison
import plot_reynolds_variation
import plot_uncertainty_boxplots
import print_sensor_drift
import print_uncertainty_table
from utils import *


def main():

    results_path = Path(project_dir) / "results"
    # results_path = Path(
    #     "/home/jellepoland/ownCloud/phd/latex_documents/WES24_KITE_WindTunnel/Images"
    # )

    ## plotting
    plot_and_process_zigzag.main(results_path, project_dir)
    plot_literature_comparison.main(results_path, project_dir)
    plot_reynolds_variation.main(results_path, project_dir)
    plot_uncertainty_boxplots.main(results_path, project_dir)

    ## printing
    print_sensor_drift.main(project_dir)
    print_uncertainty_table.main(project_dir)
    print(
        f"\n--> New plots generated in saved inside the latex-folder, located at: \n {results_path}"
    )


if __name__ == "__main__":
    main()
