from pathlib import Path
import matplotlib.pyplot as plt
from load_balance_analysis import plot_and_process_zigzag
from load_balance_analysis import plot_literature_comparison
from load_balance_analysis import plot_reynolds_variation
from load_balance_analysis import plot_uncertainty_boxplots
from load_balance_analysis import print_sensor_drift
from load_balance_analysis import print_uncertainty_table
from load_balance_analysis import print_repeatability_uncertainty
from utils import *


def main():

    # results_path = Path(project_dir) / "results"
    results_path = Path(
        "/home/jellepoland/ownCloud/phd/latex_documents/WES24_KITE_WindTunnel/Images"
    )

    ## plotting
    plot_and_process_zigzag.main(results_path, project_dir)
    plot_literature_comparison.main(results_path, project_dir)
    plot_reynolds_variation.main(results_path, project_dir)
    plot_uncertainty_boxplots.main(results_path, project_dir)

    ## printing
    print_sensor_drift.main(project_dir)
    print_uncertainty_table.main(project_dir)
    print_repeatability_uncertainty.main(project_dir)
    print(f"\n--> New plots generated, and saved inside: \n {results_path}")


if __name__ == "__main__":
    main()
