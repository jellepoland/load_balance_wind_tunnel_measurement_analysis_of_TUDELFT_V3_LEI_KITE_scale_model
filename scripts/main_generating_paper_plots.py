from pathlib import Path
import matplotlib.pyplot as plt
import reynolds_variation_plots
import literature_comparison
import uncertainty_table
import uncertainty_boxplots
import sensor_drift
import zigzag
from settings import *


def main():

    # results_path = Path(root_dir) / "results"
    results_path = Path(
        "/home/jellepoland/ownCloud/phd/latex_documents/WES24_KITE_WindTunnel/Images"
    )
    reynolds_variation_plots.main(results_path, root_dir)
    literature_comparison.main(results_path, root_dir)
    zigzag.main(results_path, root_dir)
    uncertainty_boxplots.main(results_path, root_dir)
    uncertainty_table.main(root_dir)
    sensor_drift.main(root_dir)

    print(
        f"\n--> New plots generated in saved inside the latex-folder, located at: \n {results_path}"
    )


if __name__ == "__main__":
    main()
