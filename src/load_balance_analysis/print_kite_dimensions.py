from tabulate import tabulate
import pandas as pd
from pathlib import Path
from load_balance_analysis.functions_utils import project_dir, save_latex_table


def main(project_dir: str):
    # Table 1: Dimensions Data
    dimensions_data = [
        ["Reference chord", "0.396 m", "0.395 m", "c_ref"],
        ["Height", "0.462 m", "0.462 m", "h"],
        ["Width", "1.277 m", "1.278 m", "w"],
        ["Mass", "-", "7.965 kg", "m"],
        ["Flat surface area", "0.59 m²", "-", "S"],
        ["Projected surface area", "0.46 m²", "-", "A"],
        ["Projected frontal area", "0.2 m²", "-", "A_f"],
    ]
    dimensions_headers = ["Property", "CAD", "1:6.5 Measured", "Symbol"]
    df_dimensions = pd.DataFrame(dimensions_data, columns=dimensions_headers)
    print(f"\n--- Dimensions Table ---\n")
    print(df_dimensions)

    # Save the LaTeX table
    save_latex_table(
        df_dimensions, Path(project_dir) / "results" / "tables" / "dimensions_table.tex"
    )

    # Table 2: Parameter Ranges Data
    parameter_data = [
        [
            r"Angle of attack $\alpha$ [\unit{\degree}]",
            "-12.7, -7.2, -3.0, -2.2, 2.3, 4.8, 6.8, 8.8, 10.9, 11.0, 11.9, 12.8, 14.0, 15.8, 17.9, 19.8, 22.6, 24.0",
        ],
        [
            r"Inflow speed $U_{\infty}$ [\unit{m/s}]",
            "5.0, 10.0, 15.0, 20.0, 25.0",
        ],
        [
            "Reynolds Number Re [−] $\times 10^5$",
            "1.4, 2.8, 4.2, 5.6, 6.9",
        ],
        [
            r"Side slip $\beta$ [\unit{\degree}]",
            "-20.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0",
        ],
    ]
    parameter_headers = ["Parameter", "Range"]
    df_parameters = pd.DataFrame(parameter_data, columns=parameter_headers)

    # Save the LaTeX table
    save_latex_table(
        df_parameters, Path(project_dir) / "results" / "tables" / "parameter_table.tex"
    )
    print(f"\n--- Parameter Ranges Table ---\n")
    print(df_parameters)


# Main Function
if __name__ == "__main__":
    main(project_dir)
