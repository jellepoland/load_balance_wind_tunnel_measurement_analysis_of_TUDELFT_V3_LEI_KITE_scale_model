import pandas as pd
import numpy as np
from pathlib import Path
import os
from load_balance_analysis.functions_utils import project_dir
from load_balance_analysis.functions_processing import processing_raw_lvm_data_into_csv


def main():
    # processing all the folders for the normal
    support_struc_aero_interp_coeffs_path = (
        Path(project_dir) / "processed_data" / "interpolation_coefficients.csv"
    )
    is_kite = True
    is_zigzag = False
    print(f"\n Processing all the folders")
    for folder in os.listdir(Path(project_dir) / "processed_data" / "normal_csv"):
        if "alpha" not in folder:
            continue
        folder_dir = Path(project_dir) / "processed_data" / "normal_csv" / folder
        processing_raw_lvm_data_into_csv(
            folder_dir,
            is_kite,
            is_zigzag,
            support_struc_aero_interp_coeffs_path,
        )


if __name__ == "__main__":
    main()
