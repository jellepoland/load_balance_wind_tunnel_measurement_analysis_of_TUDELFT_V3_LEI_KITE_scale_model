import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import (
    project_dir,
    saving_pdf_and_pdf_tex,
    x_axis_labels,
    y_axis_labels,
    reduce_df_by_parameter_mean_and_std,
    apply_angle_wind_tunnel_corrections_to_df,
)
from load_balance_analysis.functions_statistics import (
    hac_newey_west_confidence_interval,
)
from matplotlib.patches import Patch
from plot_styling import plot_on_ax


from load_balance_analysis.plot_literature_comparison_saving import (
    saving_alpha_and_beta_sweeps,
)
from load_balance_analysis.plot_literature_comparison_alpha import (
    plotting_polars_alpha,
    plotting_polars_alpha_moments,
)
from load_balance_analysis.plot_literature_comparison_beta import (
    plotting_polars_beta,
    plotting_polars_beta_moments,
)


def main(results_dir, project_dir):

    confidence_interval = 99
    # saving_alpha_and_beta_sweeps(
    #     project_dir,
    #     confidence_interval=confidence_interval,
    #     max_lag=11,
    # )
    # plotting_polars_alpha(
    #     project_dir,
    #     results_dir,
    #     confidence_interval=confidence_interval,
    # )
    # plotting_polars_alpha_moments(
    #     project_dir,
    #     results_dir,
    #     confidence_interval=confidence_interval,
    # )
    plotting_polars_beta(
        project_dir,
        results_dir,
        confidence_interval=confidence_interval,
    )
    plotting_polars_beta_moments(
        project_dir,
        results_dir,
        confidence_interval=confidence_interval,
    )
