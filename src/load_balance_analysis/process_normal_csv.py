import pandas as pd
import numpy as np
from pathlib import Path
import os
from load_balance_analysis.functions_utils import project_dir
from load_balance_analysis.functions_processing import processing_raw_lvm_data_into_csv


def main():
    S_ref = 0.46
    c_ref = 0.4

    # parameters necessary to translate moments (aka determine position of cg)
    x_hinge = (
        441.5  # x distance between force balance coord. sys. and hinge point in mm
    )
    z_hinge = 1359  # z distance between force balance coord. sys. and hinge point in mm
    l_cg = 625.4  # distance between hinge point and kite CG
    alpha_cg_delta_with_rod = 23.82
    delta_celsius_kelvin = 273.15
    mu_0 = 1.716e-5
    T_0 = 273.15
    C_suth = 110.4
    delta_aoa_rod_to_alpha = 7.25
    l_rod = 400  # length of the rod from hinge to TE

    print(f"\nThe calculation of the center of gravity")
    print(
        f" hinge point is located at: \nx_hinge = {x_hinge} mm\nz_hinge = {z_hinge} mm"
    )
    print(f" distance between hinge point and kite CG: l_cg = {l_cg} mm")
    print(
        f" angle between the rod and the kite: alpha_cg_delta_with_rod = {alpha_cg_delta_with_rod} deg"
    )
    angle_cg = 10 - alpha_cg_delta_with_rod
    print(
        f"\n For an measured aoa of 10deg, this would make angle_cg = 10 - alpha_cg_delta_with_rod = {angle_cg} deg"
    )
    x_hinge_cg = l_cg * np.cos(np.deg2rad(angle_cg))
    print(f"x_hinge,cg = l_cg * np.cos(angle_cg) = {x_hinge_cg:.2f} mm")
    z_hinge_cg = l_cg * np.sin(np.deg2rad(angle_cg))
    print(f"z_hinge,cg = l_cg * np.sin(angle_cg) = {z_hinge_cg:.2f} mm")
    x_cg_origin = x_hinge + x_hinge_cg
    print(f"\n x_cg,origin = x_hinge + x_hinge,cg = {x_cg_origin:.2f} mm")
    z_cg_origin = z_hinge + z_hinge_cg
    print(f" z_cg,origin = z_hinge + z_hinge,cg = {z_cg_origin:.2f} mm")
    print(f"---------------------------------")
    print(f"\n Location of TE, with a 0 measured aoa")
    x_te = x_hinge + l_rod
    print(
        f" x_te = x_hinge + l_rod = {x_te:.2f} mm, with l_rod (distance from hinge-to-TE) = {l_rod:.2f} mm"
    )
    z_te = z_hinge
    print(f" z_te = z_hinge = {z_te:.2f} mm")
    angle_cg = 0 - alpha_cg_delta_with_rod
    print(
        f" angle cg with 0 deg aoa: angle_cg = 0 - alpha_cg_delta_with_rod = {angle_cg:.2f} deg"
    )
    x_cg_origin = x_hinge + l_cg * np.cos(np.deg2rad(angle_cg))
    print(f"\nx_cg,origin = {x_cg_origin:.2f} mm")
    print(f"z_cg,origin = {z_cg_origin:.2f} mm")
    z_cg_origin = z_hinge + l_cg * np.sin(np.deg2rad(angle_cg))
    print(f"\n Distance between cg and TE at 0 deg aoa")
    x_te_cg = x_cg_origin - x_te
    print(f" x_te_cg = x_cg,origin - x_te = {x_te_cg:.2f} mm")
    z_te_cg = z_cg_origin - z_te
    print(f" z_te_cg = z_cg,origin - z_te = {z_te_cg:.2f} mm")
    print(f"alternative x_te_cg: {l_cg * np.cos(np.deg2rad(angle_cg)) - l_rod:.2f}")
    print(f"alternative z_te_cg: {l_cg * np.sin(np.deg2rad(angle_cg)):.2f}")

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
            S_ref,
            c_ref,
            is_kite,
            is_zigzag,
            support_struc_aero_interp_coeffs_path,
            x_hinge,
            z_hinge,
            l_cg,
            alpha_cg_delta_with_rod,
            delta_celsius_kelvin,
            mu_0,
            T_0,
            C_suth,
            delta_aoa_rod_to_alpha,
        )


if __name__ == "__main__":
    main()
