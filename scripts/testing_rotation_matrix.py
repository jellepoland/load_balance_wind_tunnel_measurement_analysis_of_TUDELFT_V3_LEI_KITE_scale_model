import numpy as np
import pandas as pd
from load_balance_analysis.functions_processing import (
    translate_from_origin_to_cg,
    rotate_axis_at_origin,
    correcting_for_sideslip,
    nondimensionalize,
)


# -- Quick test --
# We set up a scenario with pure drag in the +x direction, no side force, no lift.
# Also, the device measured beta = +10° (clockwise), so sideslip is +10 in the DataFrame.

test_data = {
    "sideslip": [-10.0, -10, -10.0, -10],  # -10 deg clockwise, +10 deg counterclockwise
    "aoa": [0, 0, 0, 0],
    "F_X": [1, -1, 0, 0],  # Pure drag along +x
    "F_Y": [0, 0, 0, 0],  # No side force
    "F_Z": [0, 0, 1, 0],  # No lift
    "M_X": [0, 0, 0, 1],
    "M_Y": [0, 0, 0, 2],
    "M_Z": [0, 0, 0, 3],
    "Density": [1.225, 1.225, 1.225, 1.225],
    "vw": [10, 10, 10, 10],
    "F_X_raw": [1, -1, 0, 0],  # Pure drag along +x
    "F_Y_raw": [0, 0, 0, 0],  # No side force
    "F_Z_raw": [0, 0, 1, 0],  # No lift
    "M_X_raw": [0, 0, 0, 1],
    "M_Y_raw": [0, 0, 0, 2],
    "M_Z_raw": [0, 0, 0, 3],
}
df_test = pd.DataFrame(test_data)
df_translated = translate_from_origin_to_cg(
    df_test.copy(),
    x_hinge=434,
    z_hinge=1353.5,
    l_cg=625.4,
    alpha_cg_delta_with_rod=23.82,
)
df_new_reference_frame = rotate_axis_at_origin(df_translated.copy())
df_corrected_sideslip = correcting_for_sideslip(df_new_reference_frame.copy())
df_nondimensional = nondimensionalize(
    df_corrected_sideslip.copy(), S_ref=0.46, c_ref=0.395
)

for i in range(len(df_test["aoa"].values)):
    print(f"\n--- i={i} ---")
    for df, label in zip(
        [
            df_test,
            df_translated,
            df_new_reference_frame,
            df_corrected_sideslip,
            df_nondimensional,
        ],
        ["raw", "translated", "new_reference_frame", "corrected", "nondimensional"],
    ):
        print(f"\n{label}")
        print(f'sideslip: {df["sideslip"].values[i]}, aoa:{df["aoa"].values[i]}')
        if label == "nondimensional":
            print(
                f'C_d  : {df["C_D"].values[i]:.2f}, C_S   : {df["C_S"].values[i]:.2f}, C_L  : {df["C_L"].values[i]:.2f}'
            )
            print(
                f'C_roll  : {df["C_roll"].values[i]:.2f}, C_pitch   : {df["C_pitch"].values[i]:.2f}, C_yaw  : {df["C_yaw"].values[i]:.2f}'
            )
        else:
            print(
                f'F_x  : {df["F_X"].values[i]:.2f}, F_y   : {df["F_Y"].values[i]:.2f}, F_z  : {df["F_Z"].values[i]:.2f}'
            )
            print(
                f'M_x  : {df["M_X"].values[i]:.2f}, M_y   : {df["M_Y"].values[i]:.2f}, M_z  : {df["M_Z"].values[i]:.2f}'
            )


# test_data = {
#     "sideslip": [-10.0],  # -10 deg clockwise, +10 deg counterclockwise
#     "C_D": [1.0],  # Pure drag along +x
#     "C_S": [0.0],  # No side force
#     "C_L": [0.0],  # No lift
#     "C_roll": [0.0],
#     "C_pitch": [0.0],
#     "C_yaw": [0.0],
# }
# df_test = pd.DataFrame(test_data)

# print("BEFORE CORRECTION:")
# print(df_test)

# # Apply the rotation fix
# df_corrected = correcting_for_sideslip(df_test.copy())

# print("\nAFTER CORRECTION:")
# print(df_corrected)
