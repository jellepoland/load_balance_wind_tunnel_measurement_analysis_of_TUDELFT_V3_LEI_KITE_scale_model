import numpy as np
import pandas as pd


# -- Your original correcting_for_sideslip function --
def correcting_for_sideslip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Suppose the turntable measured beta positive clockwise,
    but we want it positive counterclockwise in our final x,y,z frame.
    """
    # Flip the sign so beta is now +10° => -10° for the code
    df["sideslip"] = -df["sideslip"]

    # Convert to radians
    beta = np.deg2rad(df["sideslip"])

    # Create the rotation matrix
    def create_rotation_matrix(beta_angle):
        return np.array(
            [
                [np.cos(beta_angle), -np.sin(beta_angle), 0],
                [np.sin(beta_angle), np.cos(beta_angle), 0],
                [0, 0, 1],
            ]
        )

    # Forces and moments as arrays
    forces = np.array([df["C_D"], df["C_S"], df["C_L"]]).T
    moments = np.array([df["C_roll"], df["C_pitch"], df["C_yaw"]]).T

    # Apply rotation row by row
    corrected_forces = np.zeros_like(forces)
    corrected_moments = np.zeros_like(moments)

    for i in range(len(df)):
        R = create_rotation_matrix(beta[i])
        corrected_forces[i] = R @ forces[i]
        corrected_moments[i] = R @ moments[i]

    # Put corrected values back
    df["C_D"], df["C_S"], df["C_L"] = corrected_forces.T
    df["C_roll"], df["C_pitch"], df["C_yaw"] = corrected_moments.T

    return df


# -- Quick test --
# We set up a scenario with pure drag in the +x direction, no side force, no lift.
# Also, the device measured beta = +10° (clockwise), so sideslip is +10 in the DataFrame.

test_data = {
    "sideslip": [-10.0],  # -10 deg clockwise, +10 deg counterclockwise
    "C_D": [1.0],  # Pure drag along +x
    "C_S": [0.0],  # No side force
    "C_L": [0.0],  # No lift
    "C_roll": [0.0],
    "C_pitch": [0.0],
    "C_yaw": [0.0],
}
df_test = pd.DataFrame(test_data)

print("BEFORE CORRECTION:")
print(df_test)

# Apply the rotation fix
df_corrected = correcting_for_sideslip(df_test.copy())

print("\nAFTER CORRECTION:")
print(df_corrected)

# Interpretation:
#   -- sideslip should flip sign internally so that we effectively rotate by -10° (counterclockwise)
#      to "undo" the clockwise rotation.
#   -- This means the +x drag vector will now appear partially in +y after the correction,
#      if you interpret the corrected values as "back in the wind tunnel frame."
