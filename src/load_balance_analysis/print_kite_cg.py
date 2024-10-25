import numpy as np


def main(project_dir):
    print(f"\n Kite CG calculation")
    # Given parameters
    l_cg = 625.4  # distance to CG in mm
    # angle_rod_to_alpha = 7.25  # degrees between rod and alpha angle
    angle_rod_to_cg = 23.82  # degrees from rod to CG
    h_rod_to_te = 23.85  # distance from rod to trailing edge (TE) in mm
    len_rod = 400  # length of rod in m

    # first calc. position of TE wrt to R, rotation point
    delta_phi = np.arctan((h_rod_to_te) / len_rod)
    diagonal_len = np.sqrt(h_rod_to_te**2 + len_rod**2)
    x_te = diagonal_len * np.cos(delta_phi)
    y_te = diagonal_len * np.sin(delta_phi)

    print(f"x_te: {x_te:.2f} mm")
    print(f"y_te: {y_te:.2f} mm")

    # calc. position of CG wrt to R, rotation point
    phi = np.radians(angle_rod_to_cg)
    x_cg = l_cg * np.cos(phi)
    y_cg = l_cg * np.sin(phi)

    print(f"x_cg: {x_cg:.2f} mm")
    print(f"y_cg: {y_cg:.2f} mm")

    # calculation te position wrt to CG
    x_te_cg = x_te - x_cg
    y_te_cg = y_te - y_cg

    print(f"x_te_cg: {x_te_cg:.2f} mm")
    print(f"y_te_cg: {y_te_cg:.2f} mm")


if __name__ == "__main__":
    main()
