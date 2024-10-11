import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from utils import project_dir


def print_sensor_drift_values(project_dir: str) -> None:

    folder_dir = Path(project_dir) / "data" / "sensor_drift"
    # Load the data
    bod_april6 = np.mean(np.genfromtxt(Path(folder_dir) / "bod_april6.txt"), axis=0)
    bod_april8 = np.genfromtxt(Path(folder_dir) / "bod_april8.txt")
    bod_april9 = np.genfromtxt(Path(folder_dir) / "bod_april9.txt")

    eod_april5 = np.genfromtxt(Path(folder_dir) / "eod_april5.txt")
    eod_april6 = np.genfromtxt(Path(folder_dir) / "eod_april6.txt")
    eod_april8 = np.genfromtxt(Path(folder_dir) / "eod_april8.txt")

    tot = np.vstack(
        [eod_april5, bod_april6, eod_april6, bod_april8, eod_april8, bod_april9]
    )

    CL_list = []
    CD_list = []
    CS_list = []
    CMx_list = []
    CMy_list = []
    CMz_list = []
    for day in tot:
        CL_list.append(day[1])
        CD_list.append(day[2])
        CS_list.append(day[3])
        CMx_list.append(day[4])
        CMy_list.append(day[5])
        CMz_list.append(day[6])

    ##
    Fx_min = 80
    Fx_max = 120
    Fy_min = 1
    Fy_max = 70
    Fz_min = 600
    Fz_max = 800
    ##
    Mx_min = -100
    Mx_max = 100
    My_min = -70
    My_max = 70
    Mz_min = -60
    Mz_max = 60

    print("\nSensor drift values:")
    print(
        f"Fx: mean: {np.mean(CL_list):.2f}, std: {np.std(CL_list):.2f}, min: {Fx_min:.2f}, max: {Fx_max:.2f}"
    )
    print(
        f"Fy: mean: {np.mean(CD_list):.2f}, std: {np.std(CD_list):.2f}, min: {Fy_min:.2f}, max: {Fy_max:.2f}"
    )
    print(
        f"Fz: mean: {np.mean(CS_list):.2f}, std: {np.std(CS_list):.2f}, min: {Fz_min:.2f}, max: {Fz_max:.2f}"
    )
    print(
        f"Mx: mean: {np.mean(CMx_list):.2f}, std: {np.std(CMx_list):.2f}, min: {Mx_min:.2f}, max: {Mx_max:.2f}"
    )
    print(
        f"My: mean: {np.mean(CMy_list):.2f}, std: {np.std(CMy_list):.2f}, min: {My_min:.2f}, max: {My_max:.2f}"
    )
    print(
        f"Mz: mean: {np.mean(CMz_list):.2f}, std: {np.std(CMz_list):.2f}, min: {Mz_min:.2f}, max: {Mz_max:.2f}"
    )


def main(project_dir: str) -> None:
    print_sensor_drift_values(project_dir)


if __name__ == "__main__":
    main(project_dir)
