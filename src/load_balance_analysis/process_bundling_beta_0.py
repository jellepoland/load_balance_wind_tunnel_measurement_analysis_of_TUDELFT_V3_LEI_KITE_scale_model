import os
from pathlib import Path
import pandas as pd
from load_balance_analysis.functions_utils import project_dir


def main():
    # Looping through all the folders again, to bundle the sideslip data
    print(f"\n Bundling the sideslip data, for sideslip == 0")
    vw_5, vw_10, vw_15, vw_20, vw_25 = [], [], [], [], []
    for folder in os.listdir(Path(project_dir) / "processed_data" / "normal_csv"):
        folder_dir = Path(project_dir) / "processed_data" / "normal_csv" / folder
        # skipping the folders that don't have "alpha" in their name
        if "alpha" not in folder:
            continue
        # looping through each file in the folder
        for file in os.listdir(folder_dir):
            if "raw" not in file:
                print(f"\nfile:{file}")
                file_path = Path(folder_dir) / file
                df = pd.read_csv(file_path)
                # print(f"df columns: {df.columns}")

                if file == "vw_5.csv":
                    vw_5.append(df)
                elif file == "vw_10.csv":
                    vw_10.append(df)
                elif file == "vw_15.csv":
                    vw_15.append(df)
                elif file == "vw_20.csv":
                    vw_20.append(df)
                elif file == "vw_25.csv":
                    vw_25.append(df)

    # Concatenating the dataframes
    vw_5 = pd.concat(vw_5)
    vw_10 = pd.concat(vw_10)
    vw_15 = pd.concat(vw_15)
    vw_20 = pd.concat(vw_20)
    vw_25 = pd.concat(vw_25)

    # Filtering the dataframes on sideslip == 0
    vw_5 = vw_5[vw_5["sideslip"] == 0]
    vw_10 = vw_10[vw_10["sideslip"] == 0]
    vw_15 = vw_15[vw_15["sideslip"] == 0]
    vw_20 = vw_20[vw_20["sideslip"] == 0]
    vw_25 = vw_25[vw_25["sideslip"] == 0]

    # Saving the concatenated dataframes
    folder_name = "beta_0"
    path_folder = Path(project_dir) / "processed_data" / "normal_csv" / folder_name
    if "beta_0" not in os.listdir(Path(project_dir) / "processed_data" / "normal_csv"):
        os.mkdir(path_folder)
    vw_5.to_csv(Path(path_folder) / "vw_5.csv", index=False)
    vw_10.to_csv(Path(path_folder) / "vw_10.csv", index=False)
    vw_15.to_csv(Path(path_folder) / "vw_15.csv", index=False)
    vw_20.to_csv(Path(path_folder) / "vw_20.csv", index=False)
    vw_25.to_csv(Path(path_folder) / "vw_25.csv", index=False)


if __name__ == "__main__":
    main()
