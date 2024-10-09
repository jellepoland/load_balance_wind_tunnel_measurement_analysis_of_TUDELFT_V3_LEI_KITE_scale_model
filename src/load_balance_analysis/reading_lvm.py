import os
from pathlib import Path
import pandas as pd
from utils import project_dir


def read_lvm(filename: str) -> pd.DataFrame:
    # Read the entire data file
    df = pd.read_csv(filename, skiprows=21, delimiter="\t", engine="python")
    df.drop(
        columns=df.columns[-1], inplace=True
    )  # Assuming the last column is to be dropped
    df.columns = ["time", "F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]

    # Extract the filename and remove "_unsteady.lvm" from the end
    df["Filename"] = os.path.basename(filename).replace("_unsteady.lvm", "")

    # Extract angle of attack information if available
    if "aoa" in df.columns:
        df["aoa"] = df["aoa"].fillna(
            method="ffill"
        )  # Fill NaN values with the previous non-NaN value

    # Calculate sample index based on the number of rows and the assumption that each sample has 19800 entries
    num_samples = len(df) // 19800
    df["sample index"] = sum([[i] * 19800 for i in range(num_samples)], [])

    # Select only the first 17 samples
    selected_rows = 19800 * 17  # Select the first 17 samples
    df = df.iloc[:selected_rows]

    return df


def read_all_lvm_into_df(labbook_df: pd.DataFrame, data_dir: Path) -> list:

    # Remove rows where wind speed is zero from labbook
    # labbook_df = labbook_df[labbook_df["vw"] > 1.1]

    # print(f"Labbook: {labbook_df}")

    # zigzag_data_dir = Path(project_dir) / "data" / "zigzag"
    all_data = []

    # Extract only the lvm files
    lvm_files = [f for f in os.listdir(data_dir) if f.endswith(".lvm")]

    # Loop through each file in the zigzag directory
    for file in lvm_files:
        print(f"\nProcessing file: {file}")

        # Read the lvm file
        lvm_file_path = Path(data_dir) / file
        df = read_lvm(lvm_file_path)

        # Strip lvm from file
        filename = lvm_file_path.stem
        print(f"Filename: {filename}")

        # Filter labbook for where folder matches the filename
        matching_rows = labbook_df[labbook_df["Filename"] == filename]

        # print(f"Matching rows: {matching_rows}")
        n_rows = len(matching_rows)
        print(f"Number of matching rows: {n_rows}")

        # Calculate sample index based on the number of rows and the assumption that each sample has 19800 entries
        # print(f"len(df): {len(df)}")
        num_samples = len(df) // 19800
        print(f"Number of samples: {num_samples}")

        if n_rows == 1:
            if num_samples == 1:
                df_sample = df.copy()
            elif num_samples == 2:
                df_sample = df.iloc[19800:].copy()

            last_matching_row = matching_rows.iloc[-1]  # Get the last occurrence
            properties_to_add = last_matching_row.to_dict()
            # print(f"properties_to_add: {properties_to_add}")

            # Add properties from labbook to each row in df
            for col, value in properties_to_add.items():
                df_sample[col] = (
                    value  # This will create new columns in df with properties from labbook_df
                )

            all_data.append(df_sample)
        else:
            print(f"\nMore than 1 matching row")
            # Create a new DataFrame for each sample
            for i in range(num_samples):
                # Selecting the sample from the DataFrame
                df_sample = df.iloc[i * 19800 : (i + 1) * 19800].copy()

                # Finding the matching row
                row = matching_rows.iloc[i]
                print(f"Row: {row}")
                properties_to_add = row.to_dict()
                print(f"properties_to_add: {properties_to_add}")

                # Add properties from labbook to each row in df
                for col, value in properties_to_add.items():
                    df_sample[col] = (
                        value  # This will create new columns in df with properties from labbook_df
                    )

                all_data.append(df_sample)

    return all_data


def read_labbook_into_df(labbook_path: str) -> pd.DataFrame:
    return pd.read_csv(labbook_path, delimiter=";")


if __name__ == "__main__":
    # labbook_path = Path(project_dir) / "processed_data" / "zigzag" / "labbook_zz.csv"
    labbook_path = Path(project_dir) / "data" / "labbook.csv"
    labbook_df = read_labbook_into_df(labbook_path)
    print(labbook_df.columns)
    data_dir = Path(project_dir) / "data" / "zigzag"
    data_dir = Path(project_dir) / "data" / "zigzag"
    all_data = read_all_lvm_into_df(labbook_df, data_dir)

    # print(all_data)
