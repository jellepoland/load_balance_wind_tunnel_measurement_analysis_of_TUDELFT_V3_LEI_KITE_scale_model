from pathlib import Path


def reduce_df_by_parameter(df, parameter):
    """
    Reduces a dataframe to unique values of a specified parameter (aoa_kite or sideslip),
    averaging the force and moment columns for each unique value.

    Parameters:
    df (pandas.DataFrame): The input dataframe
    parameter (str): Either 'aoa_kite' or 'sideslip'

    Returns:
    pandas.DataFrame: A reduced dataframe with unique parameter values and averaged forces/moments
    """
    # Verify parameter is valid
    if parameter not in ["aoa_kite", "sideslip"]:
        raise ValueError("Parameter must be either 'aoa_kite' or 'sideslip'")

    # Columns to average
    force_moment_cols = ["F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]

    # Verify all required columns exist in dataframe
    required_cols = [parameter] + force_moment_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Group by the parameter and calculate means
    reduced_df = df.groupby(parameter)[force_moment_cols].mean().reset_index()

    return reduced_df


project_dir = Path(__file__).resolve().parent.parent.parent

if __name__ == "__main__":
    print(project_dir)
