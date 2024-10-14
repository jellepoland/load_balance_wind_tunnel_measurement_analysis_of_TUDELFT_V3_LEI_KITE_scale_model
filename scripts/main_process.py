from load_balance_analysis import process_bundling_beta_0
from load_balance_analysis import process_support_struc_aero_interp_coeffs
from load_balance_analysis import process_raw_csv
from load_balance_analysis import process_raw_lvm_with_labbook_into_df
from load_balance_analysis import plot_and_process_zigzag
from load_balance_analysis import process_vsm


def main():
    process_raw_lvm_with_labbook_into_df.main()
    process_raw_csv.main()
    # process_vsm.main()
    process_bundling_beta_0.main()
    process_support_struc_aero_interp_coeffs.main()
    plot_and_process_zigzag.main_process()


if __name__ == "__main__":
    main()
