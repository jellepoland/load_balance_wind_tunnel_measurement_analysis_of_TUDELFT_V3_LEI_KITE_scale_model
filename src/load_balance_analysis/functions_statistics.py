import numpy as np
from scipy import stats
import statsmodels.api as sm


## Confidence Interval
def calculate_confidence_interval(data, alpha=0.01):

    # print(f"data: {data}")
    mean = data.mean()
    sem = stats.sem(data)
    critical_value = stats.t.ppf(1 - alpha / 2, df=len(data) - 1)
    conf_interval = critical_value * sem
    # print(f"mean: {mean}, conf_interval: {conf_interval}")
    return conf_interval


## Confidence Interval using BLOCK BOOTSTRAP
def block_bootstrap_confidence_interval(
    data, block_size=500, n_bootstrap=1000, alpha=0.01
):
    """
    Calculate the confidence interval using block bootstrapping to handle autocorrelation.

    Args:
        data (np.ndarray): Time series data.
        block_size (int): The size of the blocks to preserve autocorrelation.
        n_bootstrap (int): Number of bootstrap resamples to generate.
        alpha (float): Significance level (default: 0.05 for a 95% confidence interval).

    Returns:
        mean (float): The mean of the original data.
        conf_interval (tuple): The confidence interval (lower bound, upper bound).
    """
    data = np.array(data)
    n_data = len(data)

    # Step 1: Split data into blocks
    num_blocks = n_data // block_size
    blocks = [
        data[i : i + block_size] for i in range(0, num_blocks * block_size, block_size)
    ]

    # Step 2: Perform bootstrap resampling on blocks
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample blocks with replacement
        bootstrapped_data = np.concatenate(
            [blocks[i] for i in np.random.randint(0, len(blocks), num_blocks)]
        )
        bootstrap_means.append(np.mean(bootstrapped_data))

    # Step 3: Compute the mean of the original data
    mean = np.mean(data)

    # Step 4: Calculate the confidence interval from bootstrapped means
    lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    confidence_interval = np.abs(np.abs(lower_bound) - np.abs(upper_bound))
    return confidence_interval


# ## Confidence Interval using HAC/Newey-West
# def hac_newey_west_confidence_interval(data, alpha=0.01, max_lag=None):
#     """
#     Calculate the confidence interval using the HAC/Newey-West method to handle autocorrelation and heteroskedasticity.

#     Args:
#         data (np.ndarray): Time series data from wind tunnel aerodynamic experiment.
#         alpha (float): Significance level (default: 0.01 for a 99% confidence interval).
#         max_lag (int): Maximum lag to consider for autocorrelation. If None, it's set to n^(1/4) where n is the sample size.

#     Returns:
#         mean (float): The mean of the original data.
#         conf_interval (float): The confidence interval (half-width).
#     """
#     data = np.array(data)
#     n = len(data)

#     # Calculate the mean
#     mean = np.mean(data)

#     # Calculate the variance
#     centered_data = data - mean
#     variance = np.mean(centered_data**2)

#     # Set max_lag if not provided
#     if max_lag is None:
#         max_lag = int(n ** (1 / 4))
#         print(f"Max lag set to {max_lag}")
#     # Calculate autocovariances up to max_lag
#     auto_cov = np.array(
#         [np.mean(centered_data[i:] * centered_data[:-i]) for i in range(1, max_lag + 1)]
#     )

#     # Calculate Newey-West weights
#     weights = 1 - np.arange(1, max_lag + 1) / (max_lag + 1)

#     # Calculate the HAC standard error
#     hac_variance = variance + 2 * np.sum(weights * auto_cov)
#     hac_se = np.sqrt(hac_variance / n)

#     # Calculate the confidence interval
#     t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
#     conf_interval = t_value * hac_se

#     return conf_interval


def hac_newey_west_confidence_interval(data, confidence_interval=99.99, max_lag=None):
    """
    Calculate the confidence interval using the HAC/Newey-West method to handle autocorrelation and heteroskedasticity.

    Args:
        data (np.ndarray): Time series data from wind tunnel aerodynamic experiment.
        alpha (float): Significance level (default: 0.01 for a 99% confidence interval).
        max_lag (int): Maximum lag to consider for autocorrelation. If None, it's set to n^(1/4) where n is the sample size.

    Returns:
        mean (float): The mean of the original data.
        conf_interval (float): The confidence interval (half-width).
    """

    alpha_ci = 1 - (confidence_interval / 100)

    data = np.array(data)
    n = len(data)

    # Set max_lag if not provided
    if max_lag is None:
        max_lag = int(n ** (1 / 4))
        # print(f"Max lag set to {max_lag}")

    # Add a constant to the data (intercept)
    X = sm.add_constant(np.arange(n))

    # Fit OLS model
    model = sm.OLS(data, X)
    results = model.fit()

    # Get Newey-West standard errors with specified max_lag
    nw_cov = results.get_robustcov_results(cov_type="HAC", maxlags=max_lag)
    nw_se = nw_cov.bse[0]  # Standard error for the intercept (constant term)

    # Calculate the t-value for the given alpha and degrees of freedom (n-1)
    t_value = stats.t.ppf(1 - alpha_ci / 2, df=n - 1)

    # Confidence interval (half-width)
    conf_interval = t_value * nw_se

    return conf_interval
