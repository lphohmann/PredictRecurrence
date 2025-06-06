# imports
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import make_scorer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#plot_variance_cutoffs(mval_matrix, top_n=200000) 
#plot_variance_cutoffs(mval_matrix, fixed_threshold=20)

def beta2m(beta, beta_threshold=1e-3):
    """
    Convert beta-values to M-values safely.

    Args:
        beta (float, array-like, or pd.DataFrame): Beta-values.
        beta_threshold (float): Lower and upper bound to avoid logit instability. 
                                Clips beta values to [beta_threshold, 1 - beta_threshold].

    Returns:
        Same shape as input: M-values.
    """
    beta = np.clip(beta, beta_threshold, 1 - beta_threshold)
    return np.log2(beta / (1 - beta))


def m2beta(m):
    """
    Convert M-values to beta-values.

    Args:
        m (float, array-like, or pd.DataFrame): M-values.

    Returns:
        Same shape as input: Beta-values.
    """
    return 2**m / (2**m + 1)


# create surv object
def create_surv(df, time="years", event="event"):
    """
    Create a survival object from a pandas DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing survival time and event columns.
        time (str, optional): Name of the time column. Defaults to "RFI_years".
        event (str, optional): Name of the event column. Defaults to "event".
    Returns:
        Surv: scikit-survival Survival object.
    """
    return Surv.from_arrays(df[event].astype(bool), df[time])

# variance filter
def variance_filter(df, min_variance=None, top_n=None):
    """
    Filter features (CpGs) based on variance. Extects CpGs as columns and patients as rows
    Args:
        X (pd.DataFrame): Feature matrix (samples x features).
        top_n (int, optional): Number of top features to select based on variance.
        min_variance (float, optional): Minimum variance threshold to retain features.
    Returns:
        pd.DataFrame: Filtered feature matrix.
    """
    variances = df.var(axis=0)
    
    if min_variance is not None:
        # Filter features with variance above threshold
        selected_features = variances[variances >= min_variance].index
    elif top_n is not None:
        # Select top_n features by variance
        selected_features = variances.sort_values(ascending=False).head(top_n).index
    else:
        raise ValueError("Either min_variance or top_n must be specified")
    
    return df.loc[:, selected_features]

def plot_variance_cutoffs(df, top_n=None, fixed_threshold=None):
    """
    Plot variance distribution of CpGs and show cutoff lines
    based on either selecting top_n CpGs by variance or a fixed variance threshold.
    
    Args:
        df (pd.DataFrame): DataFrame with CpGs as columns.
        top_n (int, optional): Number of top features to select by variance.
        fixed_threshold (float, optional): Minimum variance threshold.
    
    Raises:
        ValueError if neither or both parameters are provided.
    """
    variances = df.var(axis=0).sort_values(ascending=False)

    if (top_n is None and fixed_threshold is None) or (top_n is not None and fixed_threshold is not None):
        raise ValueError("Specify exactly one of top_n or fixed_threshold")

    plt.figure(figsize=(12, 5))

    # Plot variance distribution
    sns.histplot(variances, bins=100, kde=True, color='skyblue')
    plt.xlabel('Variance')
    plt.ylabel('Number of CpGs')
    plt.title('Variance Distribution of CpGs')

    if top_n is not None:
        if top_n > len(variances):
            raise ValueError(f"top_n ({top_n}) cannot be greater than the number of features ({len(variances)})")
        cutoff = variances.iloc[top_n - 1]
        plt.axvline(cutoff, color='red', linestyle='--', label=f'Top {top_n} cutoff: {cutoff:.4f}')
        plt.legend()

    if fixed_threshold is not None:
        plt.axvline(fixed_threshold, color='green', linestyle='--', label=f'Fixed threshold: {fixed_threshold}')
        plt.legend()

    plt.show()

    # Additional plot: Sorted variances with cutoff line
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(variances)), variances.values, color='blue')
    plt.xlabel('CpGs sorted by variance (descending)')
    plt.ylabel('Variance')
    plt.title('Sorted CpG Variances with Cutoff')

    if top_n is not None:
        plt.axhline(cutoff, color='red', linestyle='--', label=f'Top {top_n} cutoff: {cutoff:.4f}')
        plt.legend()

    if fixed_threshold is not None:
        plt.axhline(fixed_threshold, color='green', linestyle='--', label=f'Fixed threshold: {fixed_threshold}')
        plt.legend()

    plt.show()


# This is the official "built-in" scorer from sksurv docs (minimal user code)
def cindex_scorer(estimator, X, y):
    risk = estimator.predict(X)
    # If y is structured array, access like this:
    return concordance_index_censored(y['RFi_event'], y['RFi_years'], -risk)[0]

#custom_scorer = make_scorer(cindex_scorer, greater_is_better=True)


from sksurv.metrics import concordance_index_censored

def cindex_scorer_sksurv(estimator, X, y):
    # Predict risk scores (higher = higher risk)
    risk_scores = -estimator.predict(X)  # negative because predict returns survival function value or coefficient
    
    # Extract event and time from structured array y
    event = y["RFi_event"]
    time = y["RFi_years"]
    
    # Compute concordance index
    cindex = concordance_index_censored(event, time, risk_scores)[0]
    return cindex


from sksurv.metrics import concordance_index_ipcw

def cindex_ipcw_scorer(estimator, X, y):
    risk_scores = -estimator.predict(X)
    
    # Extract event and time fields from structured array
    event = y["RFi_event"]
    time = y["RFi_years"]
    
    # Compute IPCW concordance index — pass event/time twice (train == test)
    cindex = concordance_index_ipcw(
        (event, time),  # training labels
        (event, time),  # test labels (same in CV scorer)
        risk_scores
    )
    
    return cindex
