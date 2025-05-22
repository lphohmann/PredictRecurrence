# imports
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import make_scorer

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

# This is the official "built-in" scorer from sksurv docs (minimal user code)
def cindex_scorer(estimator, X, y):
    risk = estimator.predict(X)
    # If y is structured array, access like this:
    return concordance_index_censored(y['RFi_event'], y['RFi_years'], -risk)[0]



custom_scorer = make_scorer(cindex_scorer, greater_is_better=True)
