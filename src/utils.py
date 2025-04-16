# imports
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold, cross_val_score # GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from tqdm import tqdm
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# convert beta-value to m-value
def beta2m(beta):
    """
    Convert beta-value to m-value.
    Args:
        beta (float or array-like): Beta-values.
    Returns:
        float or array-like: M-values.
    """
    m = np.log2(beta / (1 - beta))
    return m


# convert m-value to beta-value
def m2beta(m):
    """
    Convert m-value to beta-value.
    Args:
        m (float or array-like): M-values.
    Returns:
        float or array-like: Beta-values.
    """
    beta = 2**m / (2**m + 1)
    return beta

# create surv object
def create_surv(df, time="RFI_years", event="event"):
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
def variance_filter(X, top_n):
    """
    Filter features (CpGs) based on variance. Extects CpGs as columns and patients as rows
    Args:
        X (pd.DataFrame): Feature matrix (samples x features).
        top_n (int): Number of top features to select based on variance.
    Returns:
        pd.DataFrame: Feature matrix with top_n features.
    """
    variances = X.var(axis=0)  # CpGs as columns
    top_features = variances.sort_values(ascending=False).index[:top_n]
    return X[top_features]

# univariate cox filter
def unicox_filter(X, clinical_df, time_col="RFI_years", event_col="event", top_n=100):
    """
    Select top_n features based on univariate Cox p-values.
    Args:
        X (pd.DataFrame): Feature matrix (samples x features).
        clinical_df (pd.DataFrame): Must contain time_col and event_col.
        time_col (str): Column with time-to-event.
        event_col (str): Column with event status (1 = event, 0 = censored).
        top_n (int): Number of top features to keep.
    Returns:
        pd.DataFrame: Filtered feature matrix with top_n features.
    """
    p_values = []

    for col in tqdm(X.columns, desc="Univariate Cox"):
        df = clinical_df[[time_col, event_col]].copy()
        df["feature"] = X[col].values

        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col=time_col, event_col=event_col)
            p = cph.summary.loc["feature", "p"]
        except Exception:
            p = 1.0

        p_values.append(p)

    pval_series = pd.Series(p_values, index=X.columns)
    top_features = pval_series.sort_values().index[:top_n]
    return X[top_features]

# combined prev functions for preprocessing data
def preprocess(beta_matrix, filter_method="variance", top_n=100, to_mval=False, y=None):
    """
    Preprocess methylation data.

    Takes a matrix of beta-values and performs filtering and optional conversion to m-values.

    Args:
        beta_matrix (pd.DataFrame or np.array): Matrix of beta-values (samples x features).
        filter_method (str, optional): Feature filtering method.
            Options: "variance", "unicox", None. Defaults to "variance".
        top_n (int, optional): Number of top features to keep after filtering. Defaults to 100.
        to_mval (bool, optional): Whether to convert beta-values to m-values. Defaults to False.
        y (Surv, optional): Survival object, required for "unicox" filtering. Defaults to None.

    Returns:
        pd.DataFrame or np.array: Preprocessed methylation data.
    """
    if to_mval:
        beta_matrix = beta2m(beta_matrix)

    if filter_method == "variance":
        beta_matrix = variance_filter(beta_matrix, top_n)

    elif filter_method == "unicox":
        if y is None:
            raise ValueError("Univariate Cox filtering requires survival object y.")
        beta_matrix = unicox_filter(beta_matrix, y, top_n)

    elif filter_method is None: # added None filter method
        pass # no filtering

    else:
        raise ValueError(f"Unknown filter method: {filter_method}")

    return beta_matrix

# train Cox LASSO model with manual alpha tuning and cross-validation
def train_cox_lasso(X, y, alphas=np.logspace(-4, 1, 10), scale=True, cv=5, scoring="neg_log_loss", plot=False):
    """
    Train and optimize a Cox LASSO model using GridSearchCV and optionally plot performance.

    Args:
        X (DataFrame or array): Feature matrix.
        y (structured array): Survival object.
        alphas (array): Alpha grid for LASSO.
        scale (bool): Whether to standardize X.
        cv (int): Number of CV folds.
        scoring (str): Scoring metric for evaluation.
        plot (bool): If True, plot alpha vs CV score.

    Returns:
        best_model: Trained Cox LASSO model.
        best_alpha: Alpha with best CV score.
        cv_results: GridSearchCV results as dict.
        scaler: Scaler object (or None).
    """
    # Scale features yes/no
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X.values if hasattr(X, "values") else X

    # Wrap model into a lambda for compatibility with GridSearchCV
    class CoxWrapper(CoxnetSurvivalAnalysis):
        def set_params(self, **params):
            return super().set_params(**params)
        def get_params(self, deep=True):
            return {"alphas": self.alphas}

    model = CoxWrapper(l1_ratio=1.0, fit_baseline_model=True)

    param_grid = {"alphas": [[a] for a in alphas]}  # wrap each alpha in list as required

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_alpha = grid.best_params_["alphas"][0]
    scores = grid.cv_results_

    print(f"✅ Best alpha: {best_alpha:.5f} (mean CV score = {grid.best_score_:.4f})")

    if plot:
        mean_scores = scores["mean_test_score"]
        plt.figure(figsize=(8, 5))
        plt.plot(alphas, mean_scores, marker="o")
        plt.xscale("log")
        plt.xlabel("Alpha")
        plt.ylabel("CV Score")
        plt.title("Cox LASSO CV Performance")
        plt.axvline(best_alpha, color="red", linestyle="--", label=f"Best alpha = {best_alpha:.5f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return best_model, best_alpha, scores, scaler
