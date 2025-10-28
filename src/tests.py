import numpy as np
import pandas as pd
import pytest
from sksurv.util import Surv

# run with: pytest -q ./src/tests.py

# import 
import sys
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")

# This decorator tells pytest that `fake_data` is a fixture:
# a reusable setup function that generates synthetic data
# and can be injected into multiple tests automatically.
@pytest.fixture
def fake_data():
    """Generate fake methylation + survival data for testing."""
    np.random.seed(42)
    n_samples, n_cpgs = 100, 500

    X = pd.DataFrame(
        np.random.randn(n_samples, n_cpgs),
        columns=[f"CpG_{i}" for i in range(n_cpgs)]
    )

    event_times = np.random.exponential(scale=5, size=n_samples)
    events = np.random.binomial(1, 0.6, size=n_samples)

    y = np.array(list(zip(events, event_times)),
                 dtype=[("RFi_event", "bool"), ("RFi_years", "f8")])

    return X, y

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
# Add project src directory to path for imports (adjust as needed)
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import _fit_univar_cox, univariate_cox_filter 
# src/tests.py


# -------------------------
# Helper to build structured y with RFi_years and RFi_event
# -------------------------
def make_y(n, seed=None):
    rng = np.random.RandomState(seed)
    y = np.zeros(n, dtype=[('RFi_years', float), ('RFi_event', int)])
    y["RFi_years"] = rng.exponential(scale=5, size=n)
    y["RFi_event"] = rng.binomial(1, 0.7, size=n)
    return y


def test_fit_univar_cox_basic():
    """Check _fit_univar_cox returns a valid p-value and is smaller than shuffled predictor's p."""
    rng = np.random.RandomState(42)
    n = 200  # increase n a bit for more stable signal
    time = rng.exponential(scale=5, size=n)
    event = rng.binomial(1, 0.7, size=n)

    # Create a feature with modest association to survival
    x = rng.randn(n)
    time_signal = time - 0.5 * x  # introduce association (shorter time when x is larger)

    pval_signal = _fit_univar_cox(x, time_signal, event)
    assert np.isfinite(pval_signal) and 0.0 <= pval_signal <= 1.0

    # Shuffle x to produce a null predictor and compute its p-value
    x_shuf = rng.permutation(x)
    pval_shuf = _fit_univar_cox(x_shuf, time_signal, event)
    assert np.isfinite(pval_shuf) and 0.0 <= pval_shuf <= 1.0

    # Signal should not be worse than shuffled (robust non-flaky check)
    assert pval_signal <= pval_shuf + 1e-12, f"signal p={pval_signal} not <= shuffled p={pval_shuf}"


def test_fit_univar_cox_constant():
    """Should return NaN when feature has no variance."""
    rng = np.random.RandomState(1)
    n = 50
    y = make_y(n, seed=1)
    x = np.ones(n)  # constant feature
    pval = _fit_univar_cox(x, y["RFi_years"], y["RFi_event"])
    assert np.isnan(pval)


def test_univariate_cox_filter_basic():
    """End-to-end: check univariate_cox_filter runs and returns valid feature list."""
    rng = np.random.RandomState(0)
    n = 120
    X = pd.DataFrame({
        "cpg_good": rng.randn(n),
        "cpg_bad": rng.randn(n),
        "cpg_noise": rng.randn(n)
    })
    y = np.zeros(n, dtype=[('RFi_years', float), ('RFi_event', int)])
    y["RFi_years"] = rng.exponential(scale=5, size=n)
    y["RFi_event"] = rng.binomial(1, 0.8, size=n)

    # introduce weak signal
    y["RFi_years"] -= 0.1 * X["cpg_good"].values

    # call the filter
    selected = univariate_cox_filter(X, y, top_n=2)

    # ---- checks ----
    assert isinstance(selected, list), "Filter should return a list"
    assert 1 <= len(selected) <= 2, "Filter should return at most top_n features, at least 1"
    for feat in selected:
        assert feat in X.columns, f"Selected feature {feat} not in X columns"



def test_univariate_cox_filter_keep_vars():
    """Keep_vars must always be present in the output and no duplicates."""
    rng = np.random.RandomState(1)
    n = 50
    X = pd.DataFrame({
        "a": rng.randn(n),
        "b": rng.randn(n)
    })
    y = np.zeros(n, dtype=[('RFi_years', float), ('RFi_event', int)])
    y["RFi_years"] = rng.exponential(scale=4, size=n)
    y["RFi_event"] = rng.binomial(1, 0.9, size=n)

    keep = ["a"]
    selected = univariate_cox_filter(X, y, top_n=1, keep_vars=keep)

    # Always includes keep_vars
    assert "a" in selected
    # Shouldn't contain duplicates
    assert len(selected) == len(set(selected))


def test_univariate_cox_filter_bad_y():
    """Raises ValueError if y missing required fields (now expects RFi_years)."""
    n = 30
    X = pd.DataFrame({"x1": np.random.randn(n)})
    # y missing RFi_years field
    y = np.zeros(n, dtype=[('wrong_field', float), ('event', int)])

    with pytest.raises(ValueError):
        univariate_cox_filter(X, y, top_n=1)

def test_univariate_cox_filter_strong_signal():
    """
    Test that univariate_cox_filter selects strong signal CpGs among noise.
    """
    rng = np.random.RandomState(42)
    n_samples = 200
    n_noise = 50
    n_signal = 3

    # Create noise features
    X_noise = rng.randn(n_samples, n_noise)
    noise_cols = [f"cpg_noise_{i}" for i in range(n_noise)]
    X = pd.DataFrame(X_noise, columns=noise_cols)

    # Create strong signal features
    signal_cols = [f"cpg_signal_{i}" for i in range(n_signal)]
    X_signal = rng.randn(n_samples, n_signal)
    X[signal_cols] = X_signal

    # Construct survival outcome
    base_time = rng.exponential(scale=10, size=n_samples)
    # strong association with signal features
    time = base_time - 2.0 * X_signal[:, 0] - 1.5 * X_signal[:, 1] + 0.5 * X_signal[:, 2]
    event = rng.binomial(1, 0.7, size=n_samples)

    y = np.zeros(n_samples, dtype=[("RFi_years", float), ("RFi_event", int)])
    y["RFi_years"] = time
    y["RFi_event"] = event

    # Run filter with top_n large enough to capture all signals
    selected = univariate_cox_filter(X, y, top_n=10)

    # ---- checks ----
    # All signal features should be selected
    for col in signal_cols:
        assert col in selected, f"Signal feature {col} not selected!"

    # Selected features should exist in X
    for col in selected:
        assert col in X.columns
