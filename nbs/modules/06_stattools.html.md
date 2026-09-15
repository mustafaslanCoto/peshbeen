---
title: Statistical Tools (`statstools`)
---




> Essential statistical diagnostics, unit root tests, correlation analysis, and trend/seasonality decomposition tools for time series forecasting.


::: {#6a4dc408 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
from __future__ import annotations
from typing import Union, Literal
import warnings
from statistics import NormalDist
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, kpss, pacf, ccf
from statsmodels.tsa.seasonal import STL

ArrayLike = Union[np.ndarray, list, pd.Series]
```
:::


## Helper Functions

Internal utilities for input sanitization and array conversion.

::: {#13cfbb77 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def _to_clean_1d_array(series: ArrayLike, name: str = "series") -> np.ndarray:
    """Convert input to a clean 1D float numpy array, handling NaN/Inf values."""
    if isinstance(series, pd.Series):
        arr = series.to_numpy(dtype=float)
    elif isinstance(series, (list, tuple)):
        arr = np.asarray(series, dtype=float)
    elif isinstance(series, np.ndarray):
        arr = series.astype(float).ravel()
    else:
        arr = np.asarray(series, dtype=float).ravel()

    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name} must be a non-empty 1-dimensional array-like sequence.")

    nan_mask = np.isnan(arr) | np.isinf(arr)
    if np.any(nan_mask):
        n_dropped = int(np.sum(nan_mask))
        warnings.warn(
            f"{name} contains {n_dropped} NaN or Inf values which were dropped for statistical testing.",
            UserWarning,
            stacklevel=2,
        )
        arr = arr[~nan_mask]
        if len(arr) == 0:
            raise ValueError(f"{name} has no valid non-NaN finite observations.")

    return arr
```
:::


## Unit Root and Stationarity Tests

Stationarity is a fundamental prerequisite for many time series forecasting models. `unit_root_test` provides a unified, parametric interface for:
- **Augmented Dickey-Fuller (ADF) Test**:
  - $H_0$: Unit root is present (series is **non-stationary**).
  - $H_1$: Series is **stationary** (or trend-stationary).
  - *Decision*: Reject $H_0$ if $p\text{-value} < \alpha \implies$ **Stationary**.
- **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**:
  - $H_0$: Series is **stationary** around a deterministic level/trend.
  - $H_1$: Unit root is present (series is **non-stationary**).
  - *Decision*: Reject $H_0$ if $p\text{-value} < \alpha \implies$ **Non-stationary**.
- **Joint Diagnosis (`method="both"`)**: Runs both tests simultaneously to identify stationary, difference-stationary, trend-stationary, or ambiguous series.

::: {#96c97bae .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def unit_root_test(
    series: ArrayLike,
    method: str = "ADF",
    alpha: float = 0.05,
    nlags: int | str | None = None,
    n_lag: int | None = None,
    regression: str = "c",
    autolag: str | None = "AIC",
    as_df: bool = False,
    verbose: bool = True,
) -> pd.DataFrame | dict:
    """
    Performs a unit root / stationarity test (ADF, KPSS, or both) on the given time series.

    Parameters
    ----------
    series : ArrayLike
        The time series data to be tested (pd.Series, np.ndarray, or list).
    method : str, default "ADF"
        The statistical test to perform:
        - "ADF": Augmented Dickey-Fuller test (H0: unit root present / non-stationary).
        - "KPSS": Kwiatkowski-Phillips-Schmidt-Shin test (H0: stationary around level/trend).
        - "both" or "all": Runs both ADF and KPSS tests for comprehensive diagnosis.
    alpha : float, default 0.05
        Significance level for hypothesis testing (e.g., 0.01, 0.05, 0.10).
    nlags : int, str, or None, default None
        Lag order to include in the test:
        - For ADF: maxlag parameter passed to `adfuller`. If None, autolag is used.
        - For KPSS: nlags parameter passed to `kpss`. Default is "auto".
    n_lag : int or None, default None
        Backward-compatible alias for `nlags`.
    regression : str, default "c"
        Deterministic trend components included in the test:
        - "c": constant only (default).
        - "ct": constant and linear trend.
        - "ctt": constant, linear, and quadratic trend (ADF only).
        - "n": no constant, no trend (ADF only).
    autolag : str or None, default "AIC"
        Method to choose the lag length when `method="ADF"`. Options: "AIC", "BIC", "t-stat", or None.
    as_df : bool, default False
        Whether to return the results as a `pandas.DataFrame`. If False (and method != "both"), returns a dict.
    verbose : bool, default True
        Whether to print a formatted summary of the test results and decision.

    Returns
    -------
    dict or pd.DataFrame
        Detailed statistical test output including test statistic, p-value, critical values,
        lags used, and stationarity conclusion.
    """
    y = _to_clean_1d_array(series, name="series")

    # Handle backward-compatible alias
    if n_lag is not None and nlags is None:
        nlags = n_lag

    method_clean = method.strip().upper()
    if method_clean not in ["ADF", "KPSS", "BOTH", "ALL"]:
        raise ValueError(f"Invalid test method '{method}'. Choose from 'ADF', 'KPSS', or 'both'.")

    def _run_adf(arr, lag_arg, reg_arg, auto_arg, alpha_val):
        maxlag_val = lag_arg if isinstance(lag_arg, int) else None
        res = adfuller(arr, maxlag=maxlag_val, regression=reg_arg, autolag=auto_arg)
        stat, pval, usedlag, nobs, crit_vals, icbest = res[0], res[1], res[2], res[3], res[4], res[5]
        is_stat = bool(pval < alpha_val)
        conclusion = (
            f"Stationary (p-value = {pval:.4f} < alpha = {alpha_val})"
            if is_stat
            else f"Non-stationary (p-value = {pval:.4f} >= alpha = {alpha_val})"
        )
        return {
            "test": "ADF",
            "statistic": float(stat),
            "p_value": float(pval),
            "alpha": float(alpha_val),
            "is_stationary": is_stat,
            "lags": int(usedlag),
            "nobs": int(nobs),
            "critical_values": {k: float(v) for k, v in crit_vals.items()},
            "null_hypothesis": "Unit root is present (Series is non-stationary)",
            "alternative": "Series is stationary / trend-stationary",
            "conclusion": conclusion,
        }

    def _run_kpss(arr, lag_arg, reg_arg, alpha_val):
        kpss_reg = "ct" if reg_arg == "ct" else "c"
        kpss_nlags = "auto" if lag_arg is None else lag_arg
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*p-value is.*")
            stat, pval, usedlag, crit_vals = kpss(arr, regression=kpss_reg, nlags=kpss_nlags)

        is_stat = bool(pval >= alpha_val)
        conclusion = (
            f"Stationary (p-value = {pval:.4f} >= alpha = {alpha_val})"
            if is_stat
            else f"Non-stationary (p-value = {pval:.4f} < alpha = {alpha_val})"
        )
        return {
            "test": "KPSS",
            "statistic": float(stat),
            "p_value": float(pval),
            "alpha": float(alpha_val),
            "is_stationary": is_stat,
            "lags": int(usedlag),
            "nobs": int(len(arr)),
            "critical_values": {k: float(v) for k, v in crit_vals.items()},
            "null_hypothesis": "Series is stationary around level or trend",
            "alternative": "Unit root is present (Series is non-stationary)",
            "conclusion": conclusion,
        }

    results = []
    if method_clean in ["ADF", "BOTH", "ALL"]:
        results.append(_run_adf(y, nlags, regression, autolag, alpha))
    if method_clean in ["KPSS", "BOTH", "ALL"]:
        results.append(_run_kpss(y, nlags, regression, alpha))

    if verbose:
        for r in results:
            status = "STATIONARY" if r["is_stationary"] else "NON-STATIONARY"
            print(
                f"[{r['test']}] Stat: {r['statistic']:.4f} | p-value: {r['p_value']:.4f} "
                f"(alpha={r['alpha']}) | Lags: {r['lags']} -> {status}"
            )

    if method_clean in ["BOTH", "ALL"] or as_df:
        df_rows = []
        for r in results:
            row = {
                "test": r["test"],
                "statistic": r["statistic"],
                "p_value": r["p_value"],
                "alpha": r["alpha"],
                "is_stationary": r["is_stationary"],
                "lags": r["lags"],
                "nobs": r["nobs"],
                "crit_1%": r["critical_values"].get("1%", np.nan),
                "crit_5%": r["critical_values"].get("5%", np.nan),
                "crit_10%": r["critical_values"].get("10%", np.nan),
                "conclusion": r["conclusion"],
            }
            df_rows.append(row)
        return pd.DataFrame(df_rows)

    # return results[0]
```
:::


::: {#a32d38b8 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
# Example: ADF & KPSS tests on stationary vs non-stationary series
np.random.seed(42)
ar1_stationary = np.zeros(200)
for t in range(1, 200):
    ar1_stationary[t] = 0.7 * ar1_stationary[t - 1] + np.random.normal()

# 1. Standard ADF Test
res_adf = unit_root_test(ar1_stationary, method="ADF", alpha=0.05, verbose=True)
assert res_adf["is_stationary"] == True
assert "statistic" in res_adf and "p_value" in res_adf

# 2. KPSS Test
res_kpss = unit_root_test(ar1_stationary, method="KPSS", alpha=0.05, verbose=True)
assert res_kpss["is_stationary"] == True

# 3. Joint Diagnosis (ADF + KPSS) returned as DataFrame
df_both = unit_root_test(ar1_stationary, method="both", alpha=0.05, verbose=False)
assert isinstance(df_both, pd.DataFrame)
df_both
```

::: {.cell-output .cell-output-stdout}
```
[ADF] Stat: -6.4676 | p-value: 0.0000 (alpha=0.05) | Lags: 0 -> STATIONARY
[KPSS] Stat: 0.3298 | p-value: 0.1000 (alpha=0.05) | Lags: 7 -> STATIONARY
```
:::

::: {.cell-output .cell-output-display}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test</th>
      <th>statistic</th>
      <th>p_value</th>
      <th>alpha</th>
      <th>is_stationary</th>
      <th>lags</th>
      <th>nobs</th>
      <th>crit_1%</th>
      <th>crit_5%</th>
      <th>crit_10%</th>
      <th>conclusion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ADF</td>
      <td>-6.467576</td>
      <td>1.392869e-08</td>
      <td>0.05</td>
      <td>True</td>
      <td>0</td>
      <td>199</td>
      <td>-3.463645</td>
      <td>-2.876176</td>
      <td>-2.574572</td>
      <td>Stationary (p-value = 0.0000 &lt; alpha = 0.05)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KPSS</td>
      <td>0.329830</td>
      <td>1.000000e-01</td>
      <td>0.05</td>
      <td>True</td>
      <td>7</td>
      <td>200</td>
      <td>0.739000</td>
      <td>0.463000</td>
      <td>0.347000</td>
      <td>Stationary (p-value = 0.1000 &gt;= alpha = 0.05)</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::


## Cross-Autocorrelation Analysis

`cross_autocorrelation` computes the sample cross-correlation function (CCF) between two time series $x$ and $y$:
$$r_{xy}(k) = \frac{\sum_{t=0}^{n-k-1} (x_{t+k} - \bar{x})(y_t - \bar{y})}{\sqrt{\sum_{t=0}^{n-1}(x_t - \bar{x})^2 \sum_{t=0}^{n-1}(y_t - \bar{y})^2}}$$

### Features:
- **Sample DOF Adjustment**: Scales by $\frac{n}{n - |k|}$ when `adjusted=True` for unbiased covariance estimation.
- **Confidence Intervals**:
  - *Standard White-Noise Band*: $SE_k = \frac{1}{\sqrt{n - |k|}}$ (or $\frac{1}{\sqrt{n}}$).
  - *Bartlett's Formula*: Accounts for serial correlation in $x$ and $y$ to prevent spurious cross-correlation discovery:
    $$\text{Var}(r_{xy}(k)) \approx \frac{1}{n - |k|} \left(1 + 2 \sum_{j=1}^M \hat{\rho}_{xx}(j)\hat{\rho}_{yy}(j)\right)$$
- **Two-Sided Cross-Correlation**: Evaluates both negative and positive lags ($k \in [-nlags, +nlags]$) when `two_sided=True`.

::: {#8922d517 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def cross_autocorrelation(
    x: ArrayLike,
    y: ArrayLike,
    nlags: int = 10,
    adjusted: bool = True,
    alpha: float | None = None,
    bartlett_confint: bool = False,
    two_sided: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute sample cross-autocorrelation between two time series with optional confidence bounds.

    Parameters
    ----------
    x : ArrayLike
        First time series x_t.
    y : ArrayLike
        Second time series y_t.
    nlags : int, default 10
        Maximum lag to compute. Computes lags 0, 1, ..., nlags (or -nlags to +nlags if two_sided=True).
    adjusted : bool, default True
        Whether to apply the sample degree-of-freedom adjustment factor (n / (n - |k|)).
    alpha : float or None, default None
        Significance level for confidence intervals (e.g. 0.05 for 95% CI). If None, no CI is returned.
    bartlett_confint : bool, default False
        Whether to use Bartlett's formula for cross-correlation between autocorrelated processes.
        If False, uses standard white-noise standard error.
    two_sided : bool, default False
        If True, computes cross-correlation for both negative and positive lags [-nlags, +nlags].
        If False, computes forward lags [0, +nlags].

    Returns
    -------
    tuple of (np.ndarray, np.ndarray or None)
        - cc : ndarray of cross-correlation values.
        - confint : ndarray of shape (len(cc), 2) containing [lower, upper] confidence intervals if alpha is provided, else None.
    """
    arr_x = _to_clean_1d_array(x, name="x")
    arr_y = _to_clean_1d_array(y, name="y")
    n = len(arr_x)
    if len(arr_y) != n:
        raise ValueError(f"x and y must have the same length (got len(x)={n} and len(y)={len(arr_y)})")
    if nlags < 0:
        raise ValueError("nlags must be a non-negative integer.")
    if nlags >= n:
        raise ValueError(f"nlags ({nlags}) must be smaller than the series length ({n}).")

    x_mean = np.mean(arr_x)
    y_mean = np.mean(arr_y)
    dx = arr_x - x_mean
    dy = arr_y - y_mean

    var_x = np.sum(dx ** 2)
    var_y = np.sum(dy ** 2)
    denom = np.sqrt(var_x * var_y)
    if denom == 0:
        raise ValueError("Cannot compute cross-correlation for a constant series with zero variance.")

    if two_sided:
        lags_list = np.arange(-nlags, nlags + 1, dtype=int)
    else:
        lags_list = np.arange(0, nlags + 1, dtype=int)

    cc = np.empty(len(lags_list), dtype=float)
    for idx, k in enumerate(lags_list):
        if k >= 0:
            num = np.sum(dy[:n - k] * dx[k:]) if k < n else 0.0
        else:
            m = -k
            num = np.sum(dy[m:] * dx[:n - m]) if m < n else 0.0

        r = num / denom
        if adjusted and k != 0:
            r *= n / (n - abs(k))
        cc[idx] = r

    if alpha is not None:
        z = NormalDist().inv_cdf(1 - alpha / 2)
        abs_lags = np.abs(lags_list)

        if bartlett_confint:
            max_m = min(n - 1, max(nlags, 20))
            acf_x = np.zeros(max_m + 1)
            acf_y = np.zeros(max_m + 1)
            acf_x[0] = 1.0
            acf_y[0] = 1.0
            for j in range(1, max_m + 1):
                acf_x[j] = np.sum(dx[:n - j] * dx[j:]) / var_x
                acf_y[j] = np.sum(dy[:n - j] * dy[j:]) / var_y

            prod_sum = 1.0 + 2.0 * np.sum(acf_x[1:] * acf_y[1:])
            prod_sum = max(prod_sum, 0.0)

            if adjusted:
                se = np.sqrt(prod_sum / np.maximum(n - abs_lags, 1))
            else:
                se = np.sqrt(prod_sum / n) * np.ones_like(cc)
        else:
            if adjusted:
                se = 1.0 / np.sqrt(np.maximum(n - abs_lags, 1))
            else:
                se = (1.0 / np.sqrt(n)) * np.ones_like(cc)

        confint = np.column_stack((cc - z * se, cc + z * se))
        return cc, confint
    else:
        return cc, None
```
:::


::: {#622edb40 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
# Example: Cross-autocorrelation with confidence intervals
np.random.seed(42)
x_sig = np.random.normal(size=150)
y_sig = np.roll(x_sig, 2) + 0.2 * np.random.normal(size=150)

# Forward cross-correlations with Bartlett confidence intervals
cc, confint = cross_autocorrelation(x_sig, y_sig, nlags=5, alpha=0.05, bartlett_confint=True)
assert len(cc) == 6
assert confint.shape == (6, 2)
print("Forward cross-correlations (lags 0-5):", np.round(cc, 3))

# Two-sided cross-correlations (lags -5 to +5)
cc_2s, confint_2s = cross_autocorrelation(x_sig, y_sig, nlags=5, alpha=0.05, two_sided=True)
assert len(cc_2s) == 11
print("Two-sided cross-correlations (lags -5 to +5):", np.round(cc_2s, 3))
```

::: {.cell-output .cell-output-stdout}
```
Forward cross-correlations (lags 0-5): [-0.047 -0.041 -0.085  0.139 -0.014  0.111]
Two-sided cross-correlations (lags -5 to +5): [ 0.001 -0.049 -0.061  0.986 -0.098 -0.047 -0.041 -0.085  0.139 -0.014
  0.111]
```
:::
:::


## PACF Strength and Feature Lag Selection

`pacf_strength` computes exceedance strength scores for Partial Autocorrelation Function (PACF) values against critical statistical bounds:
$$\text{Score}_k = \begin{cases} \frac{\phi_{kk} - B_k}{B_k} & \text{if } \phi_{kk} > B_k \\ \frac{\phi_{kk} + B_k}{B_k} & \text{if } \phi_{kk} < -B_k \\ 0 & \text{otherwise} \end{cases}$$
where $B_k = \frac{z_{1-\alpha/2}}{\sqrt{n-k}}$ (adjusted) or $\frac{z_{1-\alpha/2}}{\sqrt{n}}$.

- **Feature Lag Identification**: Automatically ranks candidate autoregressive lags by statistical significance.
- **Lag 0 Exclusion**: Lag 0 is strictly excluded to prevent selecting instantaneous self-correlation.

::: {#9eaa3416 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def pacf_strength(
    series: ArrayLike,
    alpha: float = 0.05,
    n_lags: int = 5,
    adjusted: bool = True,
    method: str = "ywm",
    significant_only: bool = True,
) -> pd.DataFrame:
    """
    Calculate the exceedance strength scores for the Partial Autocorrelation Function (PACF)
    to identify statistically significant feature lags.

    For positive exceedance: score = (PACF - bound) / bound
    For negative exceedance: score = (PACF + bound) / bound
    Non-significant lags have score = 0.

    Parameters
    ----------
    series : ArrayLike
        Input time series data.
    alpha : float, default 0.05
        Significance level for the critical confidence threshold.
    n_lags : int, default 5
        Maximum number of lags to evaluate (lags 1 through n_lags).
    adjusted : bool, default True
        Whether to use lag-specific adjusted degrees-of-freedom bound (z / sqrt(n - k))
        or fixed bound (z / sqrt(n)).
    method : str, default "ywm"
        PACF calculation method passed to statsmodels `pacf` ('ywm', 'ywadjusted', 'ols', 'ld', 'ldb').
    significant_only : bool, default True
        Whether to return only statistically significant lags with score != 0.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by `abs_pacf_score` descending, containing:
        - `lags`: lag order (1, 2, ...)
        - `pacf_score`: exceedance score (+ for positive exceedance, - for negative)
        - `pacf_value`: raw PACF value
        - `z_bound`: critical threshold value
        - `abs_pacf_score`: magnitude of the exceedance score
    """
    y = _to_clean_1d_array(series, name="series")
    n = len(y)
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1.")
    if n_lags >= n:
        raise ValueError(f"n_lags ({n_lags}) must be smaller than the series length ({n}).")

    pacf_vals = pacf(y, nlags=n_lags, method=method)

    # Exclude lag 0 immediately: select lags 1 to n_lags
    lags = np.arange(1, n_lags + 1, dtype=int)
    pacf_sub = pacf_vals[1:n_lags + 1]

    z = NormalDist().inv_cdf(1 - alpha / 2)
    bounds = (z / np.sqrt(n - lags)) if adjusted else (z / np.sqrt(n) * np.ones_like(lags, dtype=float))

    scores = np.zeros_like(pacf_sub, dtype=float)
    pos_mask = pacf_sub > bounds
    neg_mask = pacf_sub < -bounds

    scores[pos_mask] = (pacf_sub[pos_mask] - bounds[pos_mask]) / bounds[pos_mask]
    scores[neg_mask] = (pacf_sub[neg_mask] + bounds[neg_mask]) / bounds[neg_mask]

    df = pd.DataFrame({
        "lags": lags,
        "pacf_score": scores,
        "pacf_value": pacf_sub,
        "z_bound": bounds,
        "abs_pacf_score": np.abs(scores)
    })

    if significant_only:
        df = df[df["abs_pacf_score"] > 0]

    df = df.sort_values(by="abs_pacf_score", ascending=False).reset_index(drop=True)
    return df
```
:::


::: {#fe9fc0f6 .cell}
``` {.python .cell-code}
# Example: Identify significant lags in AR(2) process
np.random.seed(42)
ar2_data = np.zeros(300)
for t in range(2, 300):
    ar2_data[t] = 0.6 * ar2_data[t - 1] - 0.35 * ar2_data[t - 2] + np.random.normal()

sig_lags = pacf_strength(ar2_data, alpha=0.05, n_lags=6)
assert 0 not in sig_lags["lags"].values
assert len(sig_lags) >= 2
sig_lags
```

::: {.cell-output .cell-output-display}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lags</th>
      <th>pacf_score</th>
      <th>pacf_value</th>
      <th>z_bound</th>
      <th>abs_pacf_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2.491912</td>
      <td>0.395800</td>
      <td>0.113348</td>
      <td>2.491912</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-2.076277</td>
      <td>-0.349273</td>
      <td>0.113538</td>
      <td>2.076277</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0.125441</td>
      <td>0.128428</td>
      <td>0.114114</td>
      <td>0.125441</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::


## Cross-Correlation Strength

`ccf_strength` identifies significant cross-correlation lags between a target series $y$ and an exogenous predictor $x$.

::: {#ccaceca0 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def ccf_strength(x: ArrayLike, y: ArrayLike, alpha: float = 0.05, n_lags: int = 5, adjusted: bool = True) -> pd.DataFrame:
    """
    Calculate exceedance scores for the cross-correlation function (CCF).

    Parameters
    ----------
    x, y : ArrayLike
        Input time series.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    n_lags : int, default 5
        Number of lags to consider.
    adjusted : bool, default True
        If True, use lag-specific CI (sqrt(n-k)); if False, use fixed CI (sqrt(n)).

    Returns
    -------
    pd.DataFrame
        Exceedance scores for each lag (excluding lag 0).
    """
    n = len(y)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    ccr_values = ccf(x, y)[: n_lags + 1]

    exceed_score = []
    for k, j in enumerate(ccr_values):
        if adjusted:
            bound = z / np.sqrt(n - k)
        else:
            bound = z / np.sqrt(n)

        if j > bound:
            exceed_score.append([k, (j - bound) / bound, j, bound])
        elif j < -bound:
            exceed_score.append([k, (j + bound) / bound, j, -bound])
        else:
            exceed_score.append([k, 0, j, bound])

    exceed_score = pd.DataFrame(exceed_score, columns=["lags", "corr_score", "ccf_value", "z_bound"])
    exceed_score["abs_corr_score"] = exceed_score["corr_score"].abs()

    # drop lag 0 (auto-correlation with itself)
    exceed_score = exceed_score.loc[exceed_score["lags"] != 0]
    exceed_score = exceed_score[exceed_score["abs_corr_score"] > 0]
    exceed_score = exceed_score.sort_values(by="abs_corr_score", ascending=False).reset_index(drop=True)

    return exceed_score
```
:::


## Trend Modeling and Decomposition Diagnostics

Linear regression trend fitting, piecewise linear hinge trend modeling, trend forecasting, and STL-based trend/seasonality strength metrics based on Hyndman's formulas.

::: {#9488a55b .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def lr_trend_model(series: ArrayLike, breakpoints: list | None = None, type: str = 'linear', degree: int | list = 1):
    """
    Compute the linear or piecewise trend of a time series using linear regression.

    Parameters
    ----------
    series : pd.Series or ArrayLike
        The input time series.
    breakpoints : list, optional
        A list of breakpoints for piecewise segments when type is "piecewise".
    type : str, default 'linear'
        The type of model ("linear" or "piecewise").
    degree : int or list, default 1
        The degree of the polynomial trend when type is "linear".

    Returns
    -------
    tuple
        (trend, model_lr, X_trend) - Fitted trend values, LinearRegression model, and design time index matrix.
    """
    T = np.arange(len(series), dtype=int)

    if type == 'piecewise' and breakpoints is not None:
        X_trend = T.reshape(-1, 1)
        for bp in breakpoints:
            hinge = np.maximum(0, T - bp).reshape(-1, 1)
            X_trend = np.hstack([X_trend, hinge])

        model_lr = LinearRegression().fit(X_trend, np.array(series))
        trend = model_lr.predict(X_trend)
    else:
        if degree == 1:
            X_trend = T.reshape(-1, 1)
        elif degree != 1 and isinstance(degree, int):
            X_trend = np.column_stack([T**i for i in range(1, degree + 1)])
        elif isinstance(degree, list):
            X_trend = np.column_stack([T**i for i in degree])
        else:
            raise ValueError("Degree must be a positive integer or list of degrees.")
        model_lr = LinearRegression().fit(X_trend, np.array(series))
        trend = model_lr.predict(X_trend)
    return trend, model_lr, X_trend


def forecast_trend(model, H: int, start: int, degree: int | list = 1, breakpoints: list | None = None):
    """
    Forecast future trend values using the fitted linear regression trend model.

    Parameters
    ----------
    model : LinearRegression
        The fitted linear regression model.
    H : int
        The forecast horizon.
    start : int
        The starting point (time index) for the forecast.
    degree : int or list, default 1
        The degree of the polynomial trend.
    breakpoints : list, optional
        A list of breakpoints for the piecewise segments.

    Returns
    -------
    tuple
        (trend_forecast, X_future) - Forecasted trend values and future design matrix.
    """
    TH = np.arange(start, start + H, dtype=int)
    if degree == 1:
        T_future = TH.reshape(-1, 1)
    elif degree != 1 and isinstance(degree, int):
        T_future = np.column_stack([TH**i for i in range(1, degree + 1)])
    elif isinstance(degree, list):
        T_future = np.column_stack([TH**i for i in degree])
    else:
        raise ValueError("Degree must be a positive integer or list of degrees.")

    X_future = T_future
    if breakpoints is not None:
        for bp in breakpoints:
            hinge = np.maximum(0, T_future - bp).reshape(-1, 1)
            X_future = np.hstack([X_future, hinge])

    return model.predict(X_future), X_future


def trend_strength(series: ArrayLike, **kwargs) -> float:
    """
    Compute the strength of the trend component in a time series using Hyndman's formula.
    1 indicates strong trend, 0 indicates no trend.

    Parameters
    ----------
    series : pd.Series or ArrayLike
        The time series data.
    **kwargs : dict
        Additional keyword arguments passed to STL decomposition.

    Returns
    -------
    float
        Trend strength in [0, 1].
    """
    res = STL(series, **kwargs).fit()
    var_resid = np.var(res.resid)
    var_detrended = np.var(res.resid + res.trend)
    if var_detrended == 0:
        return 0.0
    return float(max(0.0, 1.0 - var_resid / var_detrended))


def seasonality_strength(series: ArrayLike, **kwargs) -> float:
    """
    Compute the strength of the seasonal component in a time series using Hyndman's formula.
    1 indicates strong seasonality, 0 indicates no seasonality.

    Parameters
    ----------
    series : pd.Series or ArrayLike
        The time series data.
    **kwargs : dict
        Additional keyword arguments passed to STL decomposition.

    Returns
    -------
    float
        Seasonal strength in [0, 1].
    """
    res = STL(series, **kwargs).fit()
    var_resid = np.var(res.resid)
    var_deseasonal = np.var(res.resid + res.seasonal)
    if var_deseasonal == 0:
        return 0.0
    return float(max(0.0, 1.0 - var_resid / var_deseasonal))
```
:::


::: {#caa1cf96 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
# Example: Trend & Seasonality Strength
t = np.linspace(0, 10, 200)
seasonal_signal = 10 * np.sin(2 * np.pi * t) + 2 * t + np.random.normal(scale=0.5, size=200)
s_strength = seasonality_strength(pd.Series(seasonal_signal), period=20)
t_strength = trend_strength(pd.Series(seasonal_signal), period=20)
print(f"Seasonality strength: {s_strength:.4f}, Trend strength: {t_strength:.4f}")
assert 0.0 <= s_strength <= 1.0
assert 0.0 <= t_strength <= 1.0
```

::: {.cell-output .cell-output-stdout}
```
Seasonality strength: 0.9969, Trend strength: 0.9955
```
:::
:::


::: {#0700ed9b .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
unit_root_test(seasonal_signal, method = "KPSS", as_df=True)
```

::: {.cell-output .cell-output-stdout}
```
[KPSS] Stat: 1.2040 | p-value: 0.0100 (alpha=0.05) | Lags: 9 -> NON-STATIONARY
```
:::

::: {.cell-output .cell-output-display}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test</th>
      <th>statistic</th>
      <th>p_value</th>
      <th>alpha</th>
      <th>is_stationary</th>
      <th>lags</th>
      <th>nobs</th>
      <th>crit_1%</th>
      <th>crit_5%</th>
      <th>crit_10%</th>
      <th>conclusion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KPSS</td>
      <td>1.20399</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>False</td>
      <td>9</td>
      <td>200</td>
      <td>0.739</td>
      <td>0.463</td>
      <td>0.347</td>
      <td>Non-stationary (p-value = 0.0100 &lt; alpha = 0.05)</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::


