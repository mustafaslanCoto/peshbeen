# Peshbeen: Introduction, Installation Guide, Quick Start, Examples, Documentation and API Reference


## Introduction

Peshbeen is a easy-to-use Python library designed for time series
forecasting. It provides a simple and intuitive interface for building,
training, and evaluating forecasting models. It automatically handles
data preprocessing specific to time series data, making it easier for
users to focus on model development and evaluation. Peshbeen supports a
variety of forecasting models, including traditional statistical models
and modern machine learning approaches, including Linear Regression,
Random Forest, XGBoost, LightGBM, CatBoost, and Cubist.

## Installation Guide

To install Peshbeen, you can use pip and conda package managers.

For pip, run the following command in your terminal:

``` bash
pip install peshbeen
```

For conda, you can use:

``` bash
conda install -c conda-forge peshbeen
```

<!-- # Probabilistic forecasting wrapper for any point-forecasting model.
&#10;- Wraps ``arima``, ``ets``, ``ml_forecaster``, ``naive_forecaster``, and any other model that exposes ``.fit(df)`` and ``.forecast(H, exog)`` methods.
&#10;- Provides four methods for producing prediction intervals or full predictive distributions:
&#10;1. **Conformal prediction** (``calibrate`` → ``predict_intervals`` ``conformal_quantiles``): distribution-free, theoretically guaranteed coverage using split-conformal scores.
&#10;2. **Empirical bootstrap** (``sample(method="empirical")``): residuals are resampled with replacement; each draw is added to the point forecast to produce a sample path.
&#10;3. **KDE bootstrap** (``sample(method="kde")``): a kernel-density estimate is fitted to each horizon's residuals; samples are drawn from the KDE for a smooth approximation of the predictive distribution.
&#10;4. **Correlated bootstrap** (``sample(method="correlated")``): a multivariate normal is fitted to the full (H-dimensional) residual vectors, preserving the correlation structure across forecast horizons. Samples are drawn jointly.
&#10;All sampling methods populate ``self.sample_paths`` (shape
``(n_samples, H)``) and ``self.point_forecast``, so
``sample_quantiles`` works identically regardless of the method used.
&#10;# Draw sample paths from the predictive distribution.
&#10;# Three methods are available:
&#10;# * ``"empirical"`` — residuals are resampled with replacement
#     independently at each horizon.
# * ``"kde"`` — a Gaussian KDE is fitted to each horizon's residuals;
#     samples are drawn from the smoothed distribution.
# * ``"correlated"`` — a multivariate normal is fitted to the full
#     ``H``-dimensional residual vectors, preserving cross-horizon
#     correlation.  Samples are drawn jointly.
&#10;# Results are stored on ``self``:
&#10;# * ``self.sample_paths`` — ``(n_samples, H)`` array of sampled
#     trajectories centred on the point forecast.
# * ``self.point_forecast`` — ``(H,)`` point forecast array.
# * ``self.sample_paths_df`` — the same data as a DataFrame with
#     columns ``h_1, …, h_H`` -->
