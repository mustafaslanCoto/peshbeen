---
title: Forecasters
---


::: {#4143e5bf .cell}
``` {.python .cell-code}

#| hide

import inspect
from IPython.display import display, Markdown

from peshbeen.models import (naive, ets, arima, ml_forecaster, glm, var, ms_arr, pesh, ml_mv_forecaster)
#| hide
from fastcore.docments import docments, DocmentTbl
from nbdev.showdoc import *

def render_api(obj):
    """
    Generates a clean, Sphinx-style API reference with multi-line 
    arguments and deduplicated module paths.
    """
    # 1. Get module and class/function names
    mod = getattr(obj, '__module__', '')
    qualname = getattr(obj, '__qualname__', obj.__name__)
    
    # Clean up __init__ to just show the class name
    is_init = obj.__name__ == "__init__"
    if is_init:
        qualname = qualname.replace(".__init__", "")
        base_name = qualname.split(".")[-1]
        func_name = f"{base_name}"
    else:
        base_name = obj.__name__
        func_name = f"{base_name}"
        
    # 2. Deduplicate the path (fixes peshbeen.models.ml_forecaster.ml_forecaster)
    path_parts = mod.split('.') + qualname.split('.')
    clean_parts = []
    for part in path_parts:
        if not clean_parts or clean_parts[-1] != part:
            clean_parts.append(part)
            
    clean_path = ".".join(clean_parts)
    
    # 3. Extract the exact Python signature and format it line-by-line
    try:
        sig = inspect.signature(obj)
        params = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue  # Hide 'self' from the documentation
            params.append(f"    {param}")
            
        if params:
            param_str = ",\n".join(params)
            sig_str = f"{func_name}(\n{param_str}\n)"
        else:
            sig_str = f"{func_name}()"
    except ValueError:
        sig_str = func_name
        
    # 4. Extract just the main description from your docstring
    doc = inspect.getdoc(obj) or ""
    description = doc.split("Parameters")[0].strip()
    
    # 5. Build the exact Markdown structure and render
    md_content = f"### {clean_path}\n\n```python\n{sig_str}\n```\n\n{description}"
    
    display(Markdown(md_content))
    display(DocmentTbl(obj))
```
:::




## Naive forecaster

::: {#51c80fe7 .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.naive

```python
naive(
    target_col: 'str',
    season_period: 'Optional[int]' = None,
    box_cox: 'Union[bool, float]' = False,
    box_cox_biasadj: 'bool' = False
)
```

Naïve forecaster.

Two modes controlled by ``season_period``:

* **Non-seasonal** (``season_period=None``): every forecast step repeats
the last observed value in the training series.
* **Seasonal** (``season_period=m``): forecast values are taken from the
last complete season and cycled forward — i.e. step ``h`` is predicted
by ``y[T - m + ((h-1) % m)]``, where ``T`` is the last training index.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| target_col | str |  | Name of the target variable column. |
| season_period | Optional[int] | None | Seasonal period ``m``.  ``None`` selects the non-seasonal naïve method.  When provided and the training series is shorter than ``m``, ``forecast`` returns an array of ``NaN``. |
| box_cox | Union[bool, float] | False | Whether to apply Box-Cox transformation to the target variable. If a float value is provided, it will be used as the lambda parameter for the Box-Cox transformation. If True, the lambda parameter will be estimated from the data. |
| box_cox_biasadj | bool | False | Bias adjustment when inverting the manual Box-Cox on forecasts. |
| **Returns** | **None** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.naive.fit

```python
fit(
    df: 'pd.DataFrame'
)
```

Store the values needed for naïve forecasting.

No statistical model is estimated.  ``fit`` simply applies
``data_prep`` and records the training series so that ``forecast``
can replicate the correct naïve pattern.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| df | pd.DataFrame | Training DataFrame containing the target column. |
| **Returns** | **None** |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.naive.forecast

```python
forecast(
    H: 'int',
    exog: 'Optional[pd.DataFrame]' = None
)
```

Generate naïve forecasts.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| H | int |  | Forecast horizon. |
| exog | Optional[pd.DataFrame] | None | Accepted for API consistency with other models but silently ignored — naïve forecasts do not use exogenous variables. |
| **Returns** | **np.ndarray** |  | **Forecast values of length `H`.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.naive.cross_validate

```python
cross_validate(
    df: 'pd.DataFrame',
    cv_split: 'int',
    test_size: 'int',
    metrics: 'List[Callable]',
    step_size: 'int' = 1,
    h_split_point: 'Optional[int]' = None
)
```

Run time-series cross-validation.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | Full dataset. |
| cv_split | int |  | Number of CV folds. |
| test_size | int |  | Test window size per fold. |
| metrics | List[Callable] |  | Metric functions (e.g. ``[MAE, RMSE]``) used to evaluate forecast accuracy across folds. Call ``.cv_summary()`` after cross-validation to retrieve the aggregated scores. |
| step_size | int | 1 | Step size to advance the test window each fold. |
| h_split_point | Optional[int] | None | Split the test window into two sub-horizons for separate short- and long-term evaluation. |
| **Returns** | **Tuple[pd.DataFrame, pd.DataFrame]** |  | **Summary DataFrame with mean metric scores across folds, and (optionally) a fold-level DataFrame with true vs. predicted values for each fold.** |
:::
:::


## ETS (Error, Trend, Seasonality) forecaster

::: {#fbb72c67 .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ets

```python
ets(
    target_col: 'str',
    trend: 'Optional[str]' = None,
    damped_trend: 'bool' = False,
    seasonal: 'Optional[str]' = None,
    seasonal_periods: 'Optional[int]' = None,
    initialization_method: 'Optional[str]' = 'estimated',
    initial_level: 'Optional[float]' = None,
    initial_trend: 'Optional[float]' = None,
    initial_seasonal: 'Optional[list]' = None,
    bounds: 'Optional[dict]' = None,
    dates=None,
    freq: 'Optional[str]' = None,
    missing: 'str' = 'none',
    optimized: 'bool' = True,
    smoothing_level: 'Optional[float]' = None,
    smoothing_trend: 'Optional[float]' = None,
    smoothing_seasonal: 'Optional[float]' = None,
    damping_trend: 'Optional[float]' = None,
    remove_bias: 'bool' = False,
    start_params=None,
    method: 'Optional[str]' = None,
    minimize_kwargs: 'Optional[dict]' = None,
    use_brute: 'bool' = True,
    box_cox: 'Union[bool, float]' = False,
    box_cox_biasadj: 'bool' = False,
    fit_kwargs: 'Optional[dict]' = None
)
```

Holt-Winters Exponential Smoothing forecaster.

A thin wrapper around ``statsmodels.tsa.holtwinters.ExponentialSmoothing``.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| target_col | str |  | Name of the target variable column. |
| trend | Optional[str] | None | Trend component type. |
| damped_trend | bool | False | Whether to damp the trend.  Only meaningful when ``trend`` is not `None`. |
| seasonal | Optional[str] | None | Seasonal component type. |
| seasonal_periods | Optional[int] | None | Number of periods in a complete seasonal cycle — e.g. 12 for monthly data with an annual cycle.  Required when ``seasonal`` is not ``None``. |
| initialization_method | Optional[str] | estimated | How to initialise the recursions.  When `"known"` is chosen, `initial_level` (and `initial_trend` / `initial_seasonal` where applicable) must also be provided. |
| initial_level | Optional[float] | None | Initial level value.  Required when `initialization_method=`"known"`. |
| initial_trend | Optional[float] | None | Initial trend value.  Required when `initialization_method=`"known"` and the model has a trend component. |
| initial_seasonal | Optional[list] | None | Initial seasonal factors (length `seasonal_periods` or ``seasonal_periods - 1``).  Required when ``initialization_method="known"`` and the model is seasonal. |
| bounds | Optional[dict] | None | Parameter bounds passed to ``ExponentialSmoothing``, e.g. `{"smoothing_level": (0, 1)}`. |
| dates | NoneType | None | Datetime index for the series.  Inferred automatically when `endog` is a Pandas object with a `DatetimeIndex`. |
| freq | Optional[str] | None | Frequency of the time series (e.g. ``"M"``, ``"D"``).  Optional when `dates` is provided. |
| missing | str | none | How to handle ``NaN`` values in the input series. |
| optimized | bool | True | Estimate smoothing parameters by maximising the log-likelihood. |
| smoothing_level | Optional[float] | None | Fixed alpha value.  When set, this value is used directly and not optimised. |
| smoothing_trend | Optional[float] | None | Fixed beta value.  Only used when the model has a trend component. |
| smoothing_seasonal | Optional[float] | None | Fixed gamma value.  Only used when the model is seasonal. |
| damping_trend | Optional[float] | None | Fixed phi (damping) value.  Only used when ``damped_trend=True``. |
| remove_bias | bool | False | Remove bias from forecast values by enforcing that the mean residual is zero. |
| start_params | NoneType | None | Starting parameter values for the optimiser. |
| method | Optional[str] | None | Optimisation method — one of `"L-BFGS-B"` (default), `"TNC"`, `"SLSQP"`, `"Powell"`, `"trust-constr"`, `"basinhopping"` (alias `"bh"`), or `"least_squares"` (alias `"ls"`). |
| minimize_kwargs | Optional[dict] | None | Extra keyword arguments forwarded to the chosen SciPy minimiser. |
| use_brute | bool | True | Search for good starting values with a brute-force grid search before running the main optimiser. |
| box_cox | Union[bool, float] | False | Whether to apply Box-Cox transformation to the target variable. If a float value is provided, it will be used as the lambda parameter for the Box-Cox transformation. If True, the lambda parameter will be estimated from the data. |
| box_cox_biasadj | bool | False | Bias adjustment when inverting the manual Box-Cox on forecasts. |
| fit_kwargs | Optional[dict] | None | Any additional keyword arguments forwarded verbatim to `ExponentialSmoothing.fit`. |
| **Returns** | **None** |  | **Fitted model object with a ``forecast`` method for making predictions and properties for information criteria scores (AIC, BIC, etc.)** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ets.fit

```python
fit(
    df: 'pd.DataFrame'
)
```

Fit ``ExponentialSmoothing`` to the training data.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| df | pd.DataFrame | Training DataFrame containing the target column |
| **Returns** | **None** |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ets.forecast

```python
forecast(
    H: 'int',
    exog: 'Optional[pd.DataFrame]' = None
)
```

Multi-step forecast.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| H | int |  | Forecast horizon. |
| exog | Optional[pd.DataFrame] | None | Accepted for API consistency with other models but silently ignored — ETS forecasts do not use exogenous variables. |
| **Returns** | **np.ndarray** |  | **Forecast values of length `H`.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ets.cross_validate

```python
cross_validate(
    df: 'pd.DataFrame',
    cv_split: 'int',
    test_size: 'int',
    metrics: 'List[Callable]',
    step_size: 'int' = 1,
    h_split_point: 'Optional[int]' = None
)
```

Run time-series cross-validation.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | Full dataset |
| cv_split | int |  | Number of CV folds. |
| test_size | int |  | Test window size per fold. |
| metrics | List[Callable] |  | Metric functions (e.g. ``[MAE, RMSE]``) used to evaluate forecast accuracy across folds. Call ``.cv_summary()`` after cross-validation to retrieve the aggregated scores. |
| step_size | int | 1 | Step size to advance the test window each fold. |
| h_split_point | Optional[int] | None | Split the test window into two sub-horizons for separate short- and long-term evaluation. |
| **Returns** | **Tuple[pd.DataFrame, pd.DataFrame]** |  | **Summary DataFrame with mean metric scores across folds, and (optionally) a fold-level DataFrame with true vs. predicted values for each fold.** |
:::
:::


## ARIMA forecaster

::: {#d9bd4abf .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.arima

```python
arima(
    target_col: 'str',
    order: 'Optional[Tuple[int, int, int]]' = (0, 0, 0),
    seasonal_order: 'Optional[Tuple[int, int, int]]' = (0, 0, 0),
    seasonal_length: 'Optional[int]' = 1,
    lag_transform: 'Optional[list]' = None,
    trend: 'Optional[str]' = None,
    pol_degree: 'int' = 1,
    ets_params: 'Optional[Dict[str, Any]]' = None,
    change_points: 'Optional[List[int]]' = None,
    box_cox: 'Union[bool, float, int]' = False,
    box_cox_biasadj: 'bool' = False,
    cat_variables: 'Optional[List[str]]' = None,
    categorical_encoder: 'Optional[Any]' = None,
    target_encode: 'bool' = False
)
```

Initialize the arima model with the specified parameters and configurations.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| target_col | str |  | Name of the target variable column in the input DataFrame. |
| order | Optional[Tuple[int, int, int]] | (0, 0, 0) | The (p, d, q) order of the ARIMA model. Default is (0, 0, 0). |
| seasonal_order | Optional[Tuple[int, int, int]] | (0, 0, 0) | The (P, D, Q) order of the seasonal ARIMA model. Default is (0, 0, 0). |
| seasonal_length | Optional[int] | 1 | The seasonal period for the seasonal ARIMA model. Default is 1. |
| lag_transform | Optional[list] | None | List of lag-transform function objects to apply to the target variable (e.g. [expanding_mean(shift=1), rolling_std(window_size=3, shift=1)]). Each function should take a pandas Series as input and return a Series of the same length. Default is None (no lag transforms). |
| trend | Optional[str] | None | Trend strategy to use. Options are 'linear' for linear trend removal, 'ets' for ETS-based trend removal, 'feature_lr' for using linear trend components as features, and 'feature_ets' for using ETS trend components as features. Default is None (no trend handling). |
| pol_degree | int | 1 | Degree of polynomial trend to fit when using 'linear' or 'feature_lr' trend strategy. Default is 1 (linear trend). |
| ets_params | Optional[Dict[str, Any]] | None | Dictionary of parameters for the ExponentialSmoothing model when using 'ets' trend strategy. The keys should be the parameter names and the values should be the parameter values. Default is None (use default ETS parameters). |
| change_points | Optional[List[int]] | None | List of indices in the time series where change points occur for piecewise linear trend fitting. Only used when trend strategy is 'linear' or 'feature_lr'. Default is None (no change points, fit a single linear trend). |
| box_cox | Union[bool, float, int] | False | Whether to apply Box-Cox transformation to the target variable. If a float or int value is provided, it will be used as the lambda parameter for the Box-Cox transformation. If True, the lambda parameter will be estimated from the data. |
| box_cox_biasadj | bool | False | Whether to apply bias adjustment when inverting the Box-Cox transformation on forecasts. Default is False. |
| cat_variables | Optional[List[str]] | None | List of categorical feature column names. If provided, these columns will be treated as categorical variables and encoded accordingly. Default is None (no categorical variables). |
| categorical_encoder | Optional[Any] | None | Categorical encoder object (e.g. OneHotEncoder(), MeanEncoder(), etc.) to apply to the categorical variables specified in cat_variables. The encoder should have fit() and transform() methods that can be applied to the input DataFrame. Default is None (no categorical encoding) and if None, categorical variables can only be used if the model can handle them natively (e.g. LGBM or CatBoost). |
| target_encode | bool | False |  |
| **Returns** | **None** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.arima.fit

```python
fit(
    df: 'pd.DataFrame'
)
```

Fit the model to the training data by applying the specified data preparation steps and then fitting the ARIMA model.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| df | pd.DataFrame | Training DataFrame containing the target and any feature columns. |
| **Returns** | **None** |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.arima.forecast

```python
forecast(
    H: 'int',
    exog: 'Optional[pd.DataFrame]' = None
)
```

Recursive multi-step forecast.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| H | int |  | Forecast horizon. |
| exog | Optional[pd.DataFrame] | None | Optional dataframe of future regressors. |
| **Returns** | **np.ndarray** |  | **Forecast values of length `H`.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.arima.cross_validate

```python
cross_validate(
    df: 'pd.DataFrame',
    cv_split: 'int',
    test_size: 'int',
    metrics: 'List[Callable]',
    step_size: 'int' = 1,
    h_split_point: 'Optional[int]' = None
)
```

Run cross-validation using time series splits.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | DataFrame containing the target and any feature columns. |
| cv_split | int |  | Number of cross-validation splits. |
| test_size | int |  | Number of periods in each test set. |
| metrics | List[Callable] |  | Metric functions (e.g. ``[MAE, RMSE]``) used to evaluate forecast accuracy across folds. Call ``.cv_summary()`` after cross-validation to retrieve the aggregated scores. |
| step_size | int | 1 | Step size to move the test window forward in each split. |
| h_split_point | Optional[int] | None | Optional index to split the test set into two parts for separate evaluation (e.g. to evaluate short-term vs long-term performance). If None, no split is done. |
| **Returns** | **Tuple[pd.DataFrame, pd.DataFrame]** |  | **DataFrame containing overall performance metrics averaged across splits, and a DataFrame with predictions and true values for each split.** |
:::
:::


## Machine learning forecaster

::: {#c5dd4666 .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ml_forecaster

```python
ml_forecaster(
    model: 'Any',
    target_col: 'str',
    lags: 'Optional[Union[int, List[int]]]' = None,
    lag_transform: 'Optional[list]' = None,
    difference: 'Optional[int]' = None,
    seasonal_diff: 'Optional[int]' = None,
    trend: 'Optional[str]' = None,
    pol_degree: 'int' = 1,
    ets_params: 'Optional[Dict[str, Any]]' = None,
    change_points: 'Optional[List[int]]' = None,
    box_cox: 'Union[bool, float, int]' = False,
    box_cox_biasadj: 'bool' = False,
    cat_variables: 'Optional[List[str]]' = None,
    categorical_encoder: 'Optional[Any]' = None
)
```

Initialize the ml_forecaster with the specified model and preprocessing options.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | Any |  | A regression model object (e.g. LGBMRegressor(), XGBRegressor(), CatBoostRegressor(), LinearRegression(), etc.) |
| target_col | str |  | Name of the target variable column in the input DataFrame. |
| lags | Optional[Union[int, List[int]]] | None | Lags to include as features. If an integer is provided, lags from 1 to that integer will be included. If a list of integers is provided, those specific lags will be included. Default is None (no lag features). |
| lag_transform | Optional[list] | None | List of lag-transform function objects to apply to the target variable (e.g. [expanding_mean(shift=1), rolling_std(window_size=3, shift=1)]). Each function should take a pandas Series as input and return a Series of the same length. Default is None (no lag transforms). |
| difference | Optional[int] | None | Order of ordinary differencing to apply to the target variable (e.g. 1 for first difference). Default is None (no differencing). |
| seasonal_diff | Optional[int] | None | Seasonal period for seasonal differencing (e.g. 12 for monthly data with yearly seasonality). Default is None (no seasonal differencing). |
| trend | Optional[str] | None | Trend strategy to use. Options are 'linear' for linear trend removal, 'ets' for ETS-based trend removal, 'feature_lr' for using linear trend components as features, and 'feature_ets' for using ETS trend components as features. Default is None (no trend handling). |
| pol_degree | int | 1 | Degree of polynomial trend to fit when using 'linear' or 'feature_lr' trend strategy. Default is 1 (linear trend). |
| ets_params | Optional[Dict[str, Any]] | None | Dictionary of parameters for the ExponentialSmoothing model when using 'ets' trend strategy. The keys should be the parameter names and the values should be the parameter values. Default is None (use default ETS parameters). |
| change_points | Optional[List[int]] | None | List of indices in the time series where change points occur for piecewise linear trend fitting. Only used when trend strategy is 'linear' or 'feature_lr'. Default is None (no change points, fit a single linear trend). |
| box_cox | Union[bool, float, int] | False | Whether to apply Box-Cox transformation to the target variable. If a float or int value is provided, it will be used as the lambda parameter for the Box-Cox transformation. If True, the lambda parameter will be estimated from the data. |
| box_cox_biasadj | bool | False | Whether to apply bias adjustment when inverting the Box-Cox transformation on forecasts. Default is False. |
| cat_variables | Optional[List[str]] | None | List of categorical feature column names. If provided, these columns will be treated as categorical variables and encoded accordingly. Default is None (no categorical variables). |
| categorical_encoder | Optional[Any] | None | Categorical encoder object (e.g. OneHotEncoder(), MeanEncoder(), etc.) to apply to the categorical variables specified in cat_variables. The encoder should have fit() and transform() methods that can be applied to the input DataFrame. Default is None (no categorical encoding) and if None, categorical variables can only be used if the model can handle them natively (e.g. LGBM or CatBoost). |
| **Returns** | **None** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ml_forecaster.fit

```python
fit(
    df: 'pd.DataFrame'
)
```

Fit the model to the training data after applying the specified data preparation steps.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| df | pd.DataFrame | Training DataFrame containing the target and any feature columns. |
| **Returns** | **None** |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ml_forecaster.forecast

```python
forecast(
    H: 'int',
    exog: 'Optional[pd.DataFrame]' = None
)
```

Recursive multi-step forecast.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| H | int |  | Forecast horizon. |
| exog | Optional[pd.DataFrame] | None | Optional dataframe of future regressors. |
| **Returns** | **np.ndarray** |  | **Forecast values of length `H`.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ml_forecaster.cross_validate

```python
cross_validate(
    df: 'pd.DataFrame',
    cv_split: 'int',
    test_size: 'int',
    metrics: 'List[Callable]',
    step_size: 'int' = 1,
    h_split_point: 'Optional[int]' = None
)
```

Run cross-validation using time series splits.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | DataFrame containing the target and any feature columns. |
| cv_split | int |  | Number of cross-validation splits. |
| test_size | int |  | Number of periods in each test set. |
| metrics | List[Callable] |  | Metric functions (e.g. ``[MAE, RMSE]``) used to evaluate forecast accuracy across folds. Call ``.cv_summary()`` after cross-validation to retrieve the aggregated scores. |
| step_size | int | 1 | Step size to move the test window forward in each split. |
| h_split_point | Optional[int] | None | Optional index to split the test set into two parts for separate evaluation (e.g. to evaluate short-term vs long-term performance). If None, no split is done. |
| **Returns** | **Tuple[pd.DataFrame, pd.DataFrame]** |  | **DataFrame containing overall performance metrics averaged across splits, and a DataFrame with predictions and true values for each split.** |
:::
:::


## GLM (Generalized Linear Model) forecaster

::: {#ec2dde61 .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.glm

```python
glm(
    family: 'Any',
    target_col: 'str',
    lags: 'Optional[Union[int, List[int]]]' = None,
    lag_transform: 'Optional[list]' = None,
    difference: 'Optional[int]' = None,
    seasonal_diff: 'Optional[int]' = None,
    trend: 'Optional[str]' = None,
    pol_degree: 'int' = 1,
    ets_params: 'Optional[Dict[str, Any]]' = None,
    change_points: 'Optional[List[int]]' = None,
    box_cox: 'Union[bool, float, int]' = False,
    box_cox_biasadj: 'bool' = False,
    add_constant: 'bool' = True,
    cat_variables: 'Optional[List[str]]' = None,
    categorical_encoder: 'Optional[Any]' = None,
    offset: 'Optional[np.ndarray]' = None,
    exposure: 'Optional[np.ndarray]' = None,
    freq_weights: 'Optional[np.ndarray]' = None,
    var_weights: 'Optional[np.ndarray]' = None,
    missing: 'Optional[str]' = None
)
```

Initialize the glm forecaster with the specified model and data preparation parameters.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| family | Any |  | A statsmodels family object specifying the error distribution and link function for the GLM (e.g. family=sm.families.Poisson() for count data, family=sm.families.Binomial() for binary data, etc.). `import statsmodels.api as sm`, so you can access the families via `sm.families`. |
| target_col | str |  | Name of the target variable column in the input DataFrame. |
| lags | Optional[Union[int, List[int]]] | None | Lags to include as features. If an integer is provided, lags from 1 to that integer will be included. If a list of integers is provided, those specific lags will be included. Default is None (no lag features). |
| lag_transform | Optional[list] | None | List of lag-transform function objects to apply to the target variable (e.g. [expanding_mean(shift=1), rolling_std(window_size=3, shift=1)]). Each function should take a pandas Series as input and return a Series of the same length. Default is None (no lag transforms). |
| difference | Optional[int] | None | Order of ordinary differencing to apply to the target variable (e.g. 1 for first difference). Default is None (no differencing). |
| seasonal_diff | Optional[int] | None | Seasonal period for seasonal differencing (e.g. 12 for monthly data with yearly seasonality). Default is None (no seasonal differencing). |
| trend | Optional[str] | None | Trend strategy to use. Options are 'linear' for linear trend removal, 'ets' for ETS-based trend removal, 'feature_lr' for using linear trend components as features, and 'feature_ets' for using ETS trend components as features. Default is None (no trend handling). |
| pol_degree | int | 1 | Degree of polynomial trend to fit when using 'linear' or 'feature_lr' trend strategy. Default is 1 (linear trend). |
| ets_params | Optional[Dict[str, Any]] | None | Dictionary of parameters for the ExponentialSmoothing model when using 'ets' trend strategy. The keys should be the parameter names and the values should be the parameter values. Default is None (use default ETS parameters). |
| change_points | Optional[List[int]] | None | List of indices in the time series where change points occur for piecewise linear trend fitting. Only used when trend strategy is 'linear' or 'feature_lr'. Default is None (no change points, fit a single linear trend). |
| box_cox | Union[bool, float, int] | False | Whether to apply Box-Cox transformation to the target variable. If a float or int value is provided, it will be used as the lambda parameter for the Box-Cox transformation. If True, the lambda parameter will be estimated from the data. |
| box_cox_biasadj | bool | False | Whether to apply bias adjustment when inverting the Box-Cox transformation on forecasts. Default is False. |
| add_constant | bool | True |  |
| cat_variables | Optional[List[str]] | None | List of categorical feature column names. If provided, these columns will be treated as categorical variables and encoded accordingly. Default is None (no categorical variables). |
| categorical_encoder | Optional[Any] | None | Categorical encoder object (e.g. OneHotEncoder(), MeanEncoder(), etc.) to apply to the categorical variables specified in cat_variables. The encoder should have fit() and transform() methods that can be applied to the input DataFrame. Default is None (no categorical encoding) and if None, categorical variables can only be used if the model can handle them natively (e.g. LGBM or CatBoost). |
| offset | Optional[np.ndarray] | None | An offset to be included in the model. If provided, must be an array whose length is the number of rows in exog. |
| exposure | Optional[np.ndarray] | None | Log(exposure) will be added to the linear prediction in the model. Exposure is only valid if the log link is used. If provided, it must be an array with the same length as endog. |
| freq_weights | Optional[np.ndarray] | None | 1d array of frequency weights. The default is None. If None is selected or a blank value, then the algorithm will replace with an array of 1’s with length equal to the endog. WARNING: Using weights is not verified yet for all possible options and results, see Notes in statsmodels documentation. |
| var_weights | Optional[np.ndarray] | None | 1d array of variance (analytic) weights. The default is None. If None is selected or a blank value, then the algorithm will replace with an array of 1’s with length equal to the endog. WARNING: Using weights is not verified yet for all possible options and results, see Notes in statsmodels documentation. |
| missing | Optional[str] | None | Available options are ‘none’, ‘drop’, and ‘raise’. If ‘none’, no nan checking is done. If ‘drop’, any observations with nans are dropped. If ‘raise’, an error is raised. Default is ‘none’. |
| **Returns** | **None** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.glm.fit

```python
fit(
    df: 'pd.DataFrame'
)
```

Fit the model to the training data after applying the specified data preparation steps.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| df | pd.DataFrame | Training DataFrame containing the target and any feature columns. |
| **Returns** | **None** |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.glm.forecast

```python
forecast(
    H: 'int',
    exog: 'Optional[pd.DataFrame]' = None
)
```

Recursive multi-step forecast.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| H | int |  | Forecast horizon. |
| exog | Optional[pd.DataFrame] | None | Optional dataframe of future regressors. |
| **Returns** | **np.ndarray** |  | **Forecast values of length `H`.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.glm.cross_validate

```python
cross_validate(
    df: 'pd.DataFrame',
    cv_split: 'int',
    test_size: 'int',
    metrics: 'List[Callable]',
    step_size: 'int' = 1,
    h_split_point: 'Optional[int]' = None
)
```

Run cross-validation using time series splits.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | DataFrame containing the target and any feature columns. |
| cv_split | int |  | Number of cross-validation splits. |
| test_size | int |  | Number of periods in each test set. |
| metrics | List[Callable] |  | Metric functions (e.g. ``[MAE, RMSE]``) used to evaluate forecast accuracy across folds. Call ``.cv_summary()`` after cross-validation to retrieve the aggregated scores. |
| step_size | int | 1 | Step size to move the test window forward in each split. |
| h_split_point | Optional[int] | None | Optional index to split the test set into two parts for separate evaluation (e.g. to evaluate short-term vs long-term performance). If None, no split is done. |
| **Returns** | **Tuple[pd.DataFrame, pd.DataFrame]** |  | **DataFrame containing overall performance metrics averaged across splits, and a DataFrame with predictions and true values for each split.** |
:::
:::


## MS-ARR (Markov Switching Autoregressive Regression) forecaster

::: {#aac044ea .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ms_arr

```python
ms_arr(
    n_components: 'int',
    target_col: 'str',
    lags: 'Optional[Union[int, List[int]]]' = None,
    lag_transform: 'Optional[list]' = None,
    difference: 'Optional[int]' = None,
    seasonal_diff: 'Optional[int]' = None,
    trend: 'Optional[str]' = None,
    pol_degree: 'int' = 1,
    ets_params: 'Optional[Dict[str, Any]]' = None,
    change_points: 'Optional[List[int]]' = None,
    box_cox: 'Union[bool, float, int]' = False,
    box_cox_biasadj: 'bool' = False,
    add_constant: 'bool' = True,
    cat_variables: 'Optional[List[str]]' = None,
    categorical_encoder: 'Optional[Any]' = None,
    method: 'str' = 'posterior',
    switching_var: 'bool' = True,
    startprob_prior: 'float' = 1000.0,
    transmat_prior: 'float' = 1000.0,
    n_iter: 'int' = 100,
    tol: 'float' = 0.001,
    ridge: 'float' = 1e-05,
    coefficients: 'Optional[np.ndarray]' = None,
    stds: 'Optional[np.ndarray]' = None,
    init_state: 'Optional[np.ndarray]' = None,
    trans_matrix: 'Optional[np.ndarray]' = None,
    random_state: 'int' = 42,
    verbose: 'bool' = False
)
```

Initialize the MS-ARR model with the specified parameters.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| n_components | int |  | Number of hidden states (regimes). |
| target_col | str |  | Name of the target variable. |
| lags | Optional[Union[int, List[int]]] | None | Lags for the autoregressive model. |
| lag_transform | Optional[list] | None | List of lag-transform function objects applied to the target. |
| difference | Optional[int] | None | Order of ordinary differencing (e.g. 1 for first difference). |
| seasonal_diff | Optional[int] | None | Seasonal period for seasonal differencing. |
| trend | Optional[str] | None | Trend strategy: 'linear' or 'ets'. |
| pol_degree | int | 1 | Degree of polynomial trend (default: 1). Used when trend='linear'. |
| ets_params | Optional[Dict[str, Any]] | None | Dictionary of parameters for the ExponentialSmoothing model when using 'ets' trend strategy. The keys should be the parameter names and the values should be the parameter values. Default is None (use default ETS parameters). |
| change_points | Optional[List[int]] | None | Change points for piecewise linear trend. List of indices where the trend slope can change. |
| box_cox | Union[bool, float, int] | False | Whether to apply Box-Cox transformation to the target variable. If a float or int value is provided, it will be used as the lambda parameter for the Box-Cox transformation. If True, the lambda parameter will be estimated from the data. |
| box_cox_biasadj | bool | False | Whether to apply bias adjustment when inverting Box-Cox transformation (default: False). |
| add_constant | bool | True | If True, prepend a constant column to the regressor matrix (default: True). |
| cat_variables | Optional[List[str]] | None | List of categorical feature column names. If provided, these columns will be treated as categorical variables and encoded accordingly. Default is None (no categorical variables). |
| categorical_encoder | Optional[Any] | None | Categorical encoder object (e.g. OneHotEncoder(), MeanEncoder(), etc.) to apply to the categorical variables specified in cat_variables. The encoder should have fit() and transform() methods that can be applied to the input DataFrame. Default is None (no categorical encoding) and if None, categorical variables can only be used if the model can handle them natively (e.g. LGBM or CatBoost). |
| method | str | posterior | State assignment method: 'posterior' (soft) or 'viterbi' (hard). Default: 'posterior'. |
| switching_var | bool | True | If True, each regime has its own variance. If False, uses pooled variance. Default: True. |
| startprob_prior | float | 1000.0 | Dirichlet concentration for initial state distribution. Default: 1e3. |
| transmat_prior | float | 1000.0 | Dirichlet concentration for transition matrix rows. Default: 1e3. |
| n_iter | int | 100 | Maximum EM iterations. Default: 100. |
| tol | float | 0.001 | Convergence tolerance on log-likelihood. Default: 1e-6. |
| ridge | float | 1e-05 | Ridge regularisation parameter for coefficient estimation. Default: 1e-5. |
| coefficients | Optional[np.ndarray] | None | Initial regression coefficients (shape: n_states x n_features). |
| stds | Optional[np.ndarray] | None | Initial state standard deviations (shape: n_states,). |
| init_state | Optional[np.ndarray] | None | Initial state probability vector (shape: n_states,). |
| trans_matrix | Optional[np.ndarray] | None | Initial transition matrix (shape: n_states x n_states). |
| random_state | int | 42 | Random seed for reproducibility. Default: 42. |
| verbose | bool | False | If True, print EM progress. Default: False. |
| **Returns** | **None** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ms_arr.fit

```python
fit(
    df: 'pd.DataFrame'
)
```

Fit the model using the EM algorithm.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| df | pd.DataFrame | Training DataFrame containing the target and any feature columns. |
| **Returns** | **float** | **Final log-likelihood after EM convergence.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ms_arr.forecast

```python
forecast(
    H: 'int',
    exog: 'Optional[pd.DataFrame]' = None
)
```

Generate forecasts for H future time steps.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| H | int |  | Forecast horizon (number of steps to forecast ahead). |
| exog | Optional[pd.DataFrame] | None | Future exogenous regressors (must contain at least H rows). Should have the same columns as the training data (excluding the target variable). |
| **Returns** | **np.ndarray** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ms_arr.cross_validate

```python
cross_validate(
    df: 'pd.DataFrame',
    cv_split: 'int',
    test_size: 'int',
    metrics: 'List[Callable]',
    step_size: 'int' = 1,
    n_iter: 'int' = 1,
    h_split_point: 'Optional[int]' = None
)
```

Run cross-validation.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | DataFrame containing the target and any feature columns. |
| cv_split | int |  | Number of cross-validation splits. |
| test_size | int |  | Number of time steps in the test set for each split. |
| metrics | List[Callable] |  | Metric functions (e.g. ``[MAE, RMSE]``) used to evaluate forecast accuracy across folds. Call ``.cv_summary()`` after cross-validation to retrieve the aggregated scores. |
| step_size | int | 1 | Step size between the start of each test set in the splits. |
| n_iter | int | 1 | Number of EM iterations to run for each training fold. |
| h_split_point | Optional[int] | None | If provided, split the test set into two parts at this index and evaluate metrics separately on each part. |
| **Returns** | **Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]** |  | **DataFrame containing the average score for each metric across all splits. If h_split_point is provided, also includes separate scores for the two parts of the test set. If cv_df is True, returns a tuple of (overall_performance_df, cv_predictions_df)** |
:::
:::


## Hybrid (pesh) forecaster

::: {#f2b67a3a .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.pesh

```python
pesh(
    models: 'dict',
    weighting_scheme: 'Optional[Dict[str, float]]' = None
)
```

Initialize the pesh model with the specified parameters for hybrid forecasting that combines forecasts from multiple models.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| models | dict |  | A dictionary of model instances to be used for forecasting. The keys should be string names for each model. |
| weighting_scheme | Optional[Dict[str, float]] | None | Optional dictionary specifying weights for each model's forecast. Default is None, which means equal weighting. |
| **Returns** | **None** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.pesh.fit

```python
fit(
    df: 'pd.DataFrame'
)
```

Fit the specified models to the training data.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| df | pd.DataFrame | Training DataFrame containing the target and any feature columns. |
| **Returns** | **None** |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.pesh.forecast

```python
forecast(
    H: 'int',
    exog: 'Optional[pd.DataFrame]' = None
)
```

Recursive multi-step forecast.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| H | int |  | Forecast horizon. |
| exog | Optional[pd.DataFrame] | None | Optional dataframe of future regressors. Must have the same columns as the exogenous variables used during training and at least `H` rows. |
| **Returns** | **np.ndarray** |  | **Forecast values of length `H`.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.pesh.cross_validate

```python
cross_validate(
    df: 'pd.DataFrame',
    cv_split: 'int',
    test_size: 'int',
    metrics: 'List[Callable]',
    step_size: 'int' = 1,
    metric_to_opt: 'Optional[Callable]' = None,
    weighting_scheme: 'Optional[Union[Dict[str, float], str]]' = None,
    optimizer: 'str' = 'SLSQP'
)
```

Perform cross-validation for the pesh model using a rolling forecasting origin approach.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | The input DataFrame containing the target and any feature columns. |
| cv_split | int |  | The number of cross-validation splits. |
| test_size | int |  | The size of the test set for each split. |
| metrics | List[Callable] |  | Metric functions (e.g. ``[MAE, RMSE]``) used to evaluate forecast accuracy across folds. Call ``.cv_summary()`` after cross-validation to retrieve the aggregated scores. |
| step_size | int | 1 | The step size for rolling the forecasting origin. |
| metric_to_opt | Optional[Callable] | None | An optional metric function to optimize when weighting_scheme is set to "optimize". If None, it defaults to the first metric in the metrics list. |
| weighting_scheme | Optional[Union[Dict[str, float], str]] | None | None: equal weights across models. dict: user-provided weights (must sum to 1). "optimize": optimize weights to minimize MSE via `scipy.optimize.minimize`. |
| optimizer | str | SLSQP | Optimization method to use when weighting_scheme is set to "optimize". Passed to `scipy.optimize.minimize`. Refer to SciPy documentation for available methods. |
| **Returns** | **pd.DataFrame** |  | **A DataFrame containing the performance metrics for each model and the combined forecast across all cross-validation splits. Also, optimized weights are stored in `self.optimal_weights_` if `weighting_scheme` is "optimize".** |
:::
:::


## VAR (Vector Autoregression) forecaster

::: {#fc87346c .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.var

```python
var(
    target_cols: 'List[str]',
    lags: 'Dict[str, Union[int, List[int]]]',
    lag_transform: 'Optional[Dict[str, list]]' = None,
    difference: 'Optional[Dict[str, int]]' = None,
    seasonal_diff: 'Optional[Dict[str, int]]' = None,
    trend: 'Optional[Dict[str, str]]' = None,
    pol_degree: 'Optional[Union[int, Dict[str, int]]]' = 1,
    ets_params: 'Optional[Dict[str, Any]]' = None,
    change_points: 'Optional[Dict[str, List[int]]]' = None,
    box_cox: 'Optional[Dict[str, Union[bool, float, int]]]' = None,
    box_cox_biasadj: 'Union[bool, Dict[str, bool]]' = False,
    add_constant: 'bool' = True,
    cat_variables: 'Optional[List[str]]' = None,
    categorical_encoder: 'Optional[Union[Dict[str, Any], Any]]' = None,
    verbose: 'bool' = False
)
```

"
Initialize the VAR model with specified preprocessing and modeling parameters.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| target_cols | List[str] |  | List of target column names to model. |
| lags | Dict[str, Union[int, List[int]]] |  | Dictionary specifying lags for each target variable. Values can be an int (number of lags) or a list of specific lag indices. |
| lag_transform | Optional[Dict[str, list]] | None | Dictionary specifying lag-transform functions for each target variable. Each value is a list of transformation functions (e.g., rolling_mean, expanding_std) to apply to the lagged features of that target. |
| difference | Optional[Dict[str, int]] | None | Dictionary specifying the order of ordinary differencing to apply to each target variable. Values are integers indicating how many times to difference the series. |
| seasonal_diff | Optional[Dict[str, int]] | None | Dictionary specifying the seasonal period for seasonal differencing for each target variable. Values are integers indicating the seasonal lag (e.g., 12 for monthly data with yearly seasonality). |
| trend | Optional[Dict[str, str]] | None | Dictionary specifying the trend strategy for each target variable. Values can be 'linear' for linear trend removal or 'ets' for ETS-based trend removal. |
| pol_degree | Optional[Union[int, Dict[str, int]]] | 1 | Polynomial degree for linear trend removal. Can be a single integer applied to all targets or a dictionary specifying the degree for each target. |
| ets_params | Optional[Dict[str, Any]] | None | Dictionary specifying ETS model and fit parameters for each target variable when using 'ets' trend removal. Each value is a dictionary of parameters for the ExponentialSmoothing model and fitting process. |
| change_points | Optional[Dict[str, List[int]]] | None | Dictionary specifying change points for piecewise linear trend removal for each target variable. Values are lists of integer indices indicating where the trend should change. Only applicable when trend strategy is 'linear'. |
| box_cox | Optional[Dict[str, Union[bool, float, int]]] | None | Dictionary specifying whether to apply Box-Cox transformation to each target variable. Values can be a boolean (True to apply, False to skip) or a float (lambda parameter for Box-Cox transformation). If True, lambda will be estimated from the data. |
| box_cox_biasadj | Union[bool, Dict[str, bool]] | False | Whether to apply bias adjustment when inverting the Box-Cox transformation on forecasts. Can be a single boolean applied to all targets or a dictionary specifying the bias adjustment for each target. |
| add_constant | bool | True | If True, a constant column will be added to the regressor matrix for the VAR model. This is typically used to allow for an intercept in the model. |
| cat_variables | Optional[List[str]] | None | List of categorical feature column names to encode. These will be shared across all target variables. |
| categorical_encoder | Optional[Union[Dict[str, Any], Any]] | None | A categorical encoder instance, or a single-entry dictionary mapping the target column to the encoder when the encoder requires access to the target variable during fitting (e.g. {target_col: MeanEncoder()}). If encoder requiring target access is provided directly without the dict format, first target column in target_cols will be used for fitting the encoder. For encoders that do not require target access, pass the encoder instance directly (e.g. OneHotEncoder()). |
| verbose | bool | False | If True, the model will print verbose messages. |
| **Returns** | **None** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.var.fit

```python
fit(
    df: 'pd.DataFrame'
)
```

Fit the VAR model to the provided DataFrame.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| df | pd.DataFrame | Training DataFrame containing the target and any feature columns. |
| **Returns** | **None** |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.var.forecast

```python
forecast(
    H: 'int',
    exog: 'Optional[pd.DataFrame]' = None
)
```

Generate forecasts for H future time steps.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| H | int |  | Forecast horizon (number of steps ahead to predict). |
| exog | Optional[pd.DataFrame] | None | Future exogenous regressors (must contain at least H rows). |
| **Returns** | **Dict[str, np.ndarray]** |  | **Forecasted values for each target, keyed by column name.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.var.cross_validate

```python
cross_validate(
    df: 'pd.DataFrame',
    target_col: 'str',
    cv_split: 'int',
    test_size: 'int',
    metrics: 'List[Callable]',
    step_size: 'int' = 1,
    h_split_point: 'Optional[int]' = None
)
```

Perform cross-validation.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | Input dataframe. |
| target_col | str |  | Target variable for evaluation. |
| cv_split | int |  | Number of cross-validation folds. |
| test_size | int |  | Test size per fold. |
| metrics | List[Callable] |  | Metric functions (e.g. ``[MAE, RMSE]``) used to evaluate forecast accuracy across folds. Call ``.cv_summary()`` after cross-validation to retrieve the aggregated scores. |
| step_size | int | 1 | Step size for rolling window. Default is 1. |
| h_split_point | Optional[int] | None | Point to split the test set for separate evaluation. Default is None. |
| **Returns** | **Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]** |  | **DataFrame with averaged cross-validation metric scores.** |
:::
:::


## Multivariate machine learning forecaster

::: {#af844a57 .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ml_mv_forecaster

```python
ml_mv_forecaster(
    model: 'Any',
    target_cols: 'List[str]',
    lags: 'Optional[Dict[str, Union[int, List[int]]]]' = None,
    lag_transform: 'Optional[Dict[str, list]]' = None,
    difference: 'Optional[Dict[str, int]]' = None,
    seasonal_diff: 'Optional[Dict[str, int]]' = None,
    trend: 'Optional[Dict[str, str]]' = None,
    pol_degree: 'Optional[Union[int, Dict[str, int]]]' = 1,
    ets_params: 'Optional[Dict[str, Any]]' = None,
    change_points: 'Optional[Dict[str, List[int]]]' = None,
    box_cox: 'Optional[Dict[str, Union[bool, float, int]]]' = None,
    box_cox_biasadj: 'Optional[Dict[str, bool]]' = None,
    cat_variables: 'Optional[List[str]]' = None,
    categorical_encoder: 'Optional[Union[Dict[str, Any], Any]]' = None
)
```

"
Initialize the multi-target machine learning forecaster with specified transformations and model.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | Any |  | A scikit-learn compatible regression model instance (e.g. LGBMRegressor(), CatBoostRegressor(), LinearRegression(), etc.). |
| target_cols | List[str] |  | List of target variable names to forecast. |
| lags | Optional[Dict[str, Union[int, List[int]]]] | None | Dictionary specifying lag features to create for each target variable. The value can be an integer (number of lags) or a list of specific lag periods. |
| lag_transform | Optional[Dict[str, list]] | None | Dictionary specifying lag-based transformations to apply for each target variable. The value should be a list of transformation functions (e.g. rolling_mean, expanding_std) with their parameters encapsulated in the function instance. |
| difference | Optional[Dict[str, int]] | None | Dictionary specifying the order of ordinary differencing to apply for each target variable. |
| seasonal_diff | Optional[Dict[str, int]] | None | Dictionary specifying the order of seasonal differencing to apply for each target variable. |
| trend | Optional[Dict[str, str]] | None | Dictionary specifying the trend removal strategy for each target variable. Supported values are 'linear', 'ets', 'feature_lr', and 'feature_ets'. |
| pol_degree | Optional[Union[int, Dict[str, int]]] | 1 | Polynomial degree for linear trend removal. Can be a single integer applied to all targets or a dictionary specifying the degree for each target variable. |
| ets_params | Optional[Dict[str, Any]] | None | Dictionary specifying ETS model and fit parameters for each target variable when using 'ets' trend removal. Each value is a dictionary of parameters for the ExponentialSmoothing model and fitting process. |
| change_points | Optional[Dict[str, List[int]]] | None | Dictionary specifying change points for piecewise linear trend removal for each target variable. The value should be a list of integer indices where the trend slope can change. |
| box_cox | Optional[Dict[str, Union[bool, float, int]]] | None | Dictionary specifying whether to apply Box-Cox transformation for each target variable. The value can be a boolean (True to apply with lambda estimated from data, False to skip) or a float (specific lambda value to use). |
| box_cox_biasadj | Optional[Dict[str, bool]] | None | Dictionary specifying whether to apply bias adjustment when inverting Box-Cox transformation for each target variable. |
| cat_variables | Optional[List[str]] | None | List of categorical feature column names to encode. These will be shared across all target variables. |
| categorical_encoder | Optional[Union[Dict[str, Any], Any]] | None | A categorical encoder instance, or a single-entry dictionary mapping the target column to the encoder when the encoder requires access to the target variable during fitting (e.g. {target_col: MeanEncoder()}). If encoder requiring target access is provided directly without the dict format, first target column in target_cols will be used for fitting the encoder. For encoders that do not require target access, pass the encoder instance directly (e.g. OneHotEncoder()). |
| **Returns** | **None** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ml_mv_forecaster.fit

```python
fit(
    df: 'pd.DataFrame'
)
```

Fit the model to the data passed in df
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| df | pd.DataFrame | Training DataFrame containing all target and feature columns. |
| **Returns** | **None** |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ml_mv_forecaster.forecast

```python
forecast(
    H: 'int',
    exog: 'Optional[pd.DataFrame]' = None
)
```

Generate forecasts for H future time steps.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| H | int |  | Forecast horizon (number of steps to forecast ahead). |
| exog | Optional[pd.DataFrame] | None | Future exogenous regressors (must contain at least H rows). |
| **Returns** | **Dict[str, np.ndarray]** |  | **A dictionary where keys are target column names and values are arrays of H forecasted values for each target variable.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.models.ml_mv_forecaster.cross_validate

```python
cross_validate(
    df: 'pd.DataFrame',
    target_col: 'str',
    cv_split: 'int',
    test_size: 'int',
    metrics: 'List[Callable]',
    step_size: 'int' = 1,
    h_split_point: 'Optional[int]' = None
)
```

Perform cross-validation.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | Input dataframe. |
| target_col | str |  | Target variable for evaluation. |
| cv_split | int |  | Number of cross-validation folds. |
| test_size | int |  | Test size per fold. |
| metrics | List[Callable] |  | Metric functions (e.g. ``[MAE, RMSE]``) used to evaluate forecast accuracy across folds. Call ``.cv_summary()`` after cross-validation to retrieve the aggregated scores. |
| step_size | int | 1 | Step size for rolling window. Default is 1. |
| h_split_point | Optional[int] | None | Point to split the test set for separate evaluation. Default is None. |
| **Returns** | **Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]** |  | **DataFrame with overall performance metrics averaged across folds. If h_split_point is provided, also includes separate performance before and after the split point.** |
:::
:::


# Probabilistic forecasting

## Probabilistic forecasting for univariate time series

::: {#e974d6f3 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
from peshbeen.probabilistic_forecasting import prob_forecasts, mv_prob_forecasts
```
:::


::: {#732d9c8a .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.probabilistic_forecasting.prob_forecasts

```python
prob_forecasts(
    model,
    H: 'int',
    n_calibration: 'Union[int, None]' = None,
    step_size: 'int' = 1,
    random_state: 'int' = 42,
    n_iter: 'Union[int, None]' = None,
    verbose: 'bool' = False
)
```

Probabilistic forecasting wrapper for any point-forecasting model.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model |  |  | Any model with ``.target_col``, ``.fit(df)``, and ``.forecast(H, exog)`` attributes. |
| H | int |  | Forecast horizon. |
| n_calibration | Union[int, None] | None | Number of calibration windows for cross-validated residual estimation. If None, in sample residuals are used without cross-validation (Horizon-specific uncalibrated intervals may be too narrow in this case. This is recommended when data size is small as the model may not have enough data to fit well in each calibration fold). |
| step_size | int | 1 | Step size between consecutive calibration windows. |
| random_state | int | 42 | Seed for all internal random-number generators. |
| n_iter | Union[int, None] | None | Number of EM iterations during each calibration window.  Only relevant for Markov-switching Autoregressive model (`ms_arr`). A smaller value than the model's default speeds up calibration at the cost of convergence quality per fold — typically a value of 3–10 is sufficient for calibration windows where the model is already close to the solution. |
| verbose | bool | False | Print progress during calibration. |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.probabilistic_forecasting.prob_forecasts.calibrate

```python
calibrate(
    df: 'pd.DataFrame',
    delta: 'Union[float, List[float]]' = 0.5
)
```

Calibrate the conformal predictor.

Runs rolling-window cross-validation (if not already done) to
collect non-conformity scores, then computes the per-horizon
conformal quantile ``q_hat`` for each requested ``delta`` level.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | Calibration dataset. |
| delta | Union[float, List[float]] | 0.5 | Coverage level(s).  A single float produces one symmetric interval; a list produces one interval per level.  For example,``delta=0.9`` produces a 90 % prediction interval. |
| **Returns** | **'prob_forecasts'** |  | **The fitted object, with ``self.q_hat`` set to the calibrated** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.probabilistic_forecasting.prob_forecasts.sample

```python
sample(
    df: 'pd.DataFrame',
    n_samples: 'int' = 1000,
    method: 'str' = 'empirical',
    future_exog: 'Union[pd.DataFrame, None]' = None
)
```

Draw sample paths from the predictive distribution.

Three methods are available:

* ``"empirical"`` — residuals are resampled with replacement
  independently at each horizon.
* ``"kde"`` — a Gaussian KDE is fitted to each horizon's residuals;
  samples are drawn from the smoothed distribution.
* ``"correlated"`` — a multivariate normal is fitted to the full
  ``H``-dimensional residual vectors, preserving cross-horizon
  correlation.  Samples are drawn jointly.

Results are stored on ``self``:

* ``self.sample_paths`` — ``(n_samples, H)`` array of sampled trajectories centred on the point forecast.
* ``self.point_forecast`` — ``(H,)`` point forecast array.
* ``self.sample_paths_df`` — the same data as a DataFrame with columns ``h_1, …, h_H``.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | Training data.  Residuals are computed via cross-validation if not yet available. |
| n_samples | int | 1000 | Number of sample paths to draw. |
| method | str | empirical | Sampling strategy (see above). |
| future_exog | Union[pd.DataFrame, None] | None | Future exogenous variables passed to ``forecast``. |
| **Returns** | **'prob_forecasts'** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.probabilistic_forecasting.prob_forecasts.sample_quantiles

```python
sample_quantiles(
    quantiles: 'Union[float, List[float]]'
)
```

Compute quantiles from the sample paths generated by ``sample``.

Works identically regardless of which ``method`` was passed to ``sample``.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| quantiles | Union[float, List[float]] | Desired quantile levels (e.g. ``[0.1, 0.5, 0.9]``). |
| **Returns** | **pd.DataFrame** | **Columns: ``point_forecast``, ``q_<level>`` for each level.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.probabilistic_forecasting.prob_forecasts.conformal_quantiles

```python
conformal_quantiles(
    df: 'pd.DataFrame',
    quantiles: 'Union[float, List[float]]',
    future_exog: 'Union[pd.DataFrame, None]' = None
)
```

Generate conformal prediction quantiles.

Requires ``calibrate`` to have been called first.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | Training data for the final model fit. |
| quantiles | Union[float, List[float]] |  | Desired quantile levels (e.g. ``[0.1, 0.5, 0.9]``). |
| future_exog | Union[pd.DataFrame, None] | None | Future exogenous variables. |
| **Returns** | **pd.DataFrame** |  | **Columns: ``point_forecast``, ``q_<level>`` for each level.** |
:::
:::


## Probabilistic forecasting for multivariate time series

::: {#9742497d .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.probabilistic_forecasting.mv_prob_forecasts

```python
mv_prob_forecasts(
    model,
    target_col: 'str',
    H: 'int',
    n_calibration: 'Union[int, None]' = None,
    step_size: 'int' = 1,
    random_state: 'int' = 42,
    n_iter: 'Union[int, None]' = None,
    verbose: 'bool' = False
)
```

Probabilistic forecasting wrapper for any point-forecasting model.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model |  |  | Any model with ``.target_col``, ``.fit(df)``, and ``.forecast(H, exog)`` attributes. |
| target_col | str |  | Name of the target variable column in the input DataFrames. |
| H | int |  | Forecast horizon. |
| n_calibration | Union[int, None] | None | Number of calibration windows for cross-validated residual estimation. If None, in sample residuals are used without cross-validation (Horizon-specific uncalibrated intervals may be too narrow in this case. This is recommended when data size is small as the model may not have enough data to fit well in each calibration fold). |
| step_size | int | 1 | Step size between consecutive calibration windows. |
| random_state | int | 42 | Seed for all internal random-number generators. |
| n_iter | Union[int, None] | None | Number of EM iterations during each calibration window.  Only relevant for Markov-switching Autoregressive model (`ms_arr`). A smaller value than the model's default speeds up calibration at the cost of convergence quality per fold — typically a value of 3–10 is sufficient for calibration windows where the model is already close to the solution. |
| verbose | bool | False | Print progress during calibration. |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.probabilistic_forecasting.mv_prob_forecasts.calibrate

```python
calibrate(
    df: 'pd.DataFrame',
    delta: 'Union[float, List[float]]' = 0.5
)
```

Calibrate the conformal predictor.

Runs rolling-window cross-validation (if not already done) to
collect non-conformity scores, then computes the per-horizon
conformal quantile ``q_hat`` for each requested ``delta`` level.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | Calibration dataset. |
| delta | Union[float, List[float]] | 0.5 | Coverage level(s).  A single float produces one symmetric interval; a list produces one interval per level.  For example,``delta=0.9`` produces a 90 % prediction interval. |
| **Returns** | **'prob_forecasts'** |  | **The fitted object, with ``self.q_hat`` set to the calibrated** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.probabilistic_forecasting.mv_prob_forecasts.sample

```python
sample(
    df: 'pd.DataFrame',
    n_samples: 'int' = 1000,
    method: 'str' = 'empirical',
    future_exog: 'Union[pd.DataFrame, None]' = None
)
```

Draw sample paths from the predictive distribution.

Three methods are available:

* ``"empirical"`` — residuals are resampled with replacement
  independently at each horizon.
* ``"kde"`` — a Gaussian KDE is fitted to each horizon's residuals;
  samples are drawn from the smoothed distribution.
* ``"correlated"`` — a multivariate normal is fitted to the full
  ``H``-dimensional residual vectors, preserving cross-horizon
  correlation.  Samples are drawn jointly.

Results are stored on ``self``:

* ``self.sample_paths`` — ``(n_samples, H)`` array of sampled trajectories centred on the point forecast.
* ``self.point_forecast`` — ``(H,)`` point forecast array.
* ``self.sample_paths_df`` — the same data as a DataFrame with columns ``h_1, …, h_H``.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | Training data.  Residuals are computed via cross-validation if not yet available. |
| n_samples | int | 1000 | Number of sample paths to draw. |
| method | str | empirical | Sampling strategy (see above). |
| future_exog | Union[pd.DataFrame, None] | None | Future exogenous variables passed to ``forecast``. |
| **Returns** | **'prob_forecasts'** |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.probabilistic_forecasting.mv_prob_forecasts.sample_quantiles

```python
sample_quantiles(
    quantiles: 'Union[float, List[float]]'
)
```

Compute quantiles from the sample paths generated by ``sample``.

Works identically regardless of which ``method`` was passed to ``sample``.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Details** |
| -- | -------- | ----------- |
| quantiles | Union[float, List[float]] | Desired quantile levels (e.g. ``[0.1, 0.5, 0.9]``). |
| **Returns** | **pd.DataFrame** | **Columns: ``point_forecast``, ``q_<level>`` for each level.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.probabilistic_forecasting.mv_prob_forecasts.conformal_quantiles

```python
conformal_quantiles(
    df: 'pd.DataFrame',
    quantiles: 'Union[float, List[float]]',
    future_exog: 'Union[pd.DataFrame, None]' = None
)
```

Generate conformal prediction quantiles.

Requires ``calibrate`` to have been called first.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | pd.DataFrame |  | Training data for the final model fit. |
| quantiles | Union[float, List[float]] |  | Desired quantile levels (e.g. ``[0.1, 0.5, 0.9]``). |
| future_exog | Union[pd.DataFrame, None] | None | Future exogenous variables. |
| **Returns** | **pd.DataFrame** |  | **Columns: ``point_forecast``, ``q_<level>`` for each level.** |
:::
:::


# Model selection and evaluation

## Hyperparameters tuning methods for Univariate machine learning models

::: {#9baef8ec .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
from peshbeen.model_selection import (hyperopt_tune, optuna_tune, mv_hyperopt_tune, mv_optuna_tune, forward_feature_selection,
backward_feature_selection, mv_forward_feature_selection, mv_backward_feature_selection, ms_arr_forward_feature_selection,
ms_arr_backward_feature_selection)
```
:::


::: {#819fc9a9 .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.model_selection.hyperopt_tune

```python
hyperopt_tune(
    model: Any,
    df: pandas.core.frame.DataFrame,
    cv_split: int,
    test_size: int,
    eval_metric: Callable,
    param_space: Dict[str, Any],
    step_size: int = None,
    eval_num: int = 100,
    candidate_exog: List[str] = None,
    pareto_bounds: float | Tuple[float, float] = (0.5, 0.999),
    verbose: bool = False
)
```

Tune forecasting model hyperparameters using time series cross-validation and Hyperopt.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | Any |  | Forecasting model with .fit and .forecast methods. |
| df | DataFrame |  | Time series data (datetime index, target column, optional exogenous features). |
| cv_split | int |  | Number of cross-validation splits. |
| test_size | int |  | Number of samples in each test fold. For ml_direct_forecaster, this will be overridden to be the maximum horizon in model.H. |
| eval_metric | Callable |  | Metric function to minimise. |
| param_space | Dict |  | Each value must be a callable that accepts a Hyperopt `trial` and returns a value. |
| step_size | int | None | Step size between CV folds. |
| eval_num | int | 100 | Number of Hyperopt trials. Default 100. |
| candidate_exog | List | None | List of exogenous feature names to consider for feature importance-based selection. If None, no feature selection is performed. |
| pareto_bounds | Union | (0.5, 0.999) | If a float is provided, it is used as a fixed cutoff for cumulative importance (e.g., 0.8 means keep features that explain 80% of variance). If a tuple is provided, it defines the lower and upper bounds for tuning the Pareto cutoff. Default is (0.5, 0.999), meaning the cutoff will be tuned between 50% and 99.9% of cumulative importance. |
| verbose | bool | False | Print score for every trial. Default False. |
| **Returns** | **Tuple** |  | **Best hyperparameters and best lags (if 'lags' is in param_space).** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.model_selection.optuna_tune

```python
optuna_tune(
    model: Any,
    df: pandas.core.frame.DataFrame,
    cv_split: int,
    test_size: int,
    eval_metric: Callable,
    param_space: Dict[str, Any],
    step_size: int = None,
    eval_num: int = 100,
    candidate_exog: List[str] = None,
    pareto_bounds: float | Tuple[float, float] = (0.5, 0.999),
    verbose: bool = False
)
```

Tune forecasting model hyperparameters using time series cross-validation and Optuna.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | Any |  | Forecasting model with .fit and .forecast methods. |
| df | DataFrame |  | Time series data (datetime index, target column, optional exogenous features). |
| cv_split | int |  | Number of cross-validation splits. |
| test_size | int |  | Number of samples in each test fold. For ml_direct_forecaster, this will be overridden to be the maximum horizon in model.H. |
| eval_metric | Callable |  | Metric function to minimise. |
| param_space | Dict |  | Each value must be a callable that accepts an Optuna `trial` and returns a value. |
| step_size | int | None | Step size between CV folds. |
| eval_num | int | 100 | Number of Optuna trials. Default 100. |
| candidate_exog | List | None | The 800+ features to optimize |
| pareto_bounds | Union | (0.5, 0.999) | Global Pareto bounds for feature importance. if float is passed we will use that as a fixed cutoff, if tuple is passed to be tuned |
| verbose | bool | False | Print score for every trial. Default False. |
| **Returns** | **Tuple** |  | **Best hyperparameters and best lags (if 'lags' is in param_space).** |
:::
:::


## Hyperparameters tuning methods for Multivariate machine learning models

::: {#6b233b36 .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.model_selection.mv_hyperopt_tune

```python
mv_hyperopt_tune(
    model: object,
    df: pandas.core.frame.DataFrame,
    target_col: str,
    cv_split: int,
    test_size: int,
    eval_metric: Callable,
    param_space: dict,
    step_size: int = None,
    eval_num=100,
    verbose=False
)
```

Tune forecasting model hyperparameters using time series cross-validation and hyperopt for multivariate models.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | object |  | Forecasting model object with .fit and .forecast methods and relevant attributes. |
| df | DataFrame |  | Time series data with a datetime index and a target column and optionally exogenous features. |
| target_col | str |  | Name of the target column to minimize the evaluation metric on. |
| cv_split | int |  | Number of cross-validation splits. |
| test_size | int |  | Number of samples in each test set. |
| eval_metric | Callable |  | Evaluation metric function. |
| param_space | dict |  | Hyperparameter search space for the forecasting model. |
| step_size | int | None | Step size to move the test window forward in each split. |
| eval_num | int | 100 | Number of hyperparameter combinations to evaluate. Default is 100. |
| verbose | bool | False | Whether to print the evaluation metric for each hyperparameter combination. Default is False. |
| **Returns** | **Tuple** |  | **A tuple containing the best hyperparameters, selected lags, and selected transforms.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.model_selection.mv_optuna_tune

```python
mv_optuna_tune(
    model: object,
    df: pandas.core.frame.DataFrame,
    target_col: str,
    cv_split: int,
    test_size: int,
    eval_metric: Callable,
    param_space: Dict[str, Any],
    step_size: int = None,
    eval_num: int = 100,
    verbose: bool = False
)
```

Tune forecasting model hyperparameters using time series cross-validation and Optuna.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | object |  | Forecasting model with .fit and .forecast methods. |
| df | DataFrame |  | Time series data (datetime index, target column, optional exogenous features). |
| target_col | str |  | Name of the target column to minimize the evaluation metric on. |
| cv_split | int |  | Number of cross-validation splits. |
| test_size | int |  | Number of samples in each test fold. |
| eval_metric | Callable |  | Metric function to minimise. |
| param_space | Dict |  | Each value must be a callable that accepts an Optuna `trial` and returns a value. |
| step_size | int | None | Step size between CV folds. |
| eval_num | int | 100 | Number of Optuna trials. Default 100. |
| verbose | bool | False | Print score for every trial. Default False. |
| **Returns** | **Tuple** |  | **Best hyperparameters and best lags (if 'lags' is in param_space).** |
:::
:::


## Feature selection methods for univariate time series models

::: {#8740ca29 .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.model_selection.forward_feature_selection

```python
forward_feature_selection(
    model: object,
    df: pandas.core.frame.DataFrame,
    cv_split: int,
    H: int,
    step_size: int | None = None,
    metrics: Callable | List[Callable] = None,
    lags_to_consider: int | None = None,
    candidate_features: List[str] | None = None,
    transformations: List | None = None,
    starting_lags: List[int] | None = None,
    starting_transforms: List | None = None,
    best_start_score: List[float] | None = None,
    verbose=False
)
```

Forward stepwise feature selection for `ml_forecaster` models.

At each iteration every remaining candidate (lag, exogenous column, or
lag-transform) is tested individually by adding it to the current best
feature set.  The candidate that produces the largest cross-validation
improvement is permanently added.  The loop continues until no remaining
candidate improves any of the evaluation metrics.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | object |  | A *configured but unfitted* `ml_forecaster` instance.  The function works exclusively on deep copies and never mutates the object passed in. |
| df | DataFrame |  | Full training DataFrame. Must contain the target column and any candidate exogenous columns. |
| cv_split | int |  | Number of time-series cross-validation folds. |
| H | int |  | Forecast horizon (test window size for each fold). |
| step_size | Union | None | Step size between consecutive CV folds.  If `None` (default) the step equals `H`, producing non-overlapping folds — consistent with the default behaviour of `ml_forecaster.cross_validate`. |
| metrics | Union | None | One or more metric functions accepted by `ml_forecaster.cross_validate` (e.g. `[MAE, RMSE]`). Selection is driven by the **first** metric in the list; a candidate is only accepted when it improves **all** metrics simultaneously. |
| lags_to_consider | Union | None | Consider lags `1, 2, ..., lags_to_consider` as candidates.  If `None`, lag selection is skipped. |
| candidate_features | Union | None | Column names in `df` that are exogenous feature candidates.  The function never modifies this list.  If `None`, exogenous feature selection is skipped. |
| transformations | Union | None | Lag-transform objects to test as candidates (e.g. `[rolling_mean(3, 1), expanding_std(1)]`).  The function never modifies this list.  If `None`, transform selection is skipped. |
| starting_lags | Union | None | Lags to include in the initial feature set before the search begins. These are *not* candidates — they are always included.  Must be a list (e.g. `[1]` or `[1, 2, 3]`). |
| starting_transforms | Union | None | Lag-transform objects to include in the initial feature set before the search begins.  Must be a list. |
| best_start_score | Union | None | Initial best scores for each metric. If not provided, the function will compute the baseline score using the model with the starting features (if any) before beginning the search. |
| verbose | bool | False | Print a message each time a candidate is accepted. |
| **Returns** |  |  | **A dictionary with keys `best_lags`, `best_exogs`, and `best_transforms` containing the selected features.** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.model_selection.backward_feature_selection

```python
backward_feature_selection(
    model: object,
    df: pandas.core.frame.DataFrame,
    cv_split: int,
    H: int,
    step_size: int | None = None,
    metrics: Callable | List[Callable] = None,
    lags_to_consider: List[int] | None = None,
    candidate_features: List[str] | None = None,
    transformations: List | None = None,
    verbose=False
)
```

Backward stepwise feature selection for `ml_forecaster` models.

Starts with the full feature set (all provided lags, exogenous columns, and
lag-transforms) and at each iteration tries removing each current feature
individually.  The feature whose removal produces the largest cross-validation
improvement is permanently dropped.  The loop continues until no remaining
feature can be removed without hurting any of the evaluation metrics.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | object |  | A *configured but unfitted* `ml_forecaster` instance.  The function works exclusively on deep copies and never mutates the object passed in. |
| df | DataFrame |  | Full training DataFrame. Must contain the target column and any candidate exogenous columns. |
| cv_split | int |  | Number of time-series cross-validation folds. |
| H | int |  | Forecast horizon (test window size for each fold). |
| step_size | Union | None | Step size between consecutive CV folds.  If `None` (default) the step equals `H`, producing non-overlapping folds. |
| metrics | Union | None | One or more metric functions accepted by `ml_forecaster.cross_validate` (e.g. `[MAE, RMSE]`). Selection is driven by the **first** metric in the list; a feature is only removed when doing so improves **all** metrics simultaneously. |
| lags_to_consider | Union | None | Lags to include in the initial feature set and test for removal (e.g. `[1, 2, 3, 4]`).  If `None`, no lag removal is attempted. |
| candidate_features | Union | None | Column names in `df` that start in the model and are tested for removal.  If `None`, exogenous feature removal is skipped. |
| transformations | Union | None | Lag-transform objects that start in the model and are tested for removal (e.g. `[rolling_mean(3, 1), expanding_std(1)]`).  If `None`, transform removal is skipped. |
| verbose | bool | False | Print a message each time a feature is removed. |
| **Returns** |  |  | **A dictionary with keys `best_lags`, `best_exogs`, and `best_transforms` containing the surviving features after backward selection.** |
:::
:::


## Feature selection methods for multivariate time series models

::: {#8cb95274 .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.model_selection.mv_forward_feature_selection

```python
mv_forward_feature_selection(
    model: object,
    df: pandas.core.frame.DataFrame,
    target_col: str,
    cv_split: int,
    H: int,
    step_size=None,
    metrics=None,
    lags_to_consider=None,
    candidate_features=None,
    transformations=None,
    starting_lags=None,
    starting_transforms=None,
    verbose=False
)
```

Forward stepwise feature selection for `ml_mv_forecaster`.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | object |  | Template model — never mutated. |
| df | DataFrame |  | DataFrame containing the target variable and any candidate features. |
| target_col | str |  | Target variable used to evaluate cross-validation score. |
| cv_split | int |  | Number of time-series cross-validation folds. |
| H | int |  | Forecast horizon / test size per fold. |
| step_size | NoneType | None | Rolling-window step size (defaults to H). |
| metrics | NoneType | None | One or more metric functions (e.g. `[MAE, RMSE]`). Selection is driven by the **first** metric in the list; a candidate is only accepted when it improves **all** metrics simultaneously. |
| lags_to_consider | NoneType | None | ``{col: max_lag}`` — lags 1..max_lag are candidates. |
| candidate_features | NoneType | None | Exogenous columns to consider adding. |
| transformations | NoneType | None | ``{col: [transform_objects]}`` — transform candidates per target. |
| starting_lags | NoneType | None | Lags already included before search begins. |
| starting_transforms | NoneType | None | Transforms already included before search begins. |
| verbose | bool | False |  |
| **Returns** |  |  | **`{"best_lags": {col: [...]}, "best_exogs": [...],
   "best_transforms": {col: [name_str, ...]}}`** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.model_selection.mv_backward_feature_selection

```python
mv_backward_feature_selection(
    model: object,
    df: pandas.core.frame.DataFrame,
    target_col: str,
    cv_split: int,
    H: int,
    step_size=None,
    metrics=None,
    lags_to_consider=None,
    candidate_features=None,
    transformations=None,
    verbose=False
)
```

Backward stepwise feature selection for ``ml_mv_forecaster``.

Starts with all candidate features included and iteratively removes
the one whose removal most improves cross-validation score.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | object |  | Template model — never mutated. |
| df | DataFrame |  | All candidate exog columns must already be present. |
| target_col | str |  | Target variable used to evaluate cross-validation score. |
| cv_split | int |  |  |
| H | int |  | Forecast horizon / test size per fold. |
| step_size | NoneType | None | Rolling-window step size (defaults to H). |
| metrics | NoneType | None | One or more metric functions (e.g. `[MAE, RMSE]`). A feature is only removed when its removal improves **all** metrics simultaneously. |
| lags_to_consider | NoneType | None | ``{col: max_lag}`` — all lags 1..max_lag start as selected. |
| candidate_features | NoneType | None | Exogenous columns that start as selected. |
| transformations | NoneType | None | ``{col: [transform_objects]}`` — all transforms start as selected. |
| verbose | bool | False |  |
| **Returns** |  |  | **``{"best_lags": {col: [...]}, "best_exogs": [...],
   "best_transforms": {col: [name_str, ...]}}``** |
:::
:::


## Feature selection methods for Markov Switching Autoregressive Regression

::: {#d870ca6f .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.model_selection.ms_arr_forward_feature_selection

```python
ms_arr_forward_feature_selection(
    model: object,
    df: pandas.core.frame.DataFrame,
    cv_split: int,
    H: int,
    step_size: int | None = None,
    metrics: Callable | List[Callable] = None,
    lags_to_consider: int | List[int] | None = None,
    candidate_features: List[str] | None = None,
    transformations: List | None = None,
    starting_lags: List[int] | None = None,
    starting_transforms: List | None = None,
    validation_type: str = 'cv',
    iterations: int = 10,
    verbose: bool = False
)
```

Forward stepwise feature selection for `ms_arr` models.

At each iteration every remaining candidate (lag, exogenous column, or
lag-transform) is tested individually by adding it to the current best
feature set.  The candidate that produces the largest improvement is
permanently added.  The loop continues until no remaining candidate
improves the evaluation criterion.

The HMM state (A, pi, stds, coeffs) is warm-started from the round
winner and propagated to subsequent rounds for consistent initialisation.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| model | object |  | A configured `ms_arr` instance with `fit_em()` already called (recommended to use few EM iterations for this initial fit, e.g. `iterations=10`) or a template model with the same configuration but not yet fitted.  The model is copied internally and never mutated, so the caller's instance remains unchanged. |
| df | DataFrame |  | Full training DataFrame. Must contain the target column and any candidate exogenous columns. |
| cv_split | int |  | Number of time-series cross-validation folds. |
| H | int |  | Forecast horizon (test window size for each fold). |
| step_size | Union | None | Step size between consecutive CV folds. Defaults to H. |
| metrics | Union | None | Required when validation_type='cv'. Selection driven by first metric; a candidate is accepted only when it improves all metrics. |
| lags_to_consider | Union | None | Candidate lags. Int → 1..n; list → specific lags. |
| candidate_features | Union | None | Exogenous column names to test as candidates. |
| transformations | Union | None | Lag-transform objects to test as candidates. |
| starting_lags | Union | None | Lags always included in the initial set (not candidates). |
| starting_transforms | Union | None | Transforms always included in the initial set (not candidates). |
| validation_type | str | cv | Criterion for selection: 'cv', 'AIC', 'BIC', or 'AIC_BIC'. When 'cv', metrics must be provided and drive selection. When 'AIC' or 'BIC', the respective information criterion is used. When 'AIC_BIC', a candidate is accepted only if it improves both AIC and BIC. |
| iterations | int | 10 | EM iterations used inside fit_em() for each candidate evaluation. |
| verbose | bool | False | Print a message each time a candidate is accepted. |
| **Returns** |  |  | **`{"best_lags": [...], "best_exogs": [...], "best_transforms": [...]}`** |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.model_selection.ms_arr_backward_feature_selection

```python
ms_arr_backward_feature_selection(
    df: pandas.core.frame.DataFrame,
    cv_split: int,
    H: int,
    step_size: int | None = None,
    model: object = None,
    metrics: Callable | List[Callable] = None,
    lags_to_consider: int | List[int] | None = None,
    candidate_features: List[str] | None = None,
    transformations: List | None = None,
    validation_type: str = 'cv',
    iterations: int = 100,
    verbose: bool = False
)
```

Backward stepwise feature selection for `ms_arr` models.

Starts with the full feature set and at each iteration tries removing each
current feature individually.  The feature whose removal produces the
largest improvement is permanently dropped.  The loop continues until no
removal improves the evaluation criterion.

The HMM state (A, pi, stds, coeffs) is warm-started from the round
winner and propagated to subsequent rounds.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| df | DataFrame |  | Full training DataFrame. All candidate exogenous columns must be present. |
| cv_split | int |  | Number of time-series cross-validation folds. |
| H | int |  | Forecast horizon (test window size for each fold). |
| step_size | Union | None | Step size between consecutive CV folds. Defaults to H. |
| model | object | None | A configured but unfitted ms_arr instance. Never mutated. |
| metrics | Union | None | Required when validation_type='cv'. A feature is only removed when
doing so improves all metrics simultaneously. |
| lags_to_consider | Union | None | Initial lag set. Int → 1..n; list → specific lags. |
| candidate_features | Union | None | Exogenous columns that start in the model and are tested for removal. |
| transformations | Union | None | Lag-transform objects that start in the model and are tested for removal. |
| validation_type | str | cv | Criterion for selection: 'cv', 'AIC', 'BIC', or 'AIC_BIC'. |
| iterations | int | 100 | EM iterations used inside fit_em() for each candidate evaluation. |
| verbose | bool | False | Print a message each time a feature is removed. |
| **Returns** |  |  | **`{"best_lags": [...], "best_exogs": [...], "best_transforms": [...]}`** |
:::
:::


# Transformations

::: {#abf48c7a .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
from peshbeen.transformations import rolling_mean, rolling_std, rolling_quantile, rolling_min, rolling_max, expanding_mean, expanding_std, expanding_quantile, fourier_terms
```
:::


::: {#d387f152 .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.transformations.fourier_terms

```python
fourier_terms(
    index: 'Union[pd.Index, tuple]',
    period: 'Union[int, float]',
    num_terms: 'int',
    frequency: 'Optional[str]' = None,
    t_start: 'Optional[int]' = None
)
```

Generate Fourier terms for a given index or (start, end) tuple.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| index | Union[pd.Index, tuple] |  | Either a pandas Index directly (recommended), or a (start, end) tuple of integers or datetime strings. |
| period | Union[int, float] |  | The period of the seasonality (e.g., 365.25/7 for weekly yearly seasonality). |
| num_terms | int |  | The number of Fourier term pairs (sin + cos) to generate. |
| frequency | Optional[str] | None | Frequency string (e.g., "W-SAT", "D", "M", "W"). Only relevant when index is a (start, end) tuple. |
| t_start | Optional[int] | None | Starting position of t. Only used when index is a (start, end) tuple. Use len(train_index) to ensure continuity between train and test. |
| **Returns** | **pd.DataFrame** |  | **DataFrame of Fourier terms aligned to the provided index.** |
:::
:::


::: {#453b0daf .cell}

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.transformations.rolling_mean

```python
rolling_mean(
    window_size: 'int',
    shift: 'int' = 1,
    min_samples: 'int' = 1
)
```

A class to compute the rolling mean of a time series with specified window size and shift.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| window_size | int |  | The size of the rolling window. |
| shift | int | 1 | The number of periods to shift the data before applying the rolling mean (default is 1). |
| min_samples | int | 1 | The minimum number of observations in the window required to have a value (default is 1). |
| **Returns** |  |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.transformations.rolling_std

```python
rolling_std(
    window_size: 'int',
    shift: 'int' = 1,
    min_samples: 'int' = 1
)
```

A class to compute the rolling standard deviation of a time series with specified window size and shift.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| window_size | int |  | The size of the rolling window. |
| shift | int | 1 | The number of periods to shift the data before applying the rolling standard deviation (default is 1). |
| min_samples | int | 1 | The minimum number of observations in the window required to have a value (default is 1). |
| **Returns** |  |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.transformations.rolling_quantile

```python
rolling_quantile(
    window_size: 'int',
    quantile: 'float',
    shift: 'int' = 1,
    min_samples: 'int' = 1
)
```

A class to compute the rolling quantile of a time series with specified window size, quantile, and shift.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| window_size | int |  | The size of the rolling window. |
| quantile | float |  | The quantile to compute (between 0 and 1). |
| shift | int | 1 | The number of periods to shift the data before applying the rolling quantile (default is 1). |
| min_samples | int | 1 | The minimum number of observations in the window required to have a value (default is 1). |
| **Returns** |  |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.transformations.rolling_min

```python
rolling_min(
    window_size: 'int',
    shift: 'int' = 1,
    min_samples: 'int' = 1
)
```

A class to compute the rolling minimum of a time series with specified window size and shift.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| window_size | int |  | The size of the rolling window. |
| shift | int | 1 | The number of periods to shift the data before applying the rolling minimum (default is 1). |
| min_samples | int | 1 | The minimum number of observations in the window required to have a value (default is 1). |
| **Returns** |  |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.transformations.rolling_max

```python
rolling_max(
    window_size: 'int',
    shift: 'int' = 1,
    min_samples: 'int' = 1
)
```

A class to compute the rolling maximum of a time series with specified window size and shift.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| window_size | int |  | The size of the rolling window. |
| shift | int | 1 | The number of periods to shift the data before applying the rolling maximum (default is 1). |
| min_samples | int | 1 | The minimum number of observations in the window required to have a value (default is 1). |
| **Returns** |  |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.transformations.expanding_mean

```python
expanding_mean(
    shift: 'int' = 1
)
```

A class to compute the expanding mean of a time series with specified shift.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| shift | int | 1 |  |
| **Returns** |  |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.transformations.expanding_std

```python
expanding_std(
    shift: 'int' = 1
)
```

A class to compute the expanding standard deviation of a time series with specified shift.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| shift | int | 1 | The number of periods to shift the data before applying the expanding standard deviation (default is 1). |
| **Returns** |  |  |  |
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
### peshbeen.transformations.expanding_quantile

```python
expanding_quantile(
    shift: 'int' = 1,
    quantile: 'float' = 0.5
)
```

A class to compute the expanding quantile of a time series with specified shift and quantile.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| shift | int | 1 | The number of periods to shift the data before applying the expanding quantile (default is 1). |
| quantile | float | 0.5 | The quantile to compute (between 0 and 1) (default is 0.5). |
| **Returns** |  |  |  |
:::
:::


