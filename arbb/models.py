#!/usr/bin/env python3
"""
ML Forecasting Package
=======================

This module contains various forecasting classes (using CatBoost, LightGBM, XGBoost,
RandomForest, etc.) and utility functions for cross-validation and hyperparameter
tuning for time-series forecasting.
"""
from typing import List, Dict, Optional, Callable, Tuple, Any, Union
from sklearn.base import clone
from tabnanny import verbose
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit, KFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from arbb.utils import (box_cox_transform, back_box_cox_transform, undiff_ts, seasonal_diff,
                        invert_seasonal_diff, kfold_target_encoder, target_encoder_for_test,
                        rolling_quantile, rolling_mean, rolling_std,
                        expanding_mean, expanding_std, expanding_quantile)
from catboost import CatBoostRegressor
from cubist import Cubist
# dot not show warnings
import warnings
warnings.filterwarnings("ignore")


class ml_forecaster:
    """
    ml Forecaster for time series forecasting.

    Args:
        model (class): Machine learning model class (e.g., CatBoostRegressor, LGBMRegressor).
        target_col (str): Name of the target variable.
        cat_variables (list, optional): List of categorical features.
        target_encode (bool, optional): Whether to use target encoding for categorical features. Default is False.
        n_lag (list or int, optional): Lag(s) to include as features.
        difference (int, optional): Order of difference (e.g. 1 for first difference).
        seasonal_length (int, optional): Seasonal period for seasonal differencing.
        trend (bool, optional): Whether to remove trend.
        trend_type (str, optional): Type of trend removal ('linear', 'feature_lr', 'ets', 'feature_ets').
        ets_params (tuple, optional): A tuple (model_params, fit_params) for exponential smoothing. Ex.g. ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}).
        box_cox (bool, optional): Whether to perform a Box–Cox transformation.
        box_cox_lmda (float, optional): The lambda value for Box–Cox.
        box_cox_biasadj (bool, optional): If True, adjust bias after Box–Cox inversion. Default is False.
        lag_transform (dict, optional): Dictionary specifying additional lag transformations.
    """
    def __init__(self, model, target_col, cat_variables=None, target_encode=False, n_lag=None, difference=None, seasonal_diff=None,
                 trend=False, trend_type="linear", ets_params=None, box_cox=False, box_cox_lmda=None,
                 box_cox_biasadj=False, lag_transform=None):
        # Validate that either n_lag or lag_transform is provided
        if n_lag is None and lag_transform is None:
            raise ValueError("You must supply either n_lag or lag_transform parameters")
            
        self.target_col = target_col
        self.cat_variables = cat_variables
        self.target_encode = target_encode  # whether to use target encoding for categorical variables
        self.n_lag = n_lag
        if self.n_lag is not None:
            if isinstance(self.n_lag, int):
                self.n_lag = list(range(1, self.n_lag + 1))  # convert to list if single integer
            elif isinstance(self.n_lag, list):
                if not all(isinstance(l, int) for l in self.n_lag):
                    raise TypeError("n_lag list must contain only integers")
                # No assignment needed here, already a list
            else:
                raise TypeError("n_lag must be an integer or a list of integers")
        self.difference = difference
        self.season_diff = seasonal_diff
        self.trend = trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        self.box_cox = box_cox
        self.lamda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.lag_transform = lag_transform
        
        # Set default tuned parameters and placeholders for fitted attributes
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None
        self.model = model  # the chosen ML model
    
    def data_prep(self, df):
        """
        Prepare the data with lag features, differencing, trend-removal, and Box–Cox transformation.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe ready for model fitting.
        """
        dfc = df.copy()

        # Process categorical variables if provided

        if self.cat_variables is not None:
            if self.target_encode ==True:
                for col in self.cat_variables:
                    encode_col = col+"_target_encoded"
                    dfc[encode_col] = kfold_target_encoder(dfc, col, self.target_col, 36)
                self.df_encode = dfc.copy()
                dfc = dfc.drop(columns = self.cat_variables)
                # If target encoding is not used, convert categories to dummies    
            else:
                if isinstance(self.model, (CatBoostRegressor, LGBMRegressor)):
                    for col in self.cat_variables:
                        dfc[col] = dfc[col].astype('category')

                else:
                    for col, cat in self.cat_var.items():
                        dfc[col] = dfc[col].astype('category')
                        # Set categories for categorical columns
                        dfc[col] = dfc[col].cat.set_categories(cat)
                    dfc = pd.get_dummies(dfc)

                    for i in self.drop_categ:
                        dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)

        
        if self.target_col in dfc.columns:
            # Apply Box–Cox transformation if specified
            if self.box_cox:
                self.is_zero = np.any(np.array(dfc[self.target_col]) < 1) # check for zero or negative values
                trans_data, self.lamda = box_cox_transform(x=dfc[self.target_col],
                                                        shift=self.is_zero,
                                                        box_cox_lmda=self.lamda)
                dfc[self.target_col] = trans_data
            # Detrend the series if specified
            if self.trend:
                self.len = len(dfc)
                if self.trend_type in ["linear", "feature_lr"]:
                    self.lr_model = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_col])
                    if self.trend_type == "linear":
                        dfc[self.target_col] = dfc[self.target_col] - self.lr_model.predict(np.arange(self.len).reshape(-1, 1))
                if self.trend_type in ["ets", "feature_ets"]:
                    self.target_orig = df[self.target_col]  # Store original values for later use during forecasting
                    self.ets_model = ExponentialSmoothing(df[self.target_col], **self.ets_model).fit(**self.ets_fit)
                    if self.trend_type == "ets":
                        dfc[self.target_col] = dfc[self.target_col] - self.ets_model.fittedvalues.values

            # Apply differencing if specified
            if self.difference is not None or self.season_diff is not None:
                self.orig = dfc[self.target_col].tolist()
                if self.difference is not None:
                    dfc[self.target_col] = np.diff(dfc[self.target_col], n=self.difference,
                                                prepend=np.repeat(np.nan, self.difference))
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

            # Create lag features based on n_lag parameter
            if self.n_lag is not None:
                for lag in self.n_lag:
                    dfc[f"{self.target_col}_lag_{lag}"] = dfc[self.target_col].shift(lag)
            # Create additional lag transformations if specified
            if self.lag_transform is not None:
                for func in self.lag_transform:
                    if isinstance(func, (expanding_std, expanding_mean)):
                        dfc[f"{func.__class__.__name__}_shift_{func.shift}"] = func(dfc[self.target_col])
                    elif isinstance(func, expanding_quantile):
                        dfc[f"{func.__class__.__name__}_shift_{func.shift}_q{func.quantile}"] = func(dfc[self.target_col])
                    elif isinstance(func, rolling_quantile):
                        dfc[f"{func.__class__.__name__}_{func.window_size}_shift_{func.shift}_q{func.quantile}"] = func(dfc[self.target_col])
                    else:
                        dfc[f"{func.__class__.__name__}_{func.window_size}_shift_{func.shift}"] = func(dfc[self.target_col])
            if self.trend:
                if self.trend_type == "feature_lr":
                    dfc["trend"] = self.lr_model.predict(np.arange(self.len).reshape(-1, 1))
                if self.trend_type == "feature_ets":
                    dfc["trend"] = self.ets_model.fittedvalues.values
        return dfc.dropna()
        

    def fit(self, df):
        """
        Fit the ml model.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        model_ = self.model

        if isinstance(self.model, (XGBRegressor, RandomForestRegressor, Cubist, HistGradientBoostingRegressor, AdaBoostRegressor, LinearRegression, Ridge, Lasso, ElasticNet)):
            if (self.cat_variables is not None) and (self.target_encode == False):
                # If categorical variables are provided, create a dictionary of categories
                self.cat_var = {c: sorted(df[c].drop_duplicates().tolist()) for c in self.cat_variables}
                # Create a list of the first category for each categorical variable
                self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_df = self.data_prep(df)
        self.X = model_df.drop(columns=[self.target_col])
        self.y = model_df[self.target_col]
        # Fit the model (passing the categorical features if provided)
        if isinstance(self.model, LGBMRegressor):
            self.model_fit = model_.fit(self.X, self.y, categorical_feature=self.cat_variables)
        elif isinstance(self.model, CatBoostRegressor):
            self.model_fit = model_.fit(self.X, self.y, cat_features=self.cat_variables, verbose=True)
        else:
            self.model_fit = model_.fit(self.X, self.y)


    def forecast(self, n_ahead, x_test=None):
        """
        Forecast n_ahead time steps.

        Args:
            n_ahead (int): Number of forecast steps.
            x_test (pd.DataFrame, optional): Exogenous variables for forecasting.

        Returns:
            np.array: Forecasted values.
        """
        if x_test is not None:  # if external regressors are provided
            if self.cat_variables is not None:
                if self.target_encode:
                    for col in self.cat_variables:
                        encode_col = col + "_target_encoded"
                        x_test[encode_col] = target_encoder_for_test(self.df_encode, x_test, col)
                    x_test = x_test.drop(columns=self.cat_variables)
                else:
                    if isinstance(self.model, (XGBRegressor, RandomForestRegressor, Cubist, HistGradientBoostingRegressor, AdaBoostRegressor, LinearRegression, Ridge, Lasso, ElasticNet)):
                        x_test = self.data_prep(x_test)

        lags = self.y.tolist() # to keep the latest values for lag features
        predictions = []
        
        # Compute trend forecasts if needed
        if self.trend:
            if self.trend_type in ["linear", "feature_lr"]:
                trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
            elif self.trend_type in ["ets", "feature_ets"]:
                orig_lags = self.target_orig.tolist()
                # trend_pred = self.ets_model.forecast(n_ahead).values

        for i in range(n_ahead):
            # If external regressors are provided, extract the i-th row
            if x_test is not None:
                x_var = x_test.iloc[i, :].tolist()
            else:
                x_var = []

            # Build lag-based features from the latest forecast–history
            inp_lag = []
            if self.n_lag is not None:
                inp_lag.extend([lags[-lag] for lag in self.n_lag])

            # Similarly compute additional lag transforms if available
            transform_lag = []
            if self.lag_transform is not None:
                transform_lag = []
                series_array = np.array(lags)
                for func in self.lag_transform:
                    transform_lag.append(func(series_array, is_forecast=True).to_numpy()[-1])
                
                
            # If using trend as a feature, add the forecasted trend component
            trend_var = []
            if (self.trend) and (self.trend_type == "feature_lr"):
                trend_var.append(trend_pred[i])
            elif (self.trend) and (self.trend_type in ["feature_ets", "ets"]):
                ets_fit = ExponentialSmoothing(np.array(orig_lags), **self.ets_model).fit(**self.ets_fit)
                ets_forecast = ets_fit.forecast(1)[0]
                if self.trend_type == "feature_ets":
                    trend_var.append(ets_forecast)

            # Concatenate all features for the forecast step
            inp = x_var + inp_lag + transform_lag + trend_var
            # Ensure that the input is a DataFrame with the same columns as the training data
            df_inp = pd.DataFrame(np.array(inp).reshape(1, -1), columns=self.X.columns)
            if isinstance(self.model, (LGBMRegressor, CatBoostRegressor)):
                df_inp = df_inp.astype({col: 'category' if col in self.cat_variables else 'float64' for col in df_inp.columns})
            # Get the forecast via the model
            pred = self.model_fit.predict(df_inp)[0]
            lags.append(pred)  # update lag history

            # If trend as ets is applied, add the trend component (ets_forecast) to the prediction
            if self.trend:
                if self.trend_type == "ets":
                    # ets_fit = ExponentialSmoothing(np.array(orig_lags), **self.ets_model).fit(**self.ets_fit)
                    # ets_forecast = ets_fit.forecast(1)[0]
                    orig_pred = pred + ets_forecast
                    predictions.append(orig_pred)
                    orig_lags.append(orig_pred)
                elif self.trend_type == "feature_ets":
                    orig_lags.append(pred)
                    predictions.append(pred)
                else: # linear or feature_lr
                    predictions.append(pred)
            else:
                predictions.append(pred)

        forecasts = np.array(predictions)
        # Revert seasonal differencing if applied
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)
        # Revert ordinary differencing if applied
        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)
        # Add static linear trend back if required
        if (self.trend) and (self.trend_type == "linear"):
            forecasts = trend_pred + forecasts
        # Ensure forecasts are nonnegative
        forecasts = np.array([max(0, x) for x in forecasts])
        # Finally, invert Box-Cox transform if it was applied
        if self.box_cox:
            forecasts = back_box_cox_transform(y_pred=forecasts, lmda=self.lamda,
                                                shift=self.is_zero,
                                                box_cox_biasadj=self.biasadj)
        return forecasts

class VARModel:
    """
    Vector Autoregressive Model class supporting data preprocessing, fitting, forecasting, and cross-validation.

    Parameters
    ----------
    target_cols : List[str]
        Target columns for the VAR model (dependent variables).
    lags : Dict[str, List[int]]
        Dictionary specifying lags for each target variable. For example, {'target1': [1, 2], 'target2': [1, 2, 3]} or {'target1': 3, 'target2': 5}.
    lag_transform : Optional[Dict[str, List[tuple]]] (default=None)
        Dictionary specifying lag transformations per target (e.g. rolling, quantile), each as a list of tuples:
        (lag, func, window, [quantile]). For example, { 'target1': [(1, rolling_mean, 30), (2, rolling_quantile, 30, 0.5)] }.
    difference : Optional[Dict[str, int]] (default=None)
        Dictionary specifying order of differencing for each variable. For example, {'target1': 1, 'target2': 2}.
    seasonal_diff : Optional[Dict[str, int]] (default=None)
        Dictionary specifying seasonal differencing for each variable. For example, {'target1': 12, 'target2': 7}.
    trend : Optional[Dict[str, bool]] (default=None)
        Dictionary specifying which variables require detrending.
    trend_types : Optional[Dict[str, str]] (default=None)
        Dictionary specifying trend type for each variable: "linear", "ses", "feature_lr", or "feature_ses".
    ets_params : Optional[Dict[str, tuple]] (default=None)
        Dictionary specifying params for ExponentialSmoothing per variable.
        For example, {'target1': ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}), 'target2': ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True})}.
    box_cox : Optional[Dict[str, bool]] (default=None)
        Dictionary specifying which variables require Box-Cox transform.
    box_cox_lmda : Optional[Dict[str, float]] (default=None)
        Dictionary of Box-Cox lambdas for each variable.
    box_cox_biasadj : bool or Dict[str, bool] (default=False)
        Whether to use bias adjustment for Box-Cox for each variable.
    add_constant : bool (default=True)
        If True, add a constant column to exogenous variables.
    cat_variables : Optional[List[str]] (default=None)
        List of categorical columns to one-hot encode.
    verbose : bool (default=False)
        If True, print verbose messages.

    Methods
    -------
    data_prep(df)
        Prepare the data for VAR model.
    fit(df_train)
        Fit the VAR model to training data.
    forecast(H, exog=None)
        Forecast H steps ahead.
    predict(X)
        Predict with model coefficients.
    cv_var(df, target_col, cv_split, test_size, metrics)
        Cross-validate VAR model.

    Notes
    -----
    - Assumes external utility functions exist for Box-Cox, seasonal differencing, etc.
    """

    def __init__(
        self,
        target_cols: List[str],
        lags: Dict[str, List[int]],
        lag_transform: Optional[Dict[str, List[tuple]]] = None,
        difference: Optional[Dict[str, int]] = None,
        seasonal_diff: Optional[Dict[str, int]] = None,
        trend: Optional[Dict[str, bool]] = None,
        trend_types: Optional[Dict[str, str]] = None,
        ets_params: Optional[Dict[str, tuple]] = None,
        box_cox: Optional[Dict[str, bool]] = None,
        box_cox_lmda: Optional[Dict[str, float]] = None,
        box_cox_biasadj: Any = False,
        add_constant: bool = True,
        cat_variables: Optional[List[str]] = None,
        verbose: bool = False
    ):
        self.target_cols = target_cols
        self.lags = lags
        self.lag_transform = lag_transform
        self.diffs = difference
        self.season_diffs = seasonal_diff
        self.trend = trend
        self.trend_types = trend_types
        self.ets_params = ets_params
        self.box_cox = box_cox
        self.lamdas = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.cons = add_constant
        self.cat_variables = cat_variables
        self.verbose = verbose

        # Handle box_cox bias adjustment dict default
        if self.box_cox is not None and not isinstance(self.box_cox, dict):
            raise TypeError("box_cox must be a dictionary of target values")
        if isinstance(self.box_cox, dict) and not isinstance(self.biasadj, dict):
            self.biasadj = {k: False for k in self.box_cox}
        
        # Handle trend default types
        if self.trend is not None and not isinstance(self.trend, dict):
            raise TypeError("trend must be a dictionary of target values")
        if self.trend is not None and self.trend_types is None:
            self.trend_types = {k: "linear" for k in self.trend}

    def data_prep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the data according to the model configuration.
        Applies categorical encoding, Box-Cox, detrending, differencing, seasonal differencing, lags, and lag transforms.
        Drops rows with any NaN after transformation.
        """
        dfc = df.copy()
        # Handle categorical variables
        if self.cat_variables is not None:
            for col, cats in self.cat_var.items():
                dfc[col] = pd.Categorical(dfc[col], categories=cats)
            dfc = pd.get_dummies(dfc, dtype=float)
            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
        
        # Check all target columns exist
        if all(col in dfc.columns for col in self.target_cols):

            # Box-Cox transformation
            if self.box_cox is not None:
                if self.lamdas is None:
                    self.lamdas = {i: None for i in self.box_cox}
                self.is_zeros = {i: None for i in self.lamdas}
                for k, lm in self.lamdas.items():
                    self.is_zeros[k] = (dfc[k] < 1).any()
                    trans_data, self.lamdas[k] = box_cox_transform(
                        x=dfc[k], shift=self.is_zeros[k], box_cox_lmda=lm
                    )
                    if self.box_cox.get(k, False):
                        dfc[k] = trans_data

            # Detrending
            if self.trend is not None:
                self.tr_models = {i: None for i in self.trend_types}
                self.len = len(dfc)
                for k, tr in self.trend_types.items():
                    if tr in ["linear", "feature_lr"]:
                        lr = LinearRegression()
                        lr.fit(np.arange(self.len).reshape(-1, 1), dfc[k])
                        self.tr_models[k] = lr
                        if tr == "linear":
                            dfc[k] = dfc[k] - lr.predict(np.arange(self.len).reshape(-1, 1))
                    elif tr in ["ses", "feature_ses"]:
                        ets = ExponentialSmoothing(dfc[k], **self.ets_params[k][0])
                        fit = ets.fit(**self.ets_params[k][1])
                        self.tr_models[k] = fit
                        if tr == "ses":
                            dfc[k] = dfc[k] - fit.fittedvalues.values

            # Differencing
            if self.diffs is not None:
                self.origs = {i: dfc[i].tolist() for i in self.diffs}
                for x, d in self.diffs.items():
                    dfc[x] = np.diff(dfc[x], n=d, prepend=np.repeat(np.nan, d))

            # Seasonal differencing
            if self.season_diffs is not None:
                self.orig_ds = {i: dfc[i].tolist() for i in self.season_diffs}
                for w, s in self.season_diffs.items():
                    dfc[w] = seasonal_diff(dfc[w], s)

            # Lag features
            if self.lags is not None:
                for a, lags in self.lags.items():
                    lag_used = lags if isinstance(lags, list) else range(1, lags + 1) # Ensure lags is a list, even if a single int
                    for lg in lag_used:
                        dfc[f"{a}_lag_{lg}"] = dfc[a].shift(lg)

            # Lag transforms

            if self.lag_transform is not None:
                for idx, (target, funcs) in enumerate(self.lag_transform.items()):
                    for func in funcs:
                        if isinstance(func, (expanding_std, expanding_mean)):
                            dfc[f"trg{idx}_{func.__class__.__name__}_shift_{func.shift}"] = func(dfc[target])
                        elif isinstance(func, expanding_quantile):
                            dfc[f"trg{idx}_{func.__class__.__name__}_shift_{func.shift}_q{func.quantile}"] = func(dfc[target])
                        elif isinstance(func, rolling_quantile):
                            dfc[f"trg{idx}_{func.__class__.__name__}_{func.window_size}_shift_{func.shift}_q{func.quantile}"] = func(dfc[target])
                        else:
                            dfc[f"trg{idx}_{func.__class__.__name__}_{func.window_size}_shift_{func.shift}"] = func(dfc[target])

        dfc = dfc.dropna()
        return dfc

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Fit the VAR model to the training data.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data.
        """
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df_train[c].drop_duplicates().tolist()) for c in self.cat_variables}
            self.drop_categ = [self.cat_var[c][0] for c in self.cat_variables]

        df = self.data_prep(df_train)
        X = df.drop(columns=self.target_cols)
        if self.cons:
            X = sm.add_constant(X)
        X = X.apply(pd.to_numeric, errors='raise')
        self.X = np.array(X)
        self.y = np.array(df[self.target_cols])
        self.coeffs = np.linalg.lstsq(self.X, self.y, rcond=None)[0]

    def predict(self, X: List[float]) -> np.ndarray:
        """
        Predict the model output for input X.

        Parameters
        ----------
        X : pd.DataFrame or List[float] or np.ndarray
            Feature DataFrame for prediction.

        Returns
        -------
        np.ndarray
            Model predictions for each target.
        """
        arr = np.array(X)
        return np.dot(self.coeffs.T, arr.T)

    def forecast(self, H: int, exog: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Forecast H steps ahead.

        Parameters
        ----------
        H : int
            Number of steps to forecast.
        exog : Optional[pd.DataFrame]
            Exogenous variables for forecasting.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of forecasts for each target variable.
        """
        y_lists = {j: self.y[:, i].tolist() for i, j in enumerate(self.target_cols)} # Initialize lists for each target variable
        if exog is not None:
            if self.cons:
                if exog.shape[0] == 1:
                    exog.insert(0, 'const', 1)
                else:
                    exog = sm.add_constant(exog)
            exog = np.array(self.data_prep(exog))

        forecasts = {i: [] for i in self.target_cols}

        # Trend forecasting
        if self.trend is not None:
            trend_preds = {i: [] for i in self.trend_types}
            for k, tr in self.trend_types.items():
                if tr in ["linear", "feature_lr"]:
                    trend_preds[k] = self.tr_models[k].predict(np.arange(self.len, self.len+H).reshape(-1, 1))
                elif tr in ["ses", "feature_ses"]:
                    trend_preds[k] = self.tr_models[k].forecast(H).values

        for t in range(H):
            # Exogenous input for step t
            if exog is not None:
                exo_inp = exog[t].tolist()
            else:
                exo_inp = [1] if self.cons else []

            # Lagged features
            lags = []
            if self.lags is not None:
                for tr, vals in y_lists.items():
                    if tr in self.lags:
                        lag_used = self.lags[tr] if isinstance(self.lags[tr], list) else range(1, self.lags[tr] + 1)
                        ys = [vals[-x] for x in lag_used]
                        lags += ys
            # Lag transforms
            transform_lag = []
            if self.lag_transform is not None:
                for target, funcs in self.lag_transform.items():
                    series_array = np.array(y_lists[target])
                    for func in funcs:
                        transform_lag.append(func(series_array, is_forecast=True).to_numpy()[-1])

            # Trend feature
            trend_var = []
            if self.trend is not None:
                for k, tr in self.trend_types.items():
                    if tr in ["feature_lr", "feature_ses"]:
                        trend_var.append(trend_preds[k][t])

            inp = exo_inp + lags + transform_lag + trend_var
            pred = self.predict(inp)
            for id_, ff in enumerate(forecasts):
                forecasts[ff].append(pred[id_])
                y_lists[ff].append(pred[id_])

        # Invert seasonal difference
        if self.season_diffs is not None:
            for s in self.orig_ds:
                forecasts[s] = invert_seasonal_diff(self.orig_ds[s], np.array(forecasts[s]), self.season_diffs[s])

        # Invert difference
        if self.diffs is not None:
            for d in self.diffs:
                forecasts[d] = undiff_ts(self.origs[d], np.array(forecasts[d]), self.diffs[d])

        # Add back trend
        if self.trend is not None:
            for k, tr in self.trend_types.items():
                if tr in ["linear", "ses"]:
                    forecasts[k] = trend_preds[k] + forecasts[k]

        # Non-negativity
        for f in forecasts:
            forecasts[f] = np.array([max(0, x) for x in forecasts[f]])

        # Invert Box-Cox
        if self.box_cox is not None:
            for k, lmd in self.lamdas.items():
                if self.box_cox.get(k, False):
                    forecasts[k] = back_box_cox_transform(
                        y_pred=forecasts[k], lmda=lmd, shift=self.is_zeros[k], box_cox_biasadj=self.biasadj[k]
                    )

        return forecasts

    def cv_var(
        self,
        df: pd.DataFrame,
        target_col: str,
        cv_split: int,
        test_size: int,
        metrics: List[Callable]
    ) -> pd.DataFrame:
        """
        Perform cross-validation for VAR model.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        target_col : str
            Target variable for evaluation.
        cv_split : int
            Number of cross-validation folds.
        test_size : int
            Test size per fold.
        metrics : List[Callable]
            List of metric functions.

        Returns
        -------
        pd.DataFrame
            DataFrame with averaged cross-validation metric scores.
        """
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_forecasts_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns=self.target_cols), np.array(test[target_col])
            self.fit(train)
            bb_forecast = self.forecast(H=test_size, exog=x_test)[target_col]

            forecast_df = test[target_col].to_frame()
            forecast_df["forecasts"] = bb_forecast
            self.cv_forecasts_df = pd.concat([self.cv_forecasts_df, forecast_df], axis=0)

            for m in metrics:
                if m.__name__ == 'mean_squared_error':
                    eval_score = m(y_test, bb_forecast, squared=False)
                elif m.__name__ in ['MeanAbsoluteScaledError', 'MedianAbsoluteScaledError']:
                    eval_score = m(y_test, bb_forecast, np.array(train[self.target_cols]))
                else:
                    eval_score = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval_score)

        overall_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]
        return pd.DataFrame(overall_perform, columns=["eval_metric", "score"])
    

class ml_bidirect_forecaster:
    """
    Bidirectional ml Forecaster for time-series forecasting.

    Args:
         target_cols (list): Names of the target variables.
         cat_variables (list, optional): List of categorical variable names.
         n_lag (dict, optional): Dictionary specifying the number of lags or list of lags for each target variable. Default is None. Example: {'target1': 3, 'target2': [1, 2, 3]}.
         difference (dict, optional): Dictionary specifying the order of ordinary differencing for each target variable. Default is None. Example: {'target1': 1, 'target2': 2}.
         seasonal_length (dict, optional): Seasonal differencing period. Example: {'target1': 7, 'target2': 7}.
         trend (dict, optional): Flag indicating if trend handling is applied. Example: {'Target1': True, 'Target2': False}. Default is None. If None, no trend handling is applied.
         trend_type (dict, optional): Trend handling strategy; one of 'linear' or 'ets'. Default is None. If trend is applied, default is 'linear'. Example: {'Target1': 'linear', 'Target2': 'ets'}.
         ets_params (dict, optional): Dictionary of ETS model parameters (values are lists of dictionaries of params) and fit settings for each target variable. Example: {'Target1': [{'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}], 'Target2': [{'trend': 'mul', 'seasonal': 'mul'}, {'damped_trend': False}]}.
         target_encode (dict, optional): Flag determining if target encoding is used for categorical features for each target variable. Default is False. Example: {'Target1': True, 'Target2': False}.
         box_cox (dict, optional): Whether to apply a Box–Cox transformation for each target variable. Default is False. Example: {'Target1': True, 'Target2': False}.
         box_cox_lmda (dict, optional): Lambda parameter for the Box–Cox transformation for each target variable. Example: {'Target1': 0.5, 'Target2': 0.5}.
         box_cox_biasadj (dict, optional): Whether to adjust bias when inverting the Box–Cox transform for each target variable. Default is False. Example: {'Target1': True, 'Target2': False}.
         lag_transform (dict, optional): Dictionary specifying additional lag transformation functions for each target variable. List of functions to apply for each target variable. Example: {'Target1': [func1, func2], 'Target2': [func3]}.
    """
    def __init__(self, model, target_cols, cat_variables=None, n_lag=None, difference=None, seasonal_length=None,
                 trend=None, trend_type=None, ets_params=None, target_encode=False,
                 box_cox=None, box_cox_lmda=None, box_cox_biasadj=None, lag_transform=None):
        if n_lag is None and lag_transform is None:
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = model
        self.target_cols = target_cols
        self.cat_variables = cat_variables
        self.n_lag = n_lag
        if self.n_lag is not None:
            if not isinstance(self.n_lag, dict):
                raise TypeError("n_lag must be a dictionary of target values")
            for col, lags in self.n_lag.items():
                if isinstance(lags, int):
                    self.n_lag[col] = list(range(1, lags + 1))
                elif isinstance(lags, list):
                    self.n_lag[col] = lags
                else:
                    raise TypeError("n_lag values must be int or list of ints")

        if lag_transform is not None:
            if not isinstance(lag_transform, dict):
                raise TypeError("lag_transform must be a dictionary of target values")
            for col in lag_transform.keys():
                if not isinstance(lag_transform[col], list):
                    raise TypeError("lag_transform values must be a list of functions")
            self.lag_transform = lag_transform
        else:
            self.lag_transform = lag_transform
        
        # if difference is None, set it to None for all target columns
        if difference is None:
            self.difference = {col: None for col in target_cols}
        else:
            if not isinstance(difference, dict):
                raise TypeError("difference must be a dictionary of target values")
            self.difference = difference
            for col in target_cols:
                if col not in self.difference:
                    self.difference[col] = None
        # if seasonal_length is None, set it to None for all target columns
        # if seasonal_length is a dictionary, it must contain all target columns
        if seasonal_length is None:
            self.season_diff = {col: None for col in target_cols}
        else:
            if not isinstance(seasonal_length, dict):
                raise TypeError("seasonal_length must be a dictionary of target values")
            self.season_diff = seasonal_length
            # if any target column is not in the seasonal_length, set it to None
            for col in target_cols:
                if col not in self.season_diff:
                    self.season_diff[col] = None
        # if trend is None, set it to False for all target columns
        # if trend is a dictionary, it must contain all target columns
        if trend is None:
            self.trend = {col: False for col in target_cols}
        else:
            if not isinstance(trend, dict):
                raise TypeError("trend must be a dictionary of target values")
            self.trend = trend
            # if any target column is not in the trend, set it to False
            for col in target_cols:
                if col not in self.trend:
                    self.trend[col] = False
        if trend_type is None:
            self.trend_type = {col: "linear" for col in target_cols}
        else:
            if not isinstance(trend_type, dict):
                raise TypeError("trend_type must be a dictionary of target values")
            self.trend_type = trend_type
            # if any target column is not in the trend_type, set it to "linear"
            for col in target_cols:
                if col not in self.trend_type:
                    self.trend_type[col] = "linear"
        # if ets_params is not None:
        #     self.ets_model1 = ets_params[target_cols[0]][0]
        #     self.ets_fit1 = ets_params[target_cols[0]][1]
        #     self.ets_model2 = ets_params[target_cols[1]][0]
        #     self.ets_fit2 = ets_params[target_cols[1]][1]
        self.ets_params = ets_params
        self.target_encode = target_encode
        if box_cox is None:
            self.box_cox = {col: False for col in target_cols}
        else:
            if not isinstance(box_cox, dict):
                raise TypeError("box_cox must be a dictionary of target values")
            self.box_cox = box_cox

        if box_cox_lmda is None:
            self.lamda = {col: None for col in target_cols}
        else:
            if not isinstance(box_cox_lmda, dict):
                raise TypeError("box_cox_lmda must be a dictionary of target values")
            self.lamda = box_cox_lmda
            # if any target column is not in the box_cox_lmda, set it to None
            for col in target_cols:
                if col not in self.lamda:
                    self.lamda[col] = None
        
        if box_cox_biasadj is None:
            self.biasadj = {col: False for col in target_cols}
        else:
            if not isinstance(box_cox_biasadj, dict):
                raise TypeError("box_cox_biasadj must be a dictionary of target values")
            self.biasadj = box_cox_biasadj
            # if any target column is not in the box_cox_biasadj, set it to False
            for col in target_cols:
                if col not in self.biasadj:
                    self.biasadj[col] = False
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None


    def data_prep(self, df):
        """
        Prepare the data and handle categorical encoding, lag generation, trend removal, and differencing.
        """
        dfc = df.copy()
        if isinstance(self.model, (LGBMRegressor, CatBoostRegressor)):
                        # Process categorical variables if provided
            if self.cat_variables is not None:
                for col in self.cat_variables:
                    dfc[col] = dfc[col].astype('category')
        else:
            # Handle categorical variables
            if self.cat_variables is not None:
                for col, cats in self.cat_var.items():
                    dfc[col] = pd.Categorical(dfc[col], categories=cats)
                dfc = pd.get_dummies(dfc, dtype=float)
                for i in self.drop_categ:
                    dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)

        if all(col in dfc.columns for col in self.target_cols):
        # Box-Cox transformation if flag is set
            if self.box_cox[self.target_cols[0]]:
                self.is_zero1 = np.any(np.array(dfc[self.target_cols[0]]) < 1) # Check if any values are less than 1 for Box-Cox
                trans_data1, self.lamda1 = box_cox_transform(x=dfc[self.target_cols[0]],
                                                            shift=self.is_zero1,
                                                            box_cox_lmda=self.lamda[self.target_cols[0]])
                dfc[self.target_cols[0]] = trans_data1
            if self.box_cox[self.target_cols[1]]:
                self.is_zero2 = np.any(np.array(dfc[self.target_cols[1]]) < 1)
                trans_data2, self.lamda2 = box_cox_transform(x=dfc[self.target_cols[1]],
                                                            shift=self.is_zero2,
                                                            box_cox_lmda=self.lamda[self.target_cols[1]])
                dfc[self.target_cols[1]] = trans_data2

            # Handle trend removal if specified
            # if atleast one target column has a trend, we need to apply the trend removal
            if self.trend[self.target_cols[0]] or self.trend[self.target_cols[1]]:
                self.len = len(dfc)
                if self.trend[self.target_cols[0]]:
                    if self.trend_type[self.target_cols[0]] == "linear":
                        self.lr_model1 = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_cols[0]])
                        dfc[self.target_cols[0]] = dfc[self.target_cols[0]] - self.lr_model1.predict(np.arange(self.len).reshape(-1, 1))
                    if self.trend_type[self.target_cols[0]] == "ets":
                        self.orig_target1 = df[self.target_cols[0]] # Store original values for later use during forecasting
                        self.ses_model1 = ExponentialSmoothing(self.orig_target1, **self.ets_params[self.target_cols[0]][0]).fit(**self.ets_params[self.target_cols[0]][1])
                        dfc[self.target_cols[0]] = dfc[self.target_cols[0]] - self.ses_model1.fittedvalues.values
                # If the second target column has a trend, apply the same logic
                if self.trend[self.target_cols[1]]:
                    if self.trend_type[self.target_cols[1]] == "linear":
                        self.lr_model2 = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_cols[1]])
                        dfc[self.target_cols[1]] = dfc[self.target_cols[1]] - self.lr_model2.predict(np.arange(self.len).reshape(-1, 1))

                    if self.trend_type[self.target_cols[1]] == "ets":
                        self.orig_target2 = df[self.target_cols[1]]
                        self.ses_model2 = ExponentialSmoothing(self.orig_target2, **self.ets_params[self.target_cols[1]][0]).fit(**self.ets_params[self.target_cols[1]][1])
                        dfc[self.target_cols[1]] = dfc[self.target_cols[1]] - self.ses_model2.fittedvalues.values
            # Handle differencing if specified
            if self.difference[self.target_cols[0]] is not None:
                    self.orig1 = df[self.target_cols[0]].tolist()
                    dfc[self.target_cols[0]] = np.diff(dfc[self.target_cols[0]], n=self.difference[self.target_cols[0]],
                                                    prepend=np.repeat(np.nan, self.difference[self.target_cols[0]]))
            if self.difference[self.target_cols[1]] is not None:
                self.orig2 = df[self.target_cols[1]].tolist()
                dfc[self.target_cols[1]] = np.diff(dfc[self.target_cols[1]], n=self.difference[self.target_cols[1]],
                                                prepend=np.repeat(np.nan, self.difference[self.target_cols[1]]))
            if self.season_diff[self.target_cols[0]] is not None:
                self.orig_d1 = dfc[self.target_cols[0]].tolist()
                dfc[self.target_cols[0]] = seasonal_diff(dfc[self.target_cols[0]], self.season_diff[self.target_cols[0]])
            if self.season_diff[self.target_cols[1]] is not None:
                self.orig_d2 = dfc[self.target_cols[1]].tolist()
                dfc[self.target_cols[1]] = seasonal_diff(dfc[self.target_cols[1]], self.season_diff[self.target_cols[1]])

            # Create lag features based on n_lag parameter
            if self.n_lag is not None:
                for target, lags in self.n_lag.items():
                    for lag in lags:
                        dfc[f"{target}_lag_{lag}"] = dfc[target].shift(lag)

            # Create additional lag transformations if specified (check this later)
            if self.lag_transform is not None:
                for idx, (target, funcs) in enumerate(self.lag_transform.items()):
                    for func in funcs:
                        if isinstance(func, (expanding_std, expanding_mean)):
                            dfc[f"trg{idx}_{func.__class__.__name__}_shift_{func.shift}"] = func(dfc[target])
                        elif isinstance(func, expanding_quantile):
                            dfc[f"trg{idx}_{func.__class__.__name__}_shift_{func.shift}_q{func.quantile}"] = func(dfc[target])
                        elif isinstance(func, rolling_quantile):
                            dfc[f"trg{idx}_{func.__class__.__name__}_{func.window_size}_shift_{func.shift}_q{func.quantile}"] = func(dfc[target])
                        else:
                            dfc[f"trg{idx}_{func.__class__.__name__}_{func.window_size}_shift_{func.shift}"] = func(dfc[target])
            # Add trend features if specified
            # if self.trend[self.target_cols[0]]:
            #     # if self.target_cols[0] in dfc.columns:
            #     if self.trend_type[self.target_cols[0]] == "feature_lr":
            #         dfc["trend1"] = self.lr_model1.predict(np.arange(self.len).reshape(-1, 1))
            #     if self.trend_type[self.target_cols[0]] == "feature_ses":
            #         dfc["trend1"] = self.ses_model1.fittedvalues.values
            #     # if self.target_cols[1] in dfc.columns:
            # if self.trend[self.target_cols[1]]:
            #     if self.trend_type[self.target_cols[1]] == "feature_lr":
            #         dfc["trend2"] = self.lr_model2.predict(np.arange(self.len).reshape(-1, 1))
            #     if self.trend_type[self.target_cols[1]] == "feature_ses":
            #         dfc["trend2"] = self.ses_model2.fittedvalues.values

        return dfc.dropna()
            
    def fit(self, df):
        # Fit the model to the dataframe
        model1_ = clone(self.model)
        model2_ = clone(self.model)
        if isinstance(self.model, (XGBRegressor, RandomForestRegressor, Cubist, HistGradientBoostingRegressor, AdaBoostRegressor, LinearRegression, Ridge, Lasso, ElasticNet)):
            if self.cat_variables is not None:
                self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
                self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_train = self.data_prep(df)
        self.X = model_train.drop(columns=self.target_cols)
        self.y1 = model_train[self.target_cols[0]]
        self.y2 = model_train[self.target_cols[1]]
        if isinstance(self.model, CatBoostRegressor):
            self.model1_fit = model1_.fit(self.X, self.y1, cat_features=self.cat_variables, verbose = True)
            self.model2_fit = model2_.fit(self.X, self.y2, cat_features=self.cat_variables, verbose = True)
        elif isinstance(self.model, LGBMRegressor):
            self.model1_fit = model1_.fit(self.X, self.y1, categorical_feature=self.cat_variables)
            self.model2_fit = model2_.fit(self.X, self.y2, categorical_feature=self.cat_variables)
        else:
            self.model1_fit = model1_.fit(self.X, self.y1)
            self.model2_fit = model2_.fit(self.X, self.y2)

        
    def forecast(self, n_ahead, x_test=None):
        """
        Forecast future values for n_ahead periods.

        Args:
            n_ahead (int): Number of periods to forecast.
            x_test (pd.DataFrame, optional): Exogenous variables.

        Returns:
            np.array: Forecasted values.
        """
        if isinstance(self.model, (XGBRegressor, RandomForestRegressor, Cubist, HistGradientBoostingRegressor, AdaBoostRegressor, LinearRegression, Ridge, Lasso, ElasticNet)):
            if x_test is not None:
                x_test = self.data_prep(x_test)

        target1_lags = self.y1.tolist()
        target2_lags = self.y2.tolist()
        tar1_forecasts = []
        tar2_forecasts = []

        if self.trend[self.target_cols[0]]:
            if self.trend_type[self.target_cols[0]] == "linear":
                trend_pred1 = self.lr_model1.predict(np.arange(self.len, self.len + n_ahead).reshape(-1, 1))
            elif self.trend_type[self.target_cols[0]] == "ets":
                orig_target1 = self.orig_target1.tolist()
                # trend_pred1 = self.ses_model1.forecast(n_ahead).values
        if self.trend[self.target_cols[1]]:
            if self.trend_type[self.target_cols[1]] == "linear":
                trend_pred2 = self.lr_model2.predict(np.arange(self.len, self.len + n_ahead).reshape(-1, 1))
            elif self.trend_type[self.target_cols[1]] == "ets":
                orig_target2 = self.orig_target2.tolist()
                # trend_pred2 = self.ses_model2.forecast(n_ahead).values

        # Forecast recursively one step at a time
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_test.iloc[i, :].tolist()
            else:
                x_var = []
                
            inp_lag = []
            if self.n_lag is not None:
                # For the first target variable
                for col, lags in self.n_lag.items():
                    if col == self.target_cols[0]:
                        inp_lag.extend([target1_lags[-lag] for lag in lags])
                    else:
                        inp_lag.extend([target2_lags[-lag] for lag in lags])

            transform_lag = []
            if self.lag_transform is not None:
                for target, funcs in self.lag_transform.items():
                    series_array = np.array(target1_lags if target == self.target_cols[0] else target2_lags)
                    for func in funcs:
                        transform_lag.append(func(series_array, is_forecast=True).to_numpy()[-1])


            inp = x_var + inp_lag + transform_lag
            # print(f"len of x_var: {len(x_var)}, len of inp_lag1: {len(inp_lag1)}, len of inp_lag2: {len(inp_lag2)}, len of transform_lag: {len(transform_lag)}, len of inp: {len(inp)}")

            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns
            if isinstance(self.model, (LGBMRegressor, CatBoostRegressor)):
                df_inp = df_inp.astype({col: 'category' if col in self.cat_variables else 'float64' for col in df_inp.columns})
            pred1 = self.model1_fit.predict(df_inp)[0]
            target1_lags.append(pred1)
            pred2 = self.model2_fit.predict(df_inp)[0]
            target2_lags.append(pred2)
            if (self.trend[self.target_cols[0]]) and (self.trend_type[self.target_cols[0]] == "ets"):
                    ses_fit1 = ExponentialSmoothing(np.array(orig_target1), **self.ets_params[self.target_cols[0]][0]).fit(**self.ets_params[self.target_cols[0]][1])
                    ses_forecast1 = ses_fit1.forecast(1)[0]
                    orig_pred1 = pred1 + ses_forecast1
                    tar1_forecasts.append(orig_pred1)
                    orig_target1.append(orig_pred1)
            else:
                tar1_forecasts.append(pred1)

            if (self.trend[self.target_cols[1]]) and (self.trend_type[self.target_cols[1]] == "ets"):
                ses_fit2 = ExponentialSmoothing(np.array(orig_target2), **self.ets_params[self.target_cols[1]][0]).fit(**self.ets_params[self.target_cols[1]][1])
                ses_forecast2 = ses_fit2.forecast(1)[0]
                orig_pred2 = pred2 + ses_forecast2
                tar2_forecasts.append(orig_pred2)
                orig_target2.append(orig_pred2)
            else:
                tar2_forecasts.append(pred2)
                # print(f'new_pred1: {new_pred1}, ses_forecast1: {ses_forecast1}, pred1: {pred1}, new_pred2: {new_pred2}, ses_forecast2: {ses_forecast2}, pred2: {pred2}')

        forecasts1 = np.array(tar1_forecasts)
        forecasts2 = np.array(tar2_forecasts)
        # Revert seasonal differencing if applied
        if self.season_diff[self.target_cols[0]] is not None:
            forecasts1 = invert_seasonal_diff(self.orig_d1, forecasts1, self.season_diff[self.target_cols[0]])
        if self.season_diff[self.target_cols[1]] is not None:
            forecasts2 = invert_seasonal_diff(self.orig_d2, forecasts2, self.season_diff[self.target_cols[1]])
            
        if self.difference[self.target_cols[0]] is not None:
            forecasts1 = undiff_ts(self.orig1, forecasts1, self.difference[self.target_cols[0]])
        if self.difference[self.target_cols[1]] is not None:
            forecasts2 = undiff_ts(self.orig2, forecasts2, self.difference[self.target_cols[1]])
        if (self.trend[self.target_cols[0]]) and (self.trend_type[self.target_cols[0]] == "linear"):
            forecasts1 = trend_pred1 + forecasts1
        if (self.trend[self.target_cols[1]]) and (self.trend_type[self.target_cols[1]] == "linear"):
            forecasts2 = trend_pred2 + forecasts2
        forecasts1 = np.array([max(0, x) for x in forecasts1])
        forecasts2 = np.array([max(0, x) for x in forecasts2])
        if self.box_cox[self.target_cols[0]]:
            forecasts1 = back_box_cox_transform(y_pred=forecasts1,
                                                lmda=self.lamda1,
                                                shift=self.is_zero1,
                                                box_cox_biasadj=self.biasadj[self.target_cols[0]])
        if self.box_cox[self.target_cols[1]]:
            forecasts2 = back_box_cox_transform(y_pred=forecasts2,
                                                lmda=self.lamda2,
                                                shift=self.is_zero2,
                                                box_cox_biasadj=self.biasadj[self.target_cols[1]])
        forecasts = {
            self.target_cols[0]: forecasts1,
            self.target_cols[1]: forecasts2
        }
        return forecasts
    