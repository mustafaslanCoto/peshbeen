
::: {#ml_multi_forecaster_main_cell .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
from __future__ import annotations
from typing import List, Dict, Optional, Callable, Tuple, Any, Union
import numpy as np
import pandas as pd
import copy
from sklearn.base import clone
from peshbeen.model_selection import SplitTimeSeries
from peshbeen.statstools import lr_trend_model, forecast_trend
from peshbeen.transformations import (
    box_cox_transform, back_box_cox_transform
)
from peshbeen.helpers import seasonal_diff, undiff_ts, invert_seasonal_diff
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
import warnings
warnings.filterwarnings("ignore")

class ml_multi_forecaster:
    """
    Multi-Series Interdependent Machine Learning Forecaster.

    Provides a unified model-agnostic interface supporting all scikit-learn regressors, LightGBM,
    and CatBoost for simultaneous multi-series panel forecasting with cross-series lag dependencies.
    Combines robust panel feature engineering with high-performance recursive forecasting.
    """
    def __init__(
        self,
        model: Any,
        id_col: str,
        target_col: str,
        lags: Optional[Union[int, List[int], Dict[str, Union[int, List[int]]]]] = None,
        lag_transform: Optional[Union[List[Any], Dict[str, List[Any]]]] = None,
        id_col_encoder: Optional[Any] = None,
        difference: Optional[Union[int, Dict[str, int]]] = None,
        seasonal_diff: Optional[Union[int, Dict[str, int]]] = None,
        trend: Optional[Union[str, Dict[str, str]]] = None,
        pol_degree: Union[int, Dict[str, int]] = 1,
        ets_params: Optional[Dict[str, Any]] = None,
        change_points: Optional[Union[list, Dict[str, list]]] = None,
        box_cox: Optional[Union[bool, float, int, Dict[str, Any]]] = False,
        box_cox_biasadj: Optional[Union[bool, Dict[str, bool]]] = False,
        target_scaler: Optional[Any] = None,
        cat_variables: Optional[List[str]] = None,
        categorical_encoder: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the ml_multi_forecaster with the specified model and preprocessing options.

        Parameters
        ----------
        model : Any
            A regression model object (e.g. Ridge(), Lasso(), LinearRegression(), LGBMRegressor(), CatBoostRegressor(), etc.).
        id_col : str
            Name of the column containing unique series identifiers.
        target_col : str
            Name of the target variable column in the input DataFrame.
        lags : int, list of int, or dict of {str: int or list of int}, optional
            Lags to include as features. Can be specified globally as an integer (lags 1 to N) or list of integers, or as a dictionary mapping each series ID to its specific lag configuration. Default is None (no lag features).
        lag_transform : list of callable or dict of {str: list of callable}, optional
            List of lag-transformation functions (e.g. [expanding_mean(shift=1), rolling_quantile(window_size=7, quantile=0.5, shift=1)]). Can be specified globally or per series as a dictionary. Default is None (no lag transforms).
        id_col_encoder : object or str or None, default 'None'
            Strategy or scikit-learn transformer to encode unique series identifiers in panel data. Options are:
            - None: Pass series ID directly as a native categorical feature to models with native categorical support (LGBMRegressor, CatBoostRegressor, XGBRegressor, and HistGradientBoostingRegressor).
            - scikit-learn transformer: Any transformer instance such as OneHotEncoder(sparse_output=False), OrdinalEncoder(), TargetEncoder(), etc.
            - 'ordinal': Convenience shortcut for OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).
        difference : int or dict of {str: int}, optional
            Order of ordinary differencing to apply to each series before modeling. Default is None (no differencing).
        seasonal_diff : int or dict of {str: int}, optional
            Seasonal period for seasonal differencing (e.g. 7 for weekly, 12 for monthly, 24 for hourly). Default is None (no seasonal differencing).
        trend : str or dict of {str: str}, optional
            Trend removal strategy. Options are:
            - 'linear': Global or piecewise linear trend estimation and removal.
            - 'ets': Holt-Winters Exponential Smoothing trend estimation and removal.
            Default is None (no trend removal).
        pol_degree : int or dict of {str: int}, default 1
            Degree of polynomial trend to fit when using 'linear' trend strategy.
        ets_params : dict, optional
            Dictionary of keyword arguments passed to statsmodels ExponentialSmoothing when trend='ets'.
        change_points : list of int or dict of {str: list of int}, optional
            List of integer time indices where slope change points occur for piecewise linear trend fitting. Default is None.
        box_cox : bool, float, int, or dict, default False
            Whether to apply Box-Cox transformation for variance stabilization. If True, estimates optimal lambda. If float/int, uses that value as fixed lambda.
        box_cox_biasadj : bool or dict of {str: bool}, default False
            Whether to apply bias adjustment when back-transforming Box-Cox forecasts. Default is False.
        target_scaler : object or dict of {str: object}, optional
            Scikit-learn compatible scaler instance (e.g. StandardScaler(), RobustScaler(), MinMaxScaler()) or dictionary of per-series scalers. Scalers are fitted on each series' target data and inverted on forecasts. Default is None.
        cat_variables : list of str, optional
            List of categorical feature column names in exogenous data. When categorical_encoder is None, native categorical support is used (e.g. LGBMRegressor, CatBoostRegressor, XGBRegressor with enable_categorical=True, or HistGradientBoostingRegressor with categorical_features='from_dtype'). Default is None.
        categorical_encoder : object, optional
            Scikit-learn compatible transformer (e.g. OneHotEncoder(drop='first', sparse_output=False), OrdinalEncoder(), TargetEncoder()) to encode `cat_variables`. If None, categorical features must be natively supported by the model. Default is None.

        Returns
        -------
        None
        """
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.id_col = id_col
        self.target_col = target_col

        # Backwards compatibility: absorb legacy series_encoding keyword argument if passed via kwargs
        # if 'series_encoding' in kwargs:
        #     id_col_encoder = kwargs.pop('series_encoding')

        self.id_col_encoder = id_col_encoder

        # If id_col_encoder is None, model must have native categorical feature support
        if self.id_col_encoder is None:
            if self.model_name not in ["LGBMRegressor", "CatBoostRegressor", "XGBRegressor", "HistGradientBoostingRegressor"]:
                raise ValueError(
                    "id_col_encoder=None (native categorical) is only supported for LGBMRegressor, CatBoostRegressor, XGBRegressor, and HistGradientBoostingRegressor. "
                    "For other models (e.g. Ridge, Lasso, LinearRegression), please pass an encoder object such as OneHotEncoder(), OrdinalEncoder(), or TargetEncoder()."
                )

        self.lags = lags
        self.lag_transform = lag_transform
        self.difference = difference
        self.seasonal_diff = seasonal_diff
        self.trend = trend
        self.pol_degree = pol_degree
        self.ets_params = ets_params or {}
        self.change_points = change_points
        self.box_cox = box_cox
        self.box_cox_biasadj = box_cox_biasadj
        self.target_scaler = target_scaler
        self.cat_variables = cat_variables
        self.cat_encoder = categorical_encoder
        self.cat_dtypes = {} # CategoricalDtype for each exogenous categorical variable if cat_encoder is None
        self.cat_type = None # CategoricalDtype for id_col if id_col_encoder is None
        self.id_preprocess = None
        self.preprocess = None
        self.encoded_id_cols = []

        # Validate that model can handle exogenous categoricals natively if no encoder is provided
        if self.cat_variables is not None and self.cat_encoder is None:
            if self.model_name not in ["LGBMRegressor", "CatBoostRegressor", "XGBRegressor", "HistGradientBoostingRegressor"]:
                raise ValueError(
                    "Model must be LGBMRegressor, CatBoostRegressor, XGBRegressor, or HistGradientBoostingRegressor to handle categorical variables without an encoder."
                )

        # Configure models with native categorical flags if needed
        has_native_cat = (self.id_col_encoder is None) or (self.cat_variables is not None and self.cat_encoder is None)
        if has_native_cat:
            if self.model_name == "XGBRegressor" and hasattr(self.model, "set_params"):
                if not getattr(self.model, "enable_categorical", False):
                    try:
                        self.model.set_params(enable_categorical=True)
                    except Exception:
                        pass
            elif self.model_name == "HistGradientBoostingRegressor" and hasattr(self.model, "set_params"):
                if getattr(self.model, "categorical_features", None) is None:
                    try:
                        self.model.set_params(categorical_features="from_dtype")
                    except Exception:
                        pass

    def copy(self) -> "ml_multi_forecaster":
        """
        Create a deep copy of the forecaster instance.
        """
        return copy.deepcopy(self)

    def _get_per_series_param(self, param: Any, series_id: str, default: Any = None) -> Any:
        if isinstance(param, dict):
            return param.get(series_id, default)
        return param if param is not None else default

    def _normalize_lags(self, series_ids: List[str]) -> Dict[str, List[int]]:
        result = {}
        for s in series_ids:
            s_lag = self._get_per_series_param(self.lags, s, None)
            if s_lag is None:
                result[s] = []
            elif isinstance(s_lag, int):
                result[s] = list(range(1, s_lag + 1))
            elif isinstance(s_lag, list):
                result[s] = s_lag
            else:
                raise TypeError(f"Lags for series '{s}' must be int, list of ints, or None.")
        return result

    def create_encoded_features(self, df: pd.DataFrame, is_id_col: bool = False) -> pd.DataFrame:
        """
        Encode categorical features (either id_col or exogenous cat_variables)
        using the configured encoder (OneHotEncoder, OrdinalEncoder, TargetEncoder, or native categoricals).
        """
        dfc = df.copy()

        if is_id_col:
            # -------------------------------------------------------------
            # ID Column Encoding / Native Categorical Registration
            # -------------------------------------------------------------
            if self.id_col in dfc.columns:
                if self.id_col_encoder is None:
                    # Native categorical support: register CategoricalDtype and cast
                    if self.cat_type is None:
                        cats = sorted(dfc[self.id_col].dropna().unique().tolist())
                        self.cat_type = pd.CategoricalDtype(categories=cats)
                    dfc[self.id_col] = pd.Categorical(dfc[self.id_col], dtype=self.cat_type)
                    self.encoded_id_cols = []
                    return dfc
                else:
                    enc_inst = clone(self.id_col_encoder)

                    # Ensure dense output for encoders that default to sparse matrices
                    if hasattr(enc_inst, "sparse_output") and getattr(enc_inst, "sparse_output", False):
                        enc_inst.set_params(sparse_output=False)
                    if hasattr(enc_inst, "sparse") and getattr(enc_inst, "sparse", False):
                        enc_inst.set_params(sparse=False)
                    # For TargetEncoder, regression target is continuous
                    if isinstance(enc_inst, TargetEncoder) and getattr(enc_inst, "target_type", "auto") == "auto":
                        enc_inst.set_params(target_type="continuous")

                    if self.target_col in dfc.columns:
                        self.id_preprocess = ColumnTransformer(
                            transformers=[("id", enc_inst, [self.id_col])],
                            remainder="drop",
                            verbose_feature_names_out=False
                        ).set_output(transform="pandas")

                        target_series = dfc[self.target_col]
                        X_encoded = self.id_preprocess.fit_transform(dfc[[self.id_col]], y=target_series)

                        # Avoid column name collision with original self.id_col series
                        if self.id_col in X_encoded.columns:
                            X_encoded = X_encoded.rename(columns={self.id_col: f"{self.id_col}_encoded"})
                        self.encoded_id_cols = list(X_encoded.columns)
                        cols_lead = [c for c in [self.id_col, self.target_col] if c in dfc.columns]
                        other_cols = [c for c in dfc.columns if c not in cols_lead]
                        return pd.concat([dfc[cols_lead], X_encoded, dfc[other_cols]], axis=1)
                    else:
                        if hasattr(self, "id_preprocess") and self.id_preprocess is not None:
                            X_encoded = self.id_preprocess.transform(dfc[[self.id_col]])
                            if self.id_col in X_encoded.columns:
                                X_encoded = X_encoded.rename(columns={self.id_col: f"{self.id_col}_encoded"})
                            cols_lead = [c for c in [self.id_col] if c in dfc.columns]
                            other_cols = [c for c in dfc.columns if c not in cols_lead]
                            return pd.concat([dfc[cols_lead], X_encoded, dfc[other_cols]], axis=1)
                        return dfc

            return dfc

        else:
            # -------------------------------------------------------------
            # Exogenous Categorical Features Encoding
            # -------------------------------------------------------------
            if self.cat_variables is not None:
                # First register category definitions for native categorical support
                for col in self.cat_variables:
                    if col in dfc.columns:
                        if col not in self.cat_dtypes:
                            cats = sorted(dfc[col].dropna().unique().tolist())
                            self.cat_dtypes[col] = pd.CategoricalDtype(categories=cats)
                        dfc[col] = pd.Categorical(dfc[col], dtype=self.cat_dtypes[col])

                # Determine ID columns and ensure no duplicates
                id_cols = [self.id_col]
                if self.id_col_encoder is not None:
                    id_cols.extend([c for c in getattr(self, "encoded_id_cols", []) if c != self.id_col and c in dfc.columns])

                cols_lead = [c for c in id_cols if c in dfc.columns]
                if self.target_col in dfc.columns and self.target_col not in cols_lead:
                    idx = cols_lead.index(self.id_col) + 1 if self.id_col in cols_lead else 0
                    cols_lead.insert(idx, self.target_col)
                cols_to_keep = list(dict.fromkeys(cols_lead))

                # If an explicit encoder was configured, apply ColumnTransformer
                if self.cat_encoder is not None:
                    if self.target_col in dfc.columns:
                        num_cols = [
                            c for c in dfc.columns
                            if c not in self.cat_variables + cols_to_keep
                        ]
                        enc_inst = clone(self.cat_encoder)

                        # Ensure dense output for encoders that default to sparse matrices
                        if hasattr(enc_inst, "sparse_output") and getattr(enc_inst, "sparse_output", False):
                            enc_inst.set_params(sparse_output=False)
                        if hasattr(enc_inst, "sparse") and getattr(enc_inst, "sparse", False):
                            enc_inst.set_params(sparse=False)
                        # For TargetEncoder, regression target is continuous
                        if isinstance(enc_inst, TargetEncoder) and getattr(enc_inst, "target_type", "auto") == "auto":
                            enc_inst.set_params(target_type="continuous")

                        self.preprocess = ColumnTransformer(
                            transformers=[("cat", enc_inst, self.cat_variables), ("num", "passthrough", num_cols)],
                            remainder="drop",
                            verbose_feature_names_out=False
                        ).set_output(transform="pandas")

                        target_series = dfc[self.target_col]
                        X_encoded = self.preprocess.fit_transform(dfc.drop(columns=cols_to_keep), y=target_series)
                        return pd.concat([dfc[cols_to_keep], X_encoded], axis=1)
                    else:
                        X_drop = dfc.drop(columns=cols_to_keep) if cols_to_keep else dfc
                        X_encoded = self.preprocess.transform(X_drop)
                        if cols_to_keep:
                            return pd.concat([dfc[cols_to_keep], X_encoded], axis=1)
                        return X_encoded
                else:
                    # Native categoricals: order columns cleanly
                    cat_cols = [c for c in self.cat_variables if c in dfc.columns]
                    num_cols = [c for c in dfc.columns if c not in cols_to_keep + cat_cols]
                    ordered_cols = cols_to_keep + cat_cols + num_cols
                    return dfc[ordered_cols]

            return dfc

    def data_prep(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        High-performance transformation of long-format panel data into model feature matrix X and target y.
        Encodes exogenous variables, builds cross-series lag matrices, handles series ID encoding, and filters NaNs.
        """
        dfc = df.copy()

        # Step 1: Preprocess Series ID if present (encode via id_col_encoder or register native Categorical)
        if self.id_col in dfc.columns:
            dfc = self.create_encoded_features(dfc, is_id_col=True)

        # Step 2: Preprocess Exogenous Categoricals if specified
        if self.cat_variables is not None:
            dfc = self.create_encoded_features(dfc, is_id_col=False)

        # If target_col not in dfc, we are preparing exogenous data for forecasting (matches 01_ml_forecast pattern)
        if self.target_col not in dfc.columns:
            return dfc

        series_ids = sorted(dfc[self.id_col].unique().tolist())
        self.series_ids = series_ids
        if self.cat_type is None:
            self.cat_type = pd.CategoricalDtype(categories=series_ids)

        
        # 1. Pivot long-format DataFrame to wide target matrix (timestamps x series)
        wide_orig = dfc.pivot(columns=self.id_col, values=self.target_col)
        self.wide_orig = wide_orig.copy() # original wide target matrix (timestamps x series)
        wide_trans = wide_orig.copy() # will be transformed in-place for Box-Cox, Trend, Differencing, and Target Scaling

        self.transform_meta = {s: {} for s in series_ids}
        normalized_lags = self._normalize_lags(series_ids)
        self.normalized_lags = normalized_lags

        # 2. Forward Transformations applied per-series (Box-Cox, Trend, Differencing, Target Scaler)
        for s in series_ids:
            meta = self.transform_meta[s]
            meta['orig_series'] = wide_orig[s].copy()
            s_data = wide_trans[s].copy()

            # Step A: Box-Cox Transformation
            bc_param = self._get_per_series_param(self.box_cox, s, False)
            if bc_param:
                meta['orig_before_boxcox'] = s_data.copy()
                lmda = None if isinstance(bc_param, bool) else bc_param
                is_zero = bool(np.any(s_data.dropna() < 1))
                trans_s, final_lmda = box_cox_transform(x=s_data, shift=is_zero, box_cox_lmda=lmda)
                meta['box_cox'] = True
                meta['is_zero'] = is_zero
                meta['lmda'] = final_lmda
                meta['biasadj'] = self._get_per_series_param(self.box_cox_biasadj, s, False)
                s_data = pd.Series(trans_s, index=s_data.index)
            else:
                meta['box_cox'] = False

            # Step B: Trend Estimation & Removal (Linear or ETS)
            tr_param = self._get_per_series_param(self.trend, s, None)
            if tr_param is not None:
                meta['trend_type'] = tr_param
                pol = self._get_per_series_param(self.pol_degree, s, 1)
                cps = self._get_per_series_param(self.change_points, s, None)
                if tr_param == 'linear':
                    if cps is not None:
                        trend_vals, lr_mod, _ = lr_trend_model(s_data, degree=pol, breakpoints=cps, type='piecewise')
                    else:
                        trend_vals, lr_mod, _ = lr_trend_model(s_data, degree=pol)
                    meta['lr_model'] = lr_mod
                    meta['pol_degree'] = pol
                    meta['cps'] = cps
                    meta['trend_vals'] = trend_vals
                    s_data = s_data - trend_vals
                elif tr_param == 'ets':
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    ets_mod = ExponentialSmoothing(s_data, **self.ets_params)
                    ets_fit = ets_mod.fit()
                    trend_vals = ets_fit.fittedvalues
                    meta['ets_model_fit'] = ets_fit
                    meta['trend_vals'] = trend_vals
                    s_data = s_data - trend_vals
            else:
                meta['trend_type'] = None

            # Step C: Ordinary Differencing
            diff_param = self._get_per_series_param(self.difference, s, None)
            if diff_param is not None:
                meta['difference'] = diff_param
                meta['orig_before_diff'] = s_data.tolist()
                s_data = pd.Series(
                    np.diff(s_data, n=diff_param, prepend=np.repeat(np.nan, diff_param)),
                    index=s_data.index
                )
            else:
                meta['difference'] = None

            # Step D: Seasonal Differencing
            sdiff_param = self._get_per_series_param(self.seasonal_diff, s, None)
            if sdiff_param is not None:
                meta['seasonal_diff'] = sdiff_param
                meta['orig_before_sdiff'] = s_data.tolist()
                s_data = pd.Series(seasonal_diff(s_data, sdiff_param), index=s_data.index)
            else:
                meta['seasonal_diff'] = None

            # Step E: Target Scaling (e.g. StandardScaler, RobustScaler)
            if isinstance(self.target_scaler, dict):
                scaler_inst = self.target_scaler.get(s, None)
            else:
                scaler_inst = copy.deepcopy(self.target_scaler) if self.target_scaler is not None else None

            if scaler_inst is not None:
                meta['scaler'] = scaler_inst
                valid_mask_s = ~s_data.isna()
                if np.any(valid_mask_s):
                    scaled_vals = meta['scaler'].fit_transform(s_data[valid_mask_s].to_numpy().reshape(-1, 1)).ravel()
                    s_data.loc[valid_mask_s] = scaled_vals
            else:
                meta['scaler'] = None

            wide_trans[s] = s_data

        self.wide_trans = wide_trans.copy()
    
        ## first get lag names per series in a list, second get and numpy array of all laggeed values with correct shift and order to be consistet with order of series_ids and lags,
        self.lag_names = [f"{s}_lag_{lag}" for s in series_ids for lag in normalized_lags[s]]
        lagged_values = np.column_stack([
            wide_trans[s].shift(lag).to_numpy() for s in series_ids for lag in normalized_lags[s]
        ]) # shape (N_time, N_series * N_lags)
        # do the same for lag_transform features and store in self.lag_transform_names and self.lag_transform_values
        lag_transform_names = []
        lag_transform_values = []
        if self.lag_transform is not None:
            for s in series_ids:
                s_lag_tf = self._get_per_series_param(self.lag_transform, s, None)
                if s_lag_tf is not None:
                    for func in s_lag_tf:
                        fname = getattr(func, '__name__', func.__class__.__name__)
                        col_name = f"{s}_{fname}"
                        if hasattr(func, 'shift'):
                            col_name += f"_shift_{func.shift}"
                        if hasattr(func, 'window_size'):
                            col_name += f"_{func.window_size}"
                        if hasattr(func, 'quantile'):
                            col_name += f"_q{func.quantile}"
                        lag_transform_names.append(col_name)
                        lag_transform_values.append(func(wide_trans[s]).to_numpy())
        ## append lag_transform features to lag_names and lag_transform_values to lagged values to create a combined feature matrix
        self.lag_names.extend(lag_transform_names) if lag_transform_names else self.lag_names
        lagged_values = np.column_stack([lagged_values] + lag_transform_values) if lag_transform_values else lagged_values

        # self.feat_df = pd.DataFrame(self.lagged_values, columns=self.lag_names, index=wide_trans.index)
        self.n_series = len(series_ids)
    
        # Ensure dfc is strictly ordered by series ID and chronological index
        dfc = dfc.sort_index().sort_values(by=[self.id_col], kind='stable')
        feat_df_tiled = pd.DataFrame(np.tile(lagged_values, (self.n_series, 1)), index=dfc.index, columns=self.lag_names)
        
        # Concat using reset_index to guarantee exact 1-to-1 row alignment, then restore original index
        orig_index = dfc.index
        df_all = pd.concat([dfc.reset_index(drop=True), feat_df_tiled.reset_index(drop=True)], axis=1)
        df_all.index = orig_index

        X_all = df_all.drop(columns=[self.target_col]) if self.id_col_encoder is None else df_all.drop(columns=[self.target_col] + [self.id_col]) # drop id_col if id_col_encoder is None (native categorical) to avoid duplicate columns
        y_all = df_all[self.target_col]

        # 7. Fast Mask Filtering (drop rows with NaNs from lags/differencing)
        valid_mask = ~(X_all.isna().any(axis=1).to_numpy() | np.isnan(y_all.to_numpy()))
        # Track exact series ID for every remaining valid row in X (used by predict_in_sample)
        self.series_id_panel = df_all.loc[valid_mask, self.id_col].to_numpy() # shape (N_valid_rows,)
        
        return X_all.iloc[valid_mask].copy(), y_all.iloc[valid_mask].copy(), feat_df_tiled

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the multi-series forecaster on the input panel DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Long-format panel DataFrame containing time index, id_col, and target_col.

        Returns
        -------
        None
        """
        self.df_train = df.copy()
        X, y, _ = self.data_prep(df)
        self.X = X
        self.y = y
        self.X_cols = X.columns.tolist()

        # Build categorical feature parameters for native tree model training
        fit_kwargs = {}
        cat_cols = []
        if self.id_col_encoder is None:
            cat_cols.append(self.id_col)
        if self.cat_variables is not None and self.cat_encoder is None:
            cat_cols.extend([c for c in self.cat_variables if c in self.X_cols])

        if len(cat_cols) > 0:
            if self.model_name == "LGBMRegressor":
                fit_kwargs = {"categorical_feature": cat_cols}
            elif self.model_name == "CatBoostRegressor":
                fit_kwargs = {"cat_features": cat_cols, "verbose": False}
            elif self.model_name == "XGBRegressor":
                if hasattr(self.model, "set_params") and not getattr(self.model, "enable_categorical", False):
                    try:
                        self.model.set_params(enable_categorical=True)
                    except Exception:
                        pass
            elif self.model_name == "HistGradientBoostingRegressor":
                if hasattr(self.model, "set_params") and getattr(self.model, "categorical_features", None) is None:
                    try:
                        self.model.set_params(categorical_features="from_dtype")
                    except Exception:
                        pass

        self.model_fit = self.model.fit(X, y, **fit_kwargs)

    def forecast(self, H: int, exog: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Recursive multi-step forecast across all target series using pre-allocated high-speed memory buffers.

        Parameters
        ----------
        H : int
            Forecast horizon length.
        exog : pd.DataFrame, optional
            Future exogenous features DataFrame covering horizon H. Default is None.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary keyed by series ID with length-H forecast arrays on original measurement scale.
        """
        if not hasattr(self, "model_fit"):
            raise ValueError("Model has not been fitted yet. Call .fit() before .forecast().")

        if exog is not None:
            prep_exog = self.data_prep(exog)
        y_lists: Dict[str, list] = {col: self.wide_trans[col].tolist() for col in self.series_ids}
        raw_forecasts = {s: np.zeros(H, dtype=np.float64) for s in self.series_ids}

        idx_list = prep_exog.index.unique().tolist() if exog is not None else list(range(H))  

        series_ids = self.series_ids
        N = len(series_ids)
        

        # # Encoded series ID features (precomputed during fit/data_prep, invariant across horizons)
        for t in range(H):
            # Exogenous features for step t
            if exog is not None:
                x_var = prep_exog.loc[idx_list[t]]
            if isinstance(x_var, pd.Series):
                x_var = x_var.to_frame().T  # Convert to DataFrame if it's a Series
            if self.id_col in x_var.columns:
                x_var = x_var.set_index(self.id_col).reindex(self.series_ids).reset_index() # Ensure x_var is ordered by series_ids           
                ## make sure id_col is cat_type if id_col_encoder is None (native categorical)
                if self.id_col_encoder is None:
                    x_var[self.id_col] = pd.Categorical(x_var[self.id_col], dtype=self.cat_type)

            else:
                x_var = pd.DataFrame({self.id_col: series_ids})
                if self.id_col_encoder is not None and hasattr(self, "id_preprocess") and self.id_preprocess is not None:
                    X_enc = self.id_preprocess.transform(x_var[[self.id_col]])
                    if self.id_col in X_enc.columns:
                        X_enc = X_enc.rename(columns={self.id_col: f"{self.id_col}_encoded"})
                    x_var = pd.concat([x_var, X_enc], axis=1)
                elif self.cat_type is not None:
                    x_var[self.id_col] = pd.Categorical(x_var[self.id_col], dtype=self.cat_type)
            
            ## lagged features for step t (precomputed during fit/data_prep, invariant across horizons)
            self.x_var = x_var.copy() # save for debugging/inspection
            self.lagged_forecasts = np.concatenate([
                [y_lists[s][-lg] for lg in lgs] for s, lgs in self.normalized_lags.items()
            ]) if self.lags is not None else np.array([])  # shape (sum(len(normalized_lags[s]) for s in series_ids),)

            ## lag_transform features for step t (precomputed during fit/data_prep, invariant across horizons)
            ## lag_transform features for step t
            if self.lag_transform is not None:
                lag_transform_features = []
                for s in series_ids:
                    s_lag_tf = self._get_per_series_param(self.lag_transform, s, None)
                    if s_lag_tf is not None:
                        # Convert list to numpy array once per series instead of creating pd.Series
                        s_arr = np.asarray(y_lists[s])
                        for func in s_lag_tf:
                            lag_res = func(s_arr)
                            lag_val = lag_res[-1] if hasattr(lag_res, '__len__') else lag_res
                            lag_transform_features.append(lag_val)
                self.lagged_forecasts = np.concatenate([self.lagged_forecasts, lag_transform_features])

            
            x_var_features = x_var.drop(columns=[self.id_col]) if (self.id_col_encoder is not None and self.id_col in x_var.columns) else x_var
            repeated_lagged_forecasts = np.tile(self.lagged_forecasts, (N, 1)) if self.lagged_forecasts.size > 0 else np.empty((N, 0))

            if self.lagged_forecasts.size > 0:
                repeat_lag_df = pd.DataFrame(repeated_lagged_forecasts, index=x_var_features.index, columns=self.lag_names)
                # Concat positionally to prevent any index alignment mismatch
                x_full = pd.concat([x_var_features.reset_index(drop=True), repeat_lag_df.reset_index(drop=True)], axis=1)
                x_full.index = x_var_features.index
            else:
                x_full = x_var_features
            # Strictly ensure feature columns match the exact training column order
            x_full = x_full[self.X_cols]
            preds_h = self.model_fit.predict(x_full) # shape (N,)
            ## update y_lists with new predictions for each series
            for s_idx, s in enumerate(series_ids):
                ## append the new prediction to the corresponding series' history list for use in future lagged features
                y_lists[s].append(preds_h[s_idx])
                raw_forecasts[s][t] = preds_h[s_idx]

        forecasts = {}
        for s_idx, s in enumerate(series_ids):
            meta = self.transform_meta[s]
            pred_series = raw_forecasts[s].copy()

            if meta.get('scaler') is not None:
                pred_series = meta['scaler'].inverse_transform(pred_series.reshape(-1, 1)).ravel()

            if meta.get('seasonal_diff') is not None:
                pred_series = invert_seasonal_diff(meta['orig_before_sdiff'], pred_series, meta['seasonal_diff'])

            if meta.get('difference') is not None:
                pred_series = undiff_ts(meta['orig_before_diff'], pred_series, n=meta['difference'])

            if meta.get('trend_type') is not None:
                if meta['trend_type'] == 'linear':
                    trend_fc, _ = forecast_trend(
                        model=meta['lr_model'], H=H, start=len(meta['orig_series']),
                        degree=meta['pol_degree'], breakpoints=meta['cps']
                    )
                elif meta['trend_type'] == 'ets':
                    trend_fc = meta['ets_model_fit'].forecast(H).values
                pred_series = pred_series + trend_fc

            if meta.get('box_cox', False):
                lmda = meta['lmda']
                if lmda is not None and lmda != 0:
                    min_val = -1.0 / lmda + 1e-6 if lmda > 0 else -1e6
                    pred_series = np.maximum(pred_series, min_val)
                pred_series = back_box_cox_transform(
                    y_pred=pred_series, lmda=meta['lmda'],
                    shift=meta['is_zero'], box_cox_biasadj=meta.get('biasadj', False)
                )

            pred_series = np.nan_to_num(pred_series, nan=0.0, posinf=0.0, neginf=0.0)
            pred_series = np.clip(pred_series, a_min=0.0, a_max=None)
            forecasts[s] = pred_series

        return forecasts

    def predict_in_sample(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Computes in-sample fitted values and residuals for each series on the original scale.

        Returns
        -------
        fitted_values : Dict[str, np.ndarray]
            Dictionary of in-sample fitted values per series.
        residuals : Dict[str, np.ndarray]
            Dictionary of in-sample residuals (actual - fitted) per series.
        """
        if not hasattr(self, "model_fit"):
            raise ValueError("Model has not been fitted yet. Call .fit() before .predict_in_sample().")

        series_ids = self.series_ids
        fitted_dict = {}
        resid_dict = {}

        raw_in_sample = self.model_fit.predict(self.X)

        df_preds = pd.DataFrame({
            'pred': raw_in_sample,
            'actual': self.y.values
        }, index=self.X.index)

        for s_idx, s in enumerate(series_ids):
            meta = self.transform_meta[s]
            # Exact row-to-series mapping via self.series_id_panel (encoder-agnostic)
            s_mask = (self.series_id_panel == s)
            
            s_df = df_preds[s_mask]
            if len(s_df) > 0:
                sub_preds = s_df['pred'].values.copy()
                sub_y = s_df['actual'].values.copy()

                if meta.get('scaler') is not None:
                    sub_preds = meta['scaler'].inverse_transform(sub_preds.reshape(-1, 1)).ravel()
                    sub_y = meta['scaler'].inverse_transform(sub_y.reshape(-1, 1)).ravel()

                if meta.get('seasonal_diff') is not None:
                    sub_preds = invert_seasonal_diff(meta['orig_before_sdiff'], sub_preds, meta['seasonal_diff'])
                    sub_y = invert_seasonal_diff(meta['orig_before_sdiff'], sub_y, meta['seasonal_diff'])

                if meta.get('difference') is not None:
                    sub_preds = undiff_ts(meta['orig_before_diff'], sub_preds, n=meta['difference'])
                    sub_y = undiff_ts(meta['orig_before_diff'], sub_y, n=meta['difference'])

                if meta.get('trend_type') is not None:
                    if meta.get('trend_vals') is not None:
                        t_vals = meta['trend_vals']
                        if len(t_vals) >= len(sub_preds):
                            t_slice = t_vals[-len(sub_preds):]
                            sub_preds = sub_preds + t_slice
                            sub_y = sub_y + t_slice

                if meta.get('box_cox', False):
                    lmda = meta['lmda']
                    if lmda is not None and lmda != 0:
                        min_val = -1.0 / lmda + 1e-6 if lmda > 0 else -1e6
                        sub_preds = np.maximum(sub_preds, min_val)
                        sub_y = np.maximum(sub_y, min_val)
                    sub_preds = back_box_cox_transform(
                        y_pred=sub_preds, lmda=meta['lmda'],
                        shift=meta['is_zero'], box_cox_biasadj=meta.get('biasadj', False)
                    )
                    sub_y = back_box_cox_transform(
                        y_pred=sub_y, lmda=meta['lmda'],
                        shift=meta['is_zero'], box_cox_biasadj=meta.get('biasadj', False)
                    )

                sub_preds = np.nan_to_num(sub_preds, nan=0.0, posinf=0.0, neginf=0.0)
                sub_preds = np.clip(sub_preds, a_min=0.0, a_max=None)
                sub_resid = sub_y - sub_preds

                fitted_dict[s] = sub_preds
                resid_dict[s] = sub_resid

        self.fitted_values = fitted_dict
        self.residuals = resid_dict
        return fitted_dict, resid_dict

    def cross_validate(
        self,
        df: pd.DataFrame,
        cv_split: int,
        test_size: int,
        metrics: List[Callable],
        step_size: Optional[int] = None,
        ref_series_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Expanding-window cross-validation evaluated across all series.
        Parameters
        ----------
        df : pd.DataFrame
            Long-format panel DataFrame containing time index, id_col, and target_col.
        cv_split : int
            Number of cross-validation folds.
        test_size : int
            Number of time steps in the test set for each fold.
        metrics : List[Callable]
            List of metric functions to evaluate predictions (e.g., mean_squared_error, mean_absolute_error).
        step_size : int, optional
            Step size for expanding window. If None, defaults to test_size.
        ref_series_id : str, optional
            Reference series ID to determine cutoff dates for cross-validation. If None, the series with the fewest observations is used.
        Returns
        -------
        pd.DataFrame
            DataFrame containing detailed cross-validation results with columns: fold, series_id, cutoff_date,
        """
        dfc = df.copy()
        series_ids = sorted(dfc[self.id_col].unique().tolist())
        
        if ref_series_id is None:
            ref_series_id = min(series_ids, key=lambda s: len(dfc[dfc[self.id_col] == s]))
        ref_df = dfc[dfc[self.id_col] == ref_series_id]
        
        splitter = SplitTimeSeries(n_splits=cv_split, test_size=test_size, step_size=step_size)
        splits = list(splitter.split(ref_df))

        metric_names = [getattr(m, '__name__', m.__class__.__name__) for m in metrics]
        rows = []
        fold_scores = {mname: {s: [] for s in series_ids} for mname in metric_names}

        for fold, (train_idx, test_idx) in enumerate(splits):
            cutoff = ref_df.index[train_idx[-1]]
            test_dates = ref_df.index[test_idx] ## test dates or timestamps for this fold
            H_fold = len(test_dates)

            train_df = dfc[dfc.index <= cutoff]
            test_df = dfc[dfc.index.isin(test_dates)]

            forecaster = copy.deepcopy(self)
            forecaster.fit(train_df)

            fold_exog = test_df.drop(columns=[self.target_col])
            if fold_exog.empty:
                fold_exog = None

            preds = forecaster.forecast(H=H_fold, exog=fold_exog)

            for s in series_ids:
                s_test = test_df[test_df[self.id_col] == s]
                y_true = s_test[self.target_col].values
                y_pred = preds[s][:len(y_true)]

                for h_idx, (d, yt, yp) in enumerate(zip(s_test.index, y_true, y_pred)):
                    rows.append({
                        'fold': fold + 1,
                        self.id_col: s,
                        'cutoff_date': cutoff,
                        'horizon': h_idx + 1,
                        'y_true': yt,
                        'y_pred': yp
                    })

                for mfunc, mname in zip(metrics, metric_names):
                    s_train = train_df[train_df[self.id_col] == s]
                    y_train_s = s_train[self.target_col].values
                    try:
                        score = mfunc(y_true, y_pred, y_train=y_train_s)
                    except TypeError:
                        score = mfunc(y_true, y_pred)
                    fold_scores[mname][s].append(score)

        df_detailed = pd.DataFrame(rows)

        summary_dict = {}
        for s in series_ids:
            summary_dict[s] = {mname: np.mean(fold_scores[mname][s]) for mname in metric_names}
        
        summary_dict["overall"] = {mname: np.mean([np.mean(fold_scores[mname][s]) for s in series_ids]) for mname in metric_names}
        self.cv_summary = pd.DataFrame(summary_dict)

        return df_detailed
```
:::


::: {#eeb7b757 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from peshbeen.transformations import rolling_mean, expanding_mean
from peshbeen.metrics import MAE, RMSE, MASE
from peshbeen.datasets import load_sales
import numpy as np
import pandas as pd
df = load_sales()

item_filter = df["store_item"].unique().tolist()[:4]
df = df[df["store_item"].isin(item_filter)].copy()
df["day_of_week"] = df.index.dayofweek.astype(str)
df["month"] = df.index.month.astype(str)
df["time_index"] = np.arange(len(df), dtype=np.int64)
df["gm_ch"] = np.random.randint(0, 10, size=len(df), dtype=np.int64)

df_train = df[df.index <= "2025-12-31"]
df_test = df[df.index >= "2026-01-01"]
df_exog = df_test.drop(columns=["sales"]).copy()
H = len(df_test.index.drop_duplicates())

forecaster7 = ml_multi_forecaster(
    model=LGBMRegressor(n_jobs=1, verbose=-1),
    id_col="store_item",
    target_col="sales",
    lags=3,
    id_col_encoder=None,
    cat_variables=["day_of_week", "month"],
    categorical_encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
)
forecaster7.fit(df_train)
fc7 = forecaster7.forecast(H=H, exog=df_exog)
```
:::


::: {#test_suite_hide_cell .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from peshbeen.transformations import rolling_mean, expanding_mean
from peshbeen.metrics import MAE, RMSE, MASE
from peshbeen.datasets import load_sales
from peshbeen.models.ml_multi_forecaster import ml_multi_forecaster
import numpy as np
import pandas as pd

df = load_sales()
df["day_of_week"] = df.index.dayofweek.astype(str)
df["month"] = df.index.month.astype(str)
df["time_index"] = np.arange(len(df), dtype=np.int64)
df["gm_ch"] = np.random.randint(0, 10, size=len(df), dtype=np.int64)

df_train = df[df.index <= "2025-12-31"]
df_test = df[df.index >= "2026-01-01"]
df_exog = df_test.drop(columns=["sales"]).copy()
H = len(df_test.index.drop_duplicates())

print("\n=== Comprehensive Unit Tests ===")

# Test 1: id_col_encoder=OneHotEncoder() with target_scaler, difference, trend, and exog
forecaster1 = ml_multi_forecaster(
    model=Ridge(alpha=1.0),
    id_col="store_item",
    target_col="sales",
    lags=7,
    id_col_encoder=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    difference=1,
    trend="linear",
    target_scaler=StandardScaler(),
    cat_variables=["day_of_week", "month"],
    categorical_encoder=OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
)
forecaster1.fit(df_train)

scaler_s1 = forecaster1.transform_meta["store_01_item_01"]["scaler"]
assert scaler_s1 is not None, "Scaler must not be None in transform_meta!"
assert hasattr(scaler_s1, "mean_"), "StandardScaler must have fitted mean_ attribute"
assert hasattr(scaler_s1, "scale_"), "StandardScaler must have fitted scale_ attribute"

fc1 = forecaster1.forecast(H=H, exog=df_exog)
assert isinstance(fc1, dict)
assert "store_01_item_01" in fc1
assert len(fc1["store_01_item_01"]) == H
for s, f_vals in fc1.items():
    assert not np.any(np.isnan(f_vals))
    assert np.all(f_vals >= 0.0)
print("Test 1: OneHotEncoder ID + target scaler + forecast verification PASSED!")

# Test 2: OrdinalEncoder for ID column + TargetEncoder for exogenous
forecaster2 = ml_multi_forecaster(
    model=Ridge(alpha=1.0),
    id_col="store_item",
    target_col="sales",
    lags=3,
    id_col_encoder=OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    cat_variables=["day_of_week", "month"],
    categorical_encoder=TargetEncoder(smooth="auto", cv=3)
)
forecaster2.fit(df_train)
fc2 = forecaster2.forecast(H=H, exog=df_exog)
assert len(fc2) == 72
print("Test 2: OrdinalEncoder ID + TargetEncoder Exog PASSED!")

# Test 3: TargetEncoder for ID column + OneHotEncoder for exogenous + predict_in_sample
forecaster3 = ml_multi_forecaster(
    model=Ridge(alpha=1.0),
    id_col="store_item",
    target_col="sales",
    lags=3,
    id_col_encoder=TargetEncoder(smooth="auto", cv=3),
    cat_variables=["day_of_week", "month"],
    categorical_encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)
forecaster3.fit(df_train)
fc3 = forecaster3.forecast(H=H, exog=df_exog)
assert len(fc3) == 72
fit3, res3 = forecaster3.predict_in_sample()
assert len(fit3) == 72 and len(res3) == 72
print("Test 3: TargetEncoder ID + predict_in_sample PASSED!")

# Test 4: Native Categorical support (id_col_encoder=None, categorical_encoder=None) with LightGBM (n_jobs=1)
forecaster4 = ml_multi_forecaster(
    model=LGBMRegressor(n_jobs=1, verbose=-1),
    id_col="store_item",
    target_col="sales",
    lags=3,
    id_col_encoder=None,
    cat_variables=["day_of_week", "month"],
    categorical_encoder=None
)
forecaster4.fit(df_train)
fc4 = forecaster4.forecast(H=H, exog=df_exog)
assert len(fc4) == 72
print("Test 4: Native Categorical (id_col_encoder=None) with LightGBM PASSED!")

# Test 5: Cross validation
cv_df = forecaster1.cross_validate(
    df=df,
    cv_split=3,
    test_size=7,
    metrics=[MAE, RMSE, MASE],
    step_size=7
)
assert isinstance(cv_df, pd.DataFrame)
assert hasattr(forecaster1, "cv_summary")
assert len(cv_df) == 3 * 72 * 7
assert "overall" in forecaster1.cv_summary.columns
assert "MAE" in forecaster1.cv_summary.index
print("Test 5: cross_validate row structure and cv_summary PASSED!")

# Test 6: Pure lag model without exogenous features (OneHotEncoder for ID)
forecaster6 = ml_multi_forecaster(
    model=Ridge(alpha=1.0),
    id_col="store_item",
    target_col="sales",
    lags=5,
    id_col_encoder=OneHotEncoder(sparse_output=False, handle_unknown="ignore")
)
df_train_no_exog = df_train[["store_item", "sales"]].copy()
forecaster6.fit(df_train_no_exog)
fc6 = forecaster6.forecast(H=H, exog=None)
assert isinstance(fc6, dict) and len(fc6) == 72
for s, vals in fc6.items():
    assert len(vals) == H
    assert not np.any(np.isnan(vals))
print("Test 6: Pure lag model without exogenous features PASSED!")

# Test 7: Mixed Encoders: OneHotEncoder (ID) + Native Categorical Exog with LightGBM
forecaster7 = ml_multi_forecaster(
    model=LGBMRegressor(n_jobs=1, verbose=-1),
    id_col="store_item",
    target_col="sales",
    lags=3,
    id_col_encoder=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    cat_variables=["day_of_week", "month"],
    categorical_encoder=None
)
forecaster7.fit(df_train)
fc7 = forecaster7.forecast(H=H, exog=df_exog)
assert len(fc7) == 72
print("Test 7: Mixed OneHotEncoder ID + Native Categorical Exog PASSED!")

# Test 8: Box-Cox transformation + RobustScaler + OrdinalEncoder ID
forecaster8 = ml_multi_forecaster(
    model=Ridge(alpha=1.0),
    id_col="store_item",
    target_col="sales",
    lags=3,
    id_col_encoder=OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    box_cox=True,
    target_scaler=RobustScaler()
)
forecaster8.fit(df_train)
fc8 = forecaster8.forecast(H=H, exog=df_exog)
assert len(fc8) == 72
for s, vals in fc8.items():
    assert len(vals) == H
    assert not np.any(np.isnan(vals))
print("Test 8: Box-Cox + RobustScaler + OrdinalEncoder ID PASSED!")

# Test 9: Input validation: ensure id_col_encoder=None raises ValueError for models without native categoricals
try:
    ml_multi_forecaster(
        model=Ridge(),
        id_col="store_item",
        target_col="sales",
        lags=3,
        id_col_encoder=None
    )
    assert False, "Expected ValueError for Ridge with id_col_encoder=None"
except ValueError as e:
    assert "native categorical" in str(e).lower()
print("Test 9: Native categorical validation for non-tree models PASSED!")

# Test 10: lag_transform with rolling_mean & expanding_mean
forecaster10 = ml_multi_forecaster(
    model=Ridge(alpha=1.0),
    id_col="store_item",
    target_col="sales",
    lags=3,
    lag_transform=[rolling_mean(window_size=7, shift=1), expanding_mean(shift=1)],
    id_col_encoder=OneHotEncoder(sparse_output=False, handle_unknown="ignore")
)
forecaster10.fit(df_train)
fc10 = forecaster10.forecast(H=H, exog=df_exog)
assert len(fc10) == 72
print("Test 10: lag_transform with rolling_mean & expanding_mean PASSED!")

# Test 11: Per-series lag configuration (dict of lags mapping different series)
unique_series = df_train["store_item"].unique()
dict_lags = {s: [1, 2, 7] if i % 2 == 0 else [1, 3] for i, s in enumerate(unique_series)}
forecaster11 = ml_multi_forecaster(
    model=Ridge(alpha=1.0),
    id_col="store_item",
    target_col="sales",
    lags=dict_lags,
    id_col_encoder=OneHotEncoder(sparse_output=False, handle_unknown="ignore")
)
forecaster11.fit(df_train)
fc11 = forecaster11.forecast(H=H, exog=df_exog)
assert len(fc11) == 72
print("Test 11: Per-series lag configuration (dict of lags) PASSED!")

# Test 12: Combined ordinary differencing (difference=1) + seasonal differencing (seasonal_diff=7)
forecaster12 = ml_multi_forecaster(
    model=Ridge(alpha=1.0),
    id_col="store_item",
    target_col="sales",
    lags=3,
    difference=1,
    seasonal_diff=7,
    id_col_encoder=OneHotEncoder(sparse_output=False, handle_unknown="ignore")
)
forecaster12.fit(df_train)
fc12 = forecaster12.forecast(H=H, exog=df_exog)
assert len(fc12) == 72
for s, vals in fc12.items():
    assert len(vals) == H
    assert not np.any(np.isnan(vals))
print("Test 12: Combined ordinary (difference=1) and seasonal differencing (seasonal_diff=7) PASSED!")

# Test 13: HistGradientBoostingRegressor with native categoricals (id_col_encoder=None, categorical_encoder=None)
forecaster13 = ml_multi_forecaster(
    model=HistGradientBoostingRegressor(max_iter=20, random_state=42),
    id_col="store_item",
    target_col="sales",
    lags=3,
    id_col_encoder=None,
    cat_variables=["day_of_week", "month"],
    categorical_encoder=None
)
forecaster13.fit(df_train)
fc13 = forecaster13.forecast(H=H, exog=df_exog)
assert len(fc13) == 72
print("Test 13: HistGradientBoostingRegressor with native categoricals PASSED!")

# Test 14: Per-series target_scaler (dict of scalers) and MinMaxScaler
scaler_dict = {s: MinMaxScaler() if i % 2 == 0 else StandardScaler() for i, s in enumerate(unique_series)}
forecaster14 = ml_multi_forecaster(
    model=Ridge(alpha=1.0),
    id_col="store_item",
    target_col="sales",
    lags=3,
    id_col_encoder=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    target_scaler=scaler_dict
)
forecaster14.fit(df_train)
fc14 = forecaster14.forecast(H=H, exog=df_exog)
assert len(fc14) == 72
for s in unique_series:
    assert s in forecaster14.transform_meta
    assert "scaler" in forecaster14.transform_meta[s]
print("Test 14: Per-series target_scaler (dict of scalers) and MinMaxScaler PASSED!")

# Test 15: forecaster.copy() deep copy verification
f15_copy = forecaster1.copy()
assert f15_copy is not forecaster1
assert f15_copy.id_col == forecaster1.id_col
assert f15_copy.lags == forecaster1.lags
assert f15_copy.difference == forecaster1.difference
print("Test 15: forecaster.copy() deep copy verification PASSED!")

# Test 16: Handling unseen exogenous category levels gracefully during forecast
df_exog_unseen = df_exog.copy()
df_exog_unseen["day_of_week"] = "unknown_holiday"
fc16 = forecaster1.forecast(H=H, exog=df_exog_unseen)
assert isinstance(fc16, dict) and len(fc16) == 72
print("Test 16: Handling unseen exogenous category levels during forecast PASSED!")

# Test 17: Exponential smoothing trend removal (trend="ets")
forecaster17 = ml_multi_forecaster(
    model=Ridge(alpha=1.0),
    id_col="store_item",
    target_col="sales",
    lags=3,
    trend="ets",
    ets_params={"trend": "add"},
    id_col_encoder=OneHotEncoder(sparse_output=False, handle_unknown="ignore")
)
forecaster17.fit(df_train)
fc17 = forecaster17.forecast(H=H, exog=None)
assert isinstance(fc17, dict) and len(fc17) == 72
for s, vals in fc17.items():
    assert len(vals) == H
    assert not np.any(np.isnan(vals))
print("Test 17: Exponential smoothing trend removal (trend='ets') PASSED!")

print("\n=== ALL UNIT TESTS PASSED SUCCESSFULLY! ===")

```

::: {.cell-output .cell-output-stdout}
```

=== Comprehensive Unit Tests ===
Test 1: OneHotEncoder ID + target scaler + forecast verification PASSED!
Test 2: OrdinalEncoder ID + TargetEncoder Exog PASSED!
Test 3: TargetEncoder ID + predict_in_sample PASSED!
Test 4: Native Categorical (id_col_encoder=None) with LightGBM PASSED!
Test 5: cross_validate row structure and cv_summary PASSED!
Test 6: Pure lag model without exogenous features PASSED!
Test 7: Mixed OneHotEncoder ID + Native Categorical Exog PASSED!
Test 8: Box-Cox + RobustScaler + OrdinalEncoder ID PASSED!
Test 9: Native categorical validation for non-tree models PASSED!
Test 10: lag_transform with rolling_mean & expanding_mean PASSED!
Test 11: Per-series lag configuration (dict of lags) PASSED!
Test 12: Combined ordinary (difference=1) and seasonal differencing (seasonal_diff=7) PASSED!
Test 13: HistGradientBoostingRegressor with native categoricals PASSED!
Test 14: Per-series target_scaler (dict of scalers) and MinMaxScaler PASSED!
Test 15: forecaster.copy() deep copy verification PASSED!
Test 16: Handling unseen exogenous category levels during forecast PASSED!
Test 17: Exponential smoothing trend removal (trend='ets') PASSED!

=== ALL UNIT TESTS PASSED SUCCESSFULLY! ===
```
:::
:::


