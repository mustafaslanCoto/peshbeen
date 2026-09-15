---
title: Hyperparameters tuning methods for Univariate machine learning models
---





::: {#623ad0b6 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
import numpy as np
from typing import Optional

#------------------------------------------------------------------------------
# Parametric Time Series Split
#------------------------------------------------------------------------------
class SplitTimeSeries:

    def __init__(self,
                 n_splits: int,
                 test_size: int,
                 step_size: Optional[int] = None
                 ) -> None:

        """
        A time series cross-validator that generates train/test splits with a fixed test size and a configurable step size.

        Parameters
        ----------
        n_splits : int
            Number of splits to generate.
        test_size : int
            The number of samples in each test set.
        step_size : int, optional
            The number of samples to move the test set forward for each split. If None, it defaults to the test_size, meaning non-overlapping test sets.
        """
        self.test_size = test_size
        self.step_size = test_size if step_size is None else step_size
        self.n_splits = n_splits

    def split(self, X):
        n_samples = len(X)
        split_starts = []
        # Start the last test set at the last possible position
        last_test_start = n_samples - self.test_size
        # Build test starts, moving backward with fixed step_size
        current = last_test_start
        while current >= 0:
            split_starts.append(current)
            current -= self.step_size
        split_starts = split_starts[::-1]  # Reverse to start from earliest

        # Use only the last n_splits
        split_starts = split_starts[-self.n_splits:]

        for test_start in split_starts:
            test_end = test_start + self.test_size
            train_index = np.arange(0, test_start)
            test_index = np.arange(test_start, test_end)
            yield train_index, test_index

```
:::


::: {#da8122f3 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from numba import jit
import warnings
warnings.filterwarnings("ignore")
# from tqdm import tqdm_notebook
# from itertools import product
from typing import List, Dict, Optional, Callable, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler

def hyperopt_tune(
    model: Any,
    df: pd.DataFrame,
    cv_split: int,
    test_size: int,
    eval_metric: Callable,
    param_space: Dict[str, Any],
    step_size: int = None,
    eval_num: int = 100,
    candidate_exog: List[str] = None,
    pareto_bounds: Union[float, Tuple[float, float]] = (0.5, 0.999),
    eval_horizons: Union[None, int] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any], List[str]]:

    """
    Tune forecasting model hyperparameters using time series cross-validation and Hyperopt.
 
    Parameters
    ----------
    model : object
        Forecasting model with .fit and .forecast methods.
    df : pd.DataFrame
        Time series data (datetime index, target column, optional exogenous features).
    cv_split : int
        Number of cross-validation splits.
    test_size : int
        Number of samples in each test fold. For ml_direct_forecaster, this will be overridden to be the maximum horizon in model.H.
    eval_metric : Callable
        Metric function to minimise.
    param_space : dict
        Each value must be a callable that accepts a Hyperopt `trial` and returns a value.
    step_size : int, optional
        Step size between CV folds.
    eval_num : int, optional
        Number of Hyperopt trials. Default 100.
    candidate_exog : List[str], optional
        List of exogenous feature names to consider for feature importance-based selection. If None, no feature selection is performed.
    pareto_bounds : Union[float, Tuple[float, float]], optional
        If a float is provided, it is used as a fixed cutoff for cumulative importance (e.g., 0.8 means keep features that explain 80% of variance). If a tuple is provided, it defines the lower and upper bounds for tuning the Pareto cutoff. Default is (0.5, 0.999), meaning the cutoff will be tuned between 50% and 99.9% of cumulative importance.
    eval_horizons : Union[None, int], optional
        If an integer is provided, the evaluation metric will be calculated starting from that horizon onward (e.g., if 5 is passed, the metric will be calculated on horizons 5, 6, 7, etc.). If None, the metric will be calculated on all horizons.
    verbose : bool, optional
        Print score for every trial. Default False.
 
    Returns
    -------
    Tuple[Dict[str, Any], Any, Dict[str, Any], List[str]]
        Best hyperparameters and best lags (if 'lags' is in param_space).
    """

    try:
        from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval, hp
    except ImportError:
        raise ImportError("hyperopt is required. Install with: pip install hyperopt or pip install peshbeen[tuning]")
    
    # 1. SETUP DIMENSIONS
    if model.get_name() == "ml_direct_forecaster":
        test_size = max(model.H)
        eval_indices = [h - 1 for h in model.H]
    else:
        test_size = test_size

    def index_target(target_array):
        return target_array[eval_indices] if model.get_name() == "ml_direct_forecaster" else target_array

    target_col = model.target_col
    mod_name = model.model.__class__.__name__ if hasattr(model, "model") else model.get_name()

    _skip = {"box_cox", "lags", "box_cox_biasadj", "pareto_cutoff", "lag_transform"}
    if candidate_exog is not None:        
        _skip = _skip.union({f"feat_{col}" for col in candidate_exog})
    
    tscv = SplitTimeSeries(n_splits=cv_split, test_size=test_size, step_size=step_size)
    total_len = len(df)
    first_end = total_len - cv_split * (step_size or test_size)

    def _apply_forecaster_params(target_model, params):
        
        # FIX: Reset cached coefficients for ms_arr so it doesn't reuse 
        # mismatched pre-fitted weights when Optuna changes the feature space!
        if target_model.get_name() == "ms_arr":
            target_model.coeffs = None
            target_model.stds = None

        if "lags" in params:
            new_lags = list(range(1, params["lags"] + 1)) if isinstance(params["lags"], int) else list(params["lags"])
            # Safely assign to the correct attribute
            if hasattr(target_model, "lags"):
                target_model.lags = new_lags
            else:
                target_model.n_lag = new_lags
                
        if 'lag_transform' in params: 
            target_model.lag_transform = params['lag_transform']
        if "box_cox" in params:
            if isinstance(params["box_cox"], bool):
                target_model.box_cox = params["box_cox"]
                target_model.lamda = None # Force estimation if boolean True
            else:
                target_model.box_cox = True
                target_model.lamda = params["box_cox"]
        if "box_cox_biasadj" in params: 
            target_model.biasadj = params["box_cox_biasadj"]
        
    permanent_cat_vars = model.cat_variables if mod_name != "ets" and model.cat_variables is not None else []

    if eval_horizons is not None:
        # it is not valid for direct forecaster to have eval_horizons greater than the max horizon, we will raise an error in that case
        if model.get_name() == "ml_direct_forecaster":
            raise ValueError(f"eval_horizons is only applicavle for recursive models, it should be None for direct forecaster. Please set eval_horizons to None.")
        if test_size <= eval_horizons:
            raise ValueError(f"eval_horizons cannot be greater than or equal to test_size ({test_size}) for recursive models, as there would be no predictions to evaluate. Please set eval_horizons to a value less than {test_size} or set eval_horizons to None.")
    # 2. DEFINE OBJECTIVE
    def objective(params):
        m = model.copy()
        
        # A. Apply Forecasting Meta-Params
        _apply_forecaster_params(m, params)

        fit_params = {k: v for k, v in params.items() if k not in _skip}

        # B. Robust Feature Selection
        active_exog = []
        if candidate_exog is not None:
            temp_ranker = m.copy()
            
            # --- SAFE PARAMETER SETTING FOR RANKER ---
            if fit_params:
                if temp_ranker.get_name() in ("arima", "ets"):
                    temp_ranker.set_params(params=fit_params)
                elif hasattr(temp_ranker.model, "set_params"):
                    temp_ranker.model.set_params(**fit_params)

            # Universal Importance (Scale if Linear)
            # --- 2. FIT RANKER ON MASKED DATA ---
            dfl = df.copy()
            if mod_name in ("LinearRegression", "Lasso", "Ridge", "ElasticNet"):
                num_cols = [col for col in dfl.columns if col not in permanent_cat_vars]
                scaler = StandardScaler()
                dfl[num_cols] = scaler.fit_transform(dfl[num_cols])
                temp_ranker.fit(dfl.iloc[:first_end])
                importances = np.abs(temp_ranker.direct_models[test_size].coef_).flatten() if model.get_name() == "ml_direct_forecaster" else np.abs(temp_ranker.model.coef_).flatten()
                
            else:
                temp_ranker.fit(dfl.iloc[:first_end])
                importances = temp_ranker.direct_models[test_size].feature_importances_ if model.get_name() == "ml_direct_forecaster" else temp_ranker.model.feature_importances_
            
            imp_dict = dict(zip(temp_ranker.X.columns, importances))
            p_threshold = params.get("pareto_cutoff", pareto_bounds if isinstance(pareto_bounds, float) else 0.99)

            all_scores_sorted = np.sort(importances)[::-1]
            cumulative_imp = np.cumsum(all_scores_sorted) / (np.sum(all_scores_sorted) + 1e-8)
            cutoff_idx = min(np.searchsorted(cumulative_imp, p_threshold), len(all_scores_sorted) - 1)
            importance_cutoff = all_scores_sorted[cutoff_idx]

            # candidate_scores = []
            for feat in candidate_exog:
                score = sum(v for k, v in imp_dict.items() if k == feat)
                # candidate_scores.append(score)
                if score >= importance_cutoff and score > 0:
                    active_exog.append(feat)

            if not active_exog:
                # Fallback to best from the masked set
                active_exog = [candidate_exog[np.argmax([imp_dict.get(col, 0) for col in candidate_exog])]] # if candidate_exog else []

            df_trial = df[[target_col] + permanent_cat_vars + active_exog]
        else:
            df_trial = df.copy()
        
        # --- SAFE PARAMETER SETTING FOR CV MODEL ---
        if fit_params:
            if m.get_name() in ("arima", "ets"):
                m.set_params(params=fit_params)
            elif m.get_name() == "glm" or m.get_name() == "ms_arr":
                pass
            elif hasattr(m.model, "set_params"):
                filtered_final = {k: v for k, v in fit_params.items() if k not in _skip}
                m.model.set_params(**filtered_final)

        # CV Loop
        cv_scores = []
        for train_idx, test_idx in tscv.split(df_trial):
            train_fold, test_fold = df_trial.iloc[train_idx], df_trial.iloc[test_idx]
            y_true = index_target(np.array(test_fold[target_col]))
            exog_t = test_fold.drop(columns=[target_col]) if test_fold.shape[1] > 1 else None

            fold_model = m.copy()
            fold_model.fit(train_fold)
            y_pred = fold_model.forecast(H=len(y_true), exog=exog_t)

            if eval_horizons is not None:
                y_true = y_true[eval_horizons - 1:] # -1 because of zero indexing. If start from horizon 5, we want to include the 5th horizon which is at index 4
                y_pred = y_pred[eval_horizons - 1:]

            if eval_metric.__name__ in ("MASE", "SMAE", "SRMSE", "RMSSE"):
                score = eval_metric(y_true, y_pred, train_fold[target_col])
            else:
                score = eval_metric(y_true, y_pred)
            cv_scores.append(score)

        return {"loss": np.mean(cv_scores), "status": STATUS_OK}

    # 3. RUN HYPEROPT
    full_space = param_space.copy()
    
    if isinstance(pareto_bounds, tuple):
        full_space["pareto_cutoff"] = hp.uniform("pareto_cutoff", pareto_bounds[0], pareto_bounds[1])

    trials = Trials()
    best_raw = fmin(fn=objective, space=full_space, algo=tpe.suggest, max_evals=eval_num, trials=trials, verbose=verbose)

    # 4. EXTRACT BEST RESULTS
    best_results = space_eval(full_space, best_raw)
    all_best = best_results.copy()
    
    best_p_threshold = all_best.pop("pareto_cutoff", pareto_bounds if isinstance(pareto_bounds, float) else 0.99)
    best_lags = all_best.pop("lags", None)
    other_args = {k: all_best.pop(k) for k in ["box_cox", "box_cox_biasadj", "lag_transform"] if k in all_best}
    # Clean model hparams of binary mask keys
    best_hparams = {k: v for k, v in all_best.items() if not k.startswith("feat_")}

    # --- 5. FINAL FEATURE EXTRACTION ---
    best_features = []
    if candidate_exog is not None:

        final_ranker = model.copy()
        
        if best_hparams:
            if final_ranker.get_name() in ("arima", "ets"):
                final_ranker.set_params(params=best_hparams)
            elif hasattr(final_ranker.model, "set_params"):
                filtered_hparams = {k: v for k, v in best_hparams.items() if k not in _skip}
                final_ranker.model.set_params(**filtered_hparams)

        if best_lags:
            final_ranker.n_lag = list(range(1, best_lags + 1)) if isinstance(best_lags, int) else list(best_lags)
        
        if "box_cox" in other_args:
            final_ranker.box_cox = other_args["box_cox"]
            if isinstance(other_args["box_cox"], bool):
                final_ranker.lamda = None
            else:
                final_ranker.lamda = other_args["box_cox"]
        if "box_cox_biasadj" in other_args:
            final_ranker.biasadj = other_args["box_cox_biasadj"]
        final_ranker.lag_transform = other_args.get("lag_transform", final_ranker.lag_transform)
        # FIT FINAL RANKER ON MASKED DATA
        dfl_final = df.copy()
        if mod_name in ("LinearRegression", "Lasso", "Ridge", "ElasticNet"):
            num_cols = [col for col in dfl_final.columns if col not in permanent_cat_vars]
            scaler = StandardScaler()
            dfl_final[num_cols] = scaler.fit_transform(dfl_final[num_cols])
            final_ranker.fit(dfl_final.iloc[:first_end])
            importances = np.abs(final_ranker.direct_models[test_size].coef_).flatten() if model.get_name() == "ml_direct_forecaster" else np.abs(final_ranker.model.coef_).flatten()
        else:
            final_ranker.fit(dfl_final.iloc[:first_end])
            importances = final_ranker.direct_models[test_size].feature_importances_ if model.get_name() == "ml_direct_forecaster" else final_ranker.model.feature_importances_

        imp_dict = dict(zip(final_ranker.X.columns, importances))
        all_sorted = np.sort(importances)[::-1]
        cumulative = np.cumsum(all_sorted) / (np.sum(all_sorted) + 1e-8)
        cutoff_val = all_sorted[min(np.searchsorted(cumulative, best_p_threshold), len(all_sorted)-1)]

        cand_scores = []
        for feat in candidate_exog:
            score = sum(v for k, v in imp_dict.items() if k == feat)
            cand_scores.append(score)
            if score >= cutoff_val and score > 0:
                best_features.append(feat)
        
        if not best_features:
            best_features = [candidate_exog[np.argmax(cand_scores)]] if candidate_exog else []
            
    if "lag_transform" in other_args and other_args["lag_transform"] is not None:
        other_args["lag_transform"] = [tr.get_name() for tr in other_args["lag_transform"]]

    trials_data = []
    for t in trials.trials:
        # Extract trial parameters safely since Hyperopt stores them as lists
        raw_vals = {k: v[0] for k, v in t['misc']['vals'].items() if len(v) > 0}
        try:
            t_params = space_eval(full_space, raw_vals)
        except Exception:
            t_params = {}
            
        t_other = {k: v for k, v in t_params.items() if k in _skip and k != "lags" and not k.startswith("feat_")}
        if "lag_transform" in t_other and t_other["lag_transform"] is not None:
            t_other["lag_transform"] = [tr.get_name() if hasattr(tr, "get_name") else str(tr) for tr in t_other["lag_transform"]]
            
        trials_data.append({
            "trial_number": t['tid'],
            "score": t['result'].get('loss'),
            "state": t['result'].get('status'),
            "model_params": {k: v for k, v in t_params.items() if k not in _skip},
            "lags": t_params.get("lags", None),
            "other_params": t_other
        })
    model.trials_df = pd.DataFrame(trials_data).sort_values("score").reset_index(drop=True)
    return best_hparams, best_lags, other_args, best_features
```
:::


::: {#c7eb8f02 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
from peshbeen.datasets import load_wales_admissions
from peshbeen.models import ml_forecaster
from peshbeen.metrics import MAE, RMSE
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore")
from lightgbm import LGBMRegressor
wales_admissions = load_wales_admissions()
wales_admissions["day_of_week"] = wales_admissions.index.dayofweek
wales_admissions["month"] = wales_admissions.index.month
# split the data into train and test sets
train = wales_admissions[:-30]
test = wales_admissions[-30:]
cat_variables = ["day_of_week", "month"]
# import linear regression from sklearn
from sklearn.linear_model import LinearRegression
ml_linear = ml_forecaster(model=LGBMRegressor(verbose=-1),
              target_col='admissions', lags = 30,
              cat_variables=cat_variables, categorical_encoder=ohe)
ml_linear.fit(train)
# ml_linear.data_prep(train)
forecasts = ml_linear.forecast(H=30, exog=test[cat_variables])
```
:::


::: {#bc9efb65 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
from hyperopt import hp
from hyperopt.pyll import scope
lgb_param_space={'learning_rate': hp.uniform('learning_rate', 0.001, 0.6),
            'num_leaves': scope.int(hp.quniform('num_leaves', 10, 200, 1)),
           'max_depth':scope.int(hp.quniform('max_depth', 2, 18, 1)),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
           'min_data_in_leaf': scope.int(hp.quniform ('min_data_in_leaf', 5, 100, 1)), 
            'lambda_l2' : hp.uniform('lambda_l2', 0,10),
           'lambda_l1' : hp.uniform('lambda_l1', 0, 10),
            'min_gain_to_split':hp.uniform('min_gain_to_split', 0, 20),
            "max_bin": scope.int(hp.quniform('max_bin', 100, 350, 1)),
           'top_rate' : hp.quniform('top_rate', 0.05, 0.4, 0.0001),
            'other_rate' : hp.quniform('other_rate', 0.05, 0.3, 0.0001),
           'num_iterations': scope.int(hp.quniform("num_iterations", 30, 700, 1)),
           'top_k': scope.int(hp.quniform('top_k', 8, 30, 1)),
           'lags': hp.choice("lags", [
                                 [1,2,3,4,5],
                                 [1,4,7],
                                 [1,2,3,4,5,6,7],
                                 [1,2,3,4,5,6,7,14],
                                 [1,2,3,4,5,6,7,14,21],
                                 [1,2,3],
                             ]),
                "seed":0,
                "box_cox": hp.uniform("box_cox", 0.0, 4),
                "box_cox_biasadj": hp.choice("box_cox_biasadj", [True, False])}
best_params, lags_, other_, _ = hyperopt_tune(model=ml_linear, df=train, cv_split=5, step_size=10,
                                        test_size=5, eval_metric=RMSE, eval_num=5,
                                        param_space=lgb_param_space, verbose=True, eval_horizons=3)
```

::: {.cell-output .cell-output-stdout}
```
100%|██████████| 5/5 [00:15<00:00,  3.05s/trial, best loss: 90.98887966648314] 
```
:::
:::


::: {#4910f12c .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
ets_param_space = {
    "smoothing_level":     hp.uniform("smoothing_level", 0.001, 0.99),
    "trend":              hp.choice("trend", ["add", "mul", None]),
    "seasonal":           hp.choice("seasonal", ["add", "mul", None]),
    "smoothing_trend":    hp.uniform("smoothing_trend", 0.001, 0.99),
    "smoothing_seasonal": hp.uniform("smoothing_seasonal", 0.001, 0.99),
    "smoothing_level":     hp.uniform("smoothing_level", 0.001, 0.99),
    "seasonal_periods":   7,
    "damped_trend": hp.choice("damped_trend", [True, False]),
    "damping_trend": hp.uniform("damping_trend", 0, 0.99),
    "box_cox": hp.uniform("box_cox", 0.0, 4),
    "box_cox_biasadj": hp.choice("box_cox_biasadj", [True, False])
}

from peshbeen.models import ets
ets_model = ets(target_col='admissions')
best_params_ets, _, other_, _ = hyperopt_tune(model=ets_model, df=train, cv_split=5, step_size=10,
                                        test_size=1, eval_metric=RMSE, eval_num=10,
                                        param_space=ets_param_space)
```
:::


::: {#42ad67aa .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
from optuna import study
def optuna_tune(
    model: Any,
    df: pd.DataFrame,
    cv_split: int,
    test_size: int,
    eval_metric: Callable,
    param_space: Dict[str, Any],
    step_size: int = None,
    eval_num: int = 100,
    multivariate: bool = False,
    warm_up_steps: int = 10,
    startup_trials: int = 5,
    candidate_exog: List[str] = None,
    pareto_bounds: Union[float, Tuple[float, float]] = (0.5, 0.999),
    eval_horizons: Union[None, int] = None,
    verbose: bool = False,
    random_state: Optional[int] = 42,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any], List[str]]:

    """
    Tune forecasting model hyperparameters using time series cross-validation and Optuna.
 
    Parameters
    ----------
    model : object
        Forecasting model with .fit and .forecast methods.
    df : pd.DataFrame
        Time series data (datetime index, target column, optional exogenous features).
    cv_split : int
        Number of cross-validation splits.
    test_size : int
        Number of samples in each test fold. For ml_direct_forecaster, this will be overridden to be the maximum horizon in model.H.
    eval_metric : Callable
        Metric function to minimise.
    param_space : dict
        Each value must be a callable that accepts an Optuna `trial` and returns a value.
    step_size : int, optional
        Step size between CV folds.
    eval_num : int, optional
        Number of Optuna trials. Default 100.
    multivariate : bool, optional
        If True, uses TPE (Tree-structured Parzen Estimator) sampler for multivariate hyperparameter optimization. If False, uses default Independent sampler.
    warm_up_steps : int, optional
        The number of initial CV folds to complete before the Optuna pruner begins evaluating trial performance. This "warm-up" period prevents the pruner from prematurely terminating promising trials due to high variance or noise present in the earliest (smallest) cross-validation folds. Default 10.
    startup_trials : int, optional
        The number of complete trials (hyperparameter combinations) to evaluate before enabling the pruner. This ensures the pruner has a solid baseline median to compare against. Default 5.
    candidate_exog : List[str], optional
        List of exogenous feature names to consider for feature importance-based selection. If None, no feature selection is performed.
    pareto_bounds : Union[float, Tuple[float, float]], optional
        If a float is provided, it is used as a fixed cutoff for cumulative importance (e.g., 0.8 means keep features that explain 80% of variance). If a tuple is provided, it defines the lower and upper bounds for tuning the Pareto cutoff. Default is (0.5, 0.999), meaning the cutoff will be tuned between 50% and 99.9% of cumulative importance.
    eval_horizons : Union[None, int], optional
        If an integer is provided, the evaluation metric will be calculated starting from that horizon onward (e.g., if 5 is passed, the metric will be calculated on horizons 5, 6, 7, etc.). If None, the metric will be calculated on all horizons.
    verbose : bool, optional
        Print score for every trial. Default False.
 
    Returns
    -------
    Tuple[Dict[str, Any], Any]
        Best hyperparameters and best lags (if 'lags' is in param_space).
    """

    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required.")
    
    if model.get_name() == "ml_direct_forecaster":
        test_size = max(model.H) 
        eval_indices = [h - 1 for h in model.H]
    else:
        test_size = test_size

    def index_target(target_array):
        if model.get_name() == "ml_direct_forecaster":
            return target_array[eval_indices]
        else:
            return target_array

    target_col = model.target_col
    mod_name = model.model.__class__.__name__ if hasattr(model, "model") else model.get_name()

    # FIXED: Added all ml_forecaster native parameters so they are not passed to base models
    _forecaster_keys = [
        "box_cox", "lags", "box_cox_biasadj", "lag_transform"
    ]
    _skip = set(_forecaster_keys).union({"pareto_cutoff"})

    def _fit_params(params: dict) -> Optional[dict]:
        base_model = getattr(model, "model", None) 
        is_lr = isinstance(base_model, LinearRegression) 

        if is_lr:
            return {}  
        return {k: v for k, v in params.items() if k not in _skip}
        
    # NEW: Helper function to consistently apply ml_forecaster specific configurations safely
    def _apply_forecaster_params(target_model, params):
        
        # FIX: Reset cached coefficients for ms_arr so it doesn't reuse 
        # mismatched pre-fitted weights when Optuna changes the feature space!
        if target_model.get_name() == "ms_arr":
            target_model.coeffs = None
            target_model.stds = None

        if "lags" in params:
            new_lags = list(range(1, params["lags"] + 1)) if isinstance(params["lags"], int) else list(params["lags"])
            # Safely assign to the correct attribute
            if hasattr(target_model, "lags"):
                target_model.lags = new_lags
            else:
                target_model.n_lag = new_lags
                
        if 'lag_transform' in params: 
            target_model.lag_transform = params['lag_transform']
        if "box_cox" in params:
            if isinstance(params["box_cox"], bool):
                target_model.box_cox = params["box_cox"]
                target_model.lamda = None # Force estimation if boolean True
            else:
                target_model.box_cox = True
                target_model.lamda = params["box_cox"]
        if "box_cox_biasadj" in params: 
            target_model.biasadj = params["box_cox_biasadj"]
    
    tscv = SplitTimeSeries(n_splits=cv_split, test_size=test_size, step_size=step_size)
    total_len = len(df) 
    first_end = total_len - cv_split * (step_size or test_size) 
    permanent_cat_vars = model.cat_variables if mod_name != "ets" and model.cat_variables is not None else []
    
    if eval_horizons is not None:
        if model.get_name() == "ml_direct_forecaster":
            raise ValueError(f"eval_horizons is only applicable for recursive models, it should be None for direct forecaster. Please set eval_horizons to None.")

        if test_size <= eval_horizons:
            raise ValueError(f"eval_horizons cannot be greater than or equal to test_size ({test_size}) for recursive models, as there would be no predictions to evaluate. Please set eval_horizons to a value less than {test_size} or set eval_horizons to None.")
        
    def objective(trial: optuna.Trial) -> float:
        params = {name: suggest_fn(trial) for name, suggest_fn in param_space.items()}
        fit_params = _fit_params(params)
    
        active_exog = []
        if candidate_exog is not None and len(candidate_exog) > 0:
            temp_ranker = model.copy()
            
            if temp_ranker.get_name() in ("arima", "ets"):
                temp_ranker.set_params(params=fit_params)
            elif hasattr(temp_ranker, "model") and hasattr(temp_ranker.model, "set_params"):
                temp_ranker.model.set_params(**fit_params)
                
            _apply_forecaster_params(temp_ranker, params)
        
            dfl = df.copy()
            
            if mod_name in ("LinearRegression", "Lasso", "Ridge", "ElasticNet"):
                num_cols = [col for col in dfl.columns if col not in permanent_cat_vars]
                scaler = StandardScaler()
                dfl[num_cols] = scaler.fit_transform(dfl[num_cols])
                temp_ranker.fit(dfl.iloc[:first_end])
                importances = np.abs(temp_ranker.direct_models[test_size].coef_).flatten() if model.get_name() == "ml_direct_forecaster" else np.abs(temp_ranker.model.coef_).flatten()
            else:
                temp_ranker.fit(dfl.iloc[:first_end])
                importances = temp_ranker.direct_models[test_size].feature_importances_ if model.get_name() == "ml_direct_forecaster" else temp_ranker.model.feature_importances_
            
            imp_dict = dict(zip(temp_ranker.X.columns, importances))

            p_threshold = trial.suggest_float("pareto_cutoff", pareto_bounds[0], pareto_bounds[1]) if isinstance(pareto_bounds, tuple) else pareto_bounds

            all_scores_sorted = np.sort(importances)[::-1]
            cumulative_imp = np.cumsum(all_scores_sorted) / (np.sum(all_scores_sorted) + 1e-8)
            cutoff_idx = min(np.searchsorted(cumulative_imp, p_threshold), len(all_scores_sorted) - 1)
            importance_cutoff = all_scores_sorted[cutoff_idx]

            candidate_scores = []
            for feat in candidate_exog:
                score = sum(v for k, v in imp_dict.items() if k == feat)
                candidate_scores.append(score)
                if score >= importance_cutoff and score > 0:
                    active_exog.append(feat)

            if not active_exog:
                active_exog = [candidate_exog[np.argmax(candidate_scores)]]

            cols_to_use = [target_col] + permanent_cat_vars + active_exog
            df_ = df[cols_to_use]
        else:
            df_ = df.copy()

        cv_scores = []

        model_ = model.copy()
        _apply_forecaster_params(model_, params)
        for step_idx, (train_idx, test_idx) in enumerate(tscv.split(df_)):
            train = df_.iloc[train_idx]
            test = df_.iloc[test_idx]
            
            y_true = test[target_col].values
            y_true = index_target(y_true) 
            exog_t = test.drop(columns=[target_col]) if test.shape[1] > 1 else None

            
            
        
            if model_.get_name() in ("arima", "ets"):
                model_.set_params(params=fit_params)
            elif (model_.get_name() == "glm") or (model_.get_name() == "ms_arr"):
                ## do not do any parameter setting for glm since the family is set at initialization and can't be changed, and the other parameters are ml_forecaster specific and will be set in the _apply_forecaster_params function
                pass
            else:
                model_.model.set_params(**fit_params)
            model_.fit(train)
            
            y_pred = model_.forecast(H=len(test), exog=exog_t)

            if eval_horizons is not None:
                y_true = y_true[eval_horizons - 1:] 
                y_pred = y_pred[eval_horizons - 1:]

            if eval_metric.__name__ in ("MASE", "SMAE", "SRMSE", "RMSSE"):
                score = eval_metric(y_true, y_pred, train[target_col])
            else:
                score = eval_metric(y_true, y_pred)
                
            cv_scores.append(score)
        
            trial.report(np.mean(cv_scores), step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=startup_trials, n_warmup_steps=warm_up_steps),
        sampler=optuna.samplers.TPESampler(n_startup_trials=startup_trials, multivariate=multivariate, seed=random_state)
    )

    if verbose:
        def optuna_print_callback(study, trial):
            if trial.state == optuna.trial.TrialState.PRUNED:
                print(f"Trial {trial.number:3} | Status: PRUNED | Pruned at step: {list(trial.intermediate_values.keys())[-1] if trial.intermediate_values else 'N/A'}")
            else:
                print(f"Trial {trial.number:3} | Status: {trial.state.name:8} | Value: {trial.value:.6f} | Best trial: {study.best_trial.number:3} | Best Value: {study.best_value:.6f}")
        study.optimize(objective, n_trials=eval_num, callbacks=[optuna_print_callback])
    else:
        study.optimize(objective, n_trials=eval_num)

    best_results = study.best_params.copy()
    best_p_threshold = best_results.pop("pareto_cutoff", pareto_bounds if not isinstance(pareto_bounds, tuple) else 0.99)
    best_lags = best_results.pop("lags", None)

    # FIXED: Now securely captures ALL wrapper properties so they aren't incorrectly mapped to the base model
    other_args = {k: best_results.pop(k) for k in _forecaster_keys if k in best_results}
    best_hparams = best_results 

    best_features = []
    if candidate_exog is not None and len(candidate_exog) > 0:
        final_ranker = model.copy()
        
        if final_ranker.get_name() in ("arima", "ets"):
            final_ranker.set_params(params={k: v for k, v in best_hparams.items() if k not in _skip})
        elif hasattr(final_ranker, "model") and hasattr(final_ranker.model, "set_params"):
            final_ranker.model.set_params(**{k: v for k, v in best_hparams.items() if k not in _skip})

        # Process the wrapper configurations identical to the objective function
        _apply_forecaster_params(final_ranker, study.best_params)

        dfl_final = df.copy()
        num_cols = [col for col in dfl_final.columns if col not in permanent_cat_vars]
        
        if mod_name in ("LinearRegression", "Lasso", "Ridge", "ElasticNet"):
            scaler = StandardScaler()
            dfl_final[num_cols] = scaler.fit_transform(dfl_final[num_cols])
            final_ranker.fit(dfl_final.iloc[:first_end])
            importances = np.abs(final_ranker.direct_models[test_size].coef_).flatten() if model.get_name() == "ml_direct_forecaster" else np.abs(final_ranker.model.coef_).flatten()
        else:
            final_ranker.fit(dfl_final.iloc[:first_end])   
            importances = final_ranker.direct_models[test_size].feature_importances_ if model.get_name() == "ml_direct_forecaster" else final_ranker.model.feature_importances_
            
        imp_dict = dict(zip(final_ranker.X.columns, importances))
        
        all_scores_sorted = np.sort(importances)[::-1]
        cumulative_imp = np.cumsum(all_scores_sorted) / (np.sum(all_scores_sorted) + 1e-8)
        cutoff_idx = min(np.searchsorted(cumulative_imp, best_p_threshold), len(all_scores_sorted) - 1)
        importance_cutoff = all_scores_sorted[cutoff_idx]

        candidate_scores = []
        for feat in candidate_exog:
            score = sum(v for k, v in imp_dict.items() if k == feat)
            candidate_scores.append(score)
            if score >= importance_cutoff and score > 0:
                best_features.append(feat)
        
        if not best_features:
            best_features = [candidate_exog[np.argmax(candidate_scores)]]

    if "lag_transform" in other_args and other_args["lag_transform"] is not None:
        other_args["lag_transform"] = [tr.get_name() for tr in other_args["lag_transform"]]
    trials_data = []
    for t in study.trials:
        t_params = t.params or {}
        trials_data.append({
            "trial_number": t.number,
            "score": t.value,
            "state": t.state.name,
            "model_params": {k: v for k, v in t_params.items() if k not in _skip},
            "lags": t_params.get("lags", None),
            "other_params": {k: v for k, v in t_params.items() if k in _skip and k != "lags"}
        })
    model.trials_df = pd.DataFrame(trials_data).sort_values("score").reset_index(drop=True)
    return best_hparams, best_lags, other_args, best_features
```
:::


::: {#941fe9e1 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
lgb_param_space = {
    "learning_rate":     lambda t: t.suggest_float("learning_rate", 0.001, 0.6),
    "num_leaves":        lambda t: t.suggest_int("num_leaves", 10, 200),
    "max_depth":         lambda t: t.suggest_int("max_depth", 2, 18),
    "bagging_fraction":  lambda t: t.suggest_float("bagging_fraction", 0.5, 1.0),
    "feature_fraction":  lambda t: t.suggest_float("feature_fraction", 0.5, 1.0),
    "min_data_in_leaf":  lambda t: t.suggest_int("min_data_in_leaf", 5, 100),
    "lambda_l2":         lambda t: t.suggest_float("lambda_l2", 0.0, 10.0),
    "lambda_l1":         lambda t: t.suggest_float("lambda_l1", 0.0, 10.0),
    "min_gain_to_split": lambda t: t.suggest_float("min_gain_to_split", 0.0, 20.0),
    "max_bin":           lambda t: t.suggest_int("max_bin", 100, 350),
    "top_rate":          lambda t: t.suggest_float("top_rate", 0.05, 0.4),
    "other_rate":        lambda t: t.suggest_float("other_rate", 0.05, 0.3),
    "num_iterations":    lambda t: t.suggest_int("num_iterations", 30, 700),
    "top_k":             lambda t: t.suggest_int("top_k", 8, 30),
    # "box_cox":          lambda t: t.suggest_float("box_cox", 0.0, 4),  # Example of a float parameter for box_cox
    "lags":              lambda t: t.suggest_categorical(
                             "lags", [
                                 [1,2],[3,4],[5],7, [1, 2,3]
                             ]),
    # "box_cox":          lambda t: t.suggest_float("box_cox", 0.0, 4),
    "seed":              lambda t: t.suggest_int("seed", 0, 0)  # Fixed seed for reproducibility
}

ml_linear = ml_forecaster(model=LGBMRegressor(verbose=-1),
              target_col='admissions', lags = 30,
              cat_variables=cat_variables, categorical_encoder=ohe)

best_params, best_lags, other_ags, _ = optuna_tune(
    model=ml_linear,
    df=train,
    cv_split=10,
    step_size=10,
    test_size=30,
    eval_metric=RMSE,
    eval_num=5,
    warm_up_steps=2,
    param_space=lgb_param_space, verbose=True, eval_horizons=None
)
```

::: {.cell-output .cell-output-stdout}
```
Trial   0 | Status: COMPLETE | Value: 119.424682 | Best trial:   0 | Best Value: 119.424682
Trial   1 | Status: COMPLETE | Value: 126.650811 | Best trial:   0 | Best Value: 119.424682
Trial   2 | Status: COMPLETE | Value: 189.124869 | Best trial:   0 | Best Value: 119.424682
Trial   3 | Status: COMPLETE | Value: 177.404764 | Best trial:   0 | Best Value: 119.424682
Trial   4 | Status: COMPLETE | Value: 150.966182 | Best trial:   0 | Best Value: 119.424682
```
:::
:::


::: {#5e496998 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
ml_linear = ml_forecaster(model=LGBMRegressor(verbose=-1, **best_params),
              target_col='admissions', lags = best_lags,
              cat_variables=cat_variables, categorical_encoder=ohe)
ml_linear.cross_validate(train, cv_split=10,
                         step_size=10, test_size=30,
                         metrics=[RMSE])
ml_linear.cv_summary
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
      <th>eval_metric</th>
      <th>overall_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RMSE</td>
      <td>119.424682</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::


::: {#33182cfd .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
ets_param_space = {
    "smoothing_level":     lambda t: t.suggest_float("smoothing_level", 0.001, 0.99),
    "trend":              lambda t: t.suggest_categorical(
                             "trend", [
                                 "add",
                                 "mul",
                                 None
                             ]),
    "seasonal":           lambda t: t.suggest_categorical(
                             "seasonal", [
                                 "add",
                                 "mul",
                                 None
                             ]),
    "smoothing_trend":    lambda t: t.suggest_float("smoothing_trend", 0.001, 0.99),
    "smoothing_seasonal": lambda t: t.suggest_float("smoothing_seasonal", 0.001, 0.99),
    "smoothing_level":     lambda t: t.suggest_float("smoothing_level", 0.001, 0.99),
    "seasonal_periods":              lambda t: 7,   # fixed, not sampled
}

p_values = [0, 1, 2, 3, 4]
d_values = [0, 1]
q_values = [0, 1, 2, 3, 4]
from itertools import product
orders = list(product(p_values, d_values, q_values))

P_values = [0, 1, 2, 3, 4]
D_values = [0, 1]
Q_values = [0, 1, 2, 3, 4]
seasonal_orders = list(product(P_values, D_values, Q_values))

arima_param_space = {
    "order": lambda t: t.suggest_categorical(
        "order", orders
    ),
    "seasonal_order": lambda t: t.suggest_categorical(
        "seasonal_order", seasonal_orders
    ),
    "seasonal_length": lambda t: 7,   # fixed, not sampled
}


from peshbeen.models import ets, arima
ets_model = ets(target_col='admissions')
arima_model = arima(target_col='admissions')
best_params, _, other_, _ = optuna_tune(
    model=arima_model,
    df=train,
    cv_split=10,
    step_size=10,
    test_size=30,
    eval_metric=RMSE,
    eval_num=3,
    param_space=arima_param_space, verbose=False
)
```
:::


## Hyperparameters tuning methods for Multivariate machine learning models

::: {#d2e7cb3d .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def mv_hyperopt_tune(
    model: object,
    df: pd.DataFrame,
    target_col: str,
    cv_split: int,
    test_size: int,
    eval_metric: Callable,
    param_space: dict,
    step_size: int = None,
    eval_num=100,
    verbose=False
    ) -> Tuple[Dict[str, Any], List[int], List[str]]:

    """
    Tune forecasting model hyperparameters using time series cross-validation and hyperopt for multivariate models.

    Parameters
    ----------
    model : object
        Forecasting model object with .fit and .forecast methods and relevant attributes.
    df : pd.DataFrame
        Time series data with a datetime index and a target column and optionally exogenous features.
    target_col : str
        Name of the target column to minimize the evaluation metric on.
    cv_split : int
        Number of cross-validation splits.
    test_size : int
        Number of samples in each test set.
    eval_metric : Callable
        Evaluation metric function.
    step_size : int, optional
        Step size to move the test window forward in each split.
    param_space : dict
        Hyperparameter search space for the forecasting model.
    eval_num : int, optional
        Number of hyperparameter combinations to evaluate. Default is 100.
    verbose : bool, optional
        Whether to print the evaluation metric for each hyperparameter combination. Default is False.
    
    Returns
    -------
    Tuple[Dict[str, Any], List[int], List[str]]
        A tuple containing the best hyperparameters, selected lags, and selected transforms.
    """

    try:
        from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
    except ImportError:
        raise ImportError("hyperopt is required. Install with: pip install hyperopt or pip install peshbeen[tuning]")
    
    tscv = SplitTimeSeries(n_splits=cv_split, test_size=test_size, step_size=step_size)

    _skip = {"lags"}
 
    def _set_model_params(params: dict):
        if "lags" in params:
            lags =  list(range(1, params["lags"] + 1)) if isinstance(params["lags"], int) else list(params["lags"])
            model.n_lag = {trgt: lags for trgt in model.target_cols}
 
    def _fit_params(params: dict) -> Optional[dict]:
        base_model = getattr(model, "model", None) # Check if the model has a 'model' attribute (like ARIMA or ETS), otherwise use the model itself (like LinearRegression)
        is_lr = isinstance(base_model, LinearRegression) # Determine if the base model is LinearRegression, which does not require hyperparameter tuning

        if is_lr:
            return {}  # Return an empty dict for LinearRegression, as it does not have hyperparameters to tune
        return {k: v for k, v in params.items() if k not in _skip}

    def objective(params):

        metrics = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=model.target_cols)
            y_test = np.array(test[target_col])

            _set_model_params(params)
            fit_params = _fit_params(params)
            model.model.set_params(**fit_params)
            model.fit(train)
            
            exog_test = x_test if x_test.shape[1] > 0 else None
            y_pred = model.forecast(len(y_test), exog_test)[target_col]

            #Evaluate using the specified metric
            if eval_metric.__name__ in ["MASE", "SMAE", "SRMSE", "RMSSE"]:
                score = eval_metric(y_test,
                                    y_pred,
                                    train[target_col])
            else:
                score = eval_metric(y_test,y_pred)
            metrics.append(score)

        mean_score = np.mean(metrics)
        if verbose:
            print("Score:", mean_score)
        return {"loss": mean_score, "status": STATUS_OK}

    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=eval_num,
        trials=trials,
    )

    # if lags are in the param space, extract the best lags and extract remain parameters for the model
    if "lags" in param_space:
        if not isinstance(space_eval(param_space, best_hyperparams)["lags"], int):
            best_lags = list(space_eval(param_space, best_hyperparams)["lags"])
        else:
            best_lags = space_eval(param_space, best_hyperparams)["lags"]
        best_lags = {trgt: best_lags for trgt in model.target_cols}
        model_parameters = {k: v for k, v in space_eval(param_space, best_hyperparams).items() if k != "lags"}
    else:
        best_lags = None
        model_parameters = space_eval(param_space, best_hyperparams)

    return model_parameters, best_lags
```
:::


::: {#312cbc66 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
## get day of week and month as features from the date index
from peshbeen.datasets import load_admission_calls
from peshbeen.models import ml_mv_forecaster
from peshbeen.metrics import RMSE, MAE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope
from lightgbm import LGBMRegressor
admission_calls = load_admission_calls()
admission_calls["day_of_week"] = admission_calls.index.dayofweek
admission_calls["month"] = admission_calls.index.month
train = admission_calls[:-30]
test = admission_calls[-30:]

cat_variables = ["day_of_week", "month"]
ml_linear = ml_mv_forecaster(model=LGBMRegressor(verbose=-1),
              target_cols=['admissions', "calls"], lags = {"admissions": 7, "calls": 7},
                cat_variables=cat_variables, categorical_encoder=ohe,
                difference={"admissions": 1, "calls": 1},
              box_cox={"admissions": 9, "calls": 9},)
ml_linear.fit(train)
ml_linear.predict_in_sample()
forecasts = ml_linear.forecast(H=30, exog=test[cat_variables])
lgb_param_space={'learning_rate': hp.uniform('learning_rate', 0.001, 0.6),
            'num_leaves': scope.int(hp.quniform('num_leaves', 10, 200, 1)),
           'max_depth':scope.int(hp.quniform('max_depth', 2, 18, 1)),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
           'min_data_in_leaf': scope.int(hp.quniform ('min_data_in_leaf', 5, 100, 1)), 
            'lambda_l2' : hp.uniform('lambda_l2', 0,10),
           'lambda_l1' : hp.uniform('lambda_l1', 0, 10),
            'min_gain_to_split':hp.uniform('min_gain_to_split', 0, 20),
            "max_bin": scope.int(hp.quniform('max_bin', 100, 350, 1)),
           'top_rate' : hp.quniform('top_rate', 0.05, 0.4, 0.0001),
            'other_rate' : hp.quniform('other_rate', 0.05, 0.3, 0.0001),
           'num_iterations': scope.int(hp.quniform("num_iterations", 30, 700, 1)),
           'top_k': scope.int(hp.quniform('top_k', 8, 30, 1)),
                "seed":0, 'lags': hp.choice("lags", [
                                 [1,2,3,4,5],
                                 [1,4,7],
                                 [1,2,3,4,5,6,7],
                                 [1,2,3,4,5,6,7,14]])}
best_params= mv_hyperopt_tune(model=ml_linear, df=train, target_col= "admissions", cv_split=3, step_size=10,
                                        test_size=30, eval_metric=RMSE, eval_num=4,
                                        param_space=lgb_param_space)
```

::: {.cell-output .cell-output-stdout}
```
100%|██████████| 4/4 [00:07<00:00,  1.94s/trial, best loss: 183.1144350795865] 
```
:::
:::


::: {#d3e15af8 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def mv_optuna_tune(
    model: object,
    df: pd.DataFrame,
    target_col: str,
    cv_split: int,
    test_size: int,
    eval_metric: Callable,
    param_space: Dict[str, Any],
    step_size: int = None,
    eval_num: int = 100,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Any]:
    """
    Tune forecasting model hyperparameters using time series cross-validation and Optuna.
 
    Parameters
    ----------
    model : object
        Forecasting model with .fit and .forecast methods.
    df : pd.DataFrame
        Time series data (datetime index, target column, optional exogenous features).
    target_col : str
        Name of the target column to minimize the evaluation metric on.
    cv_split : int
        Number of cross-validation splits.
    test_size : int
        Number of samples in each test fold.
    eval_metric : Callable
        Metric function to minimise.
    param_space : dict
        Each value must be a callable that accepts an Optuna `trial` and returns a value.
    step_size : int, optional
        Step size between CV folds.
    eval_num : int, optional
        Number of Optuna trials. Default 100.
    verbose : bool, optional
        Print score for every trial. Default False.
 
    Returns
    -------
    Tuple[Dict[str, Any], Any]
        Best hyperparameters and best lags (if 'lags' is in param_space).

    """

    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required. Install with: pip install optuna or pip install peshbeen[tuning]")
    
    tscv = SplitTimeSeries(n_splits=cv_split, test_size=test_size, step_size=step_size)

    _skip = {"lags"}
 
    def _set_model_params(params: dict):
        if "lags" in params:
            lags =  list(range(1, params["lags"] + 1)) if isinstance(params["lags"], int) else list(params["lags"])
            model.n_lag = {trgt: lags for trgt in model.target_cols}
 
    def _fit_params(params: dict) -> Optional[dict]:
        base_model = getattr(model, "model", None) # Check if the model has a 'model' attribute (like ARIMA or ETS), otherwise use the model itself (like LinearRegression)
        is_lr = isinstance(base_model, LinearRegression) # Determine if the base model is LinearRegression, which does not require hyperparameter tuning

        if is_lr:
            return {}  # Return an empty dict for LinearRegression, as it does not have hyperparameters to tune
        return {k: v for k, v in params.items() if k not in _skip}
 
    def objective(trial: optuna.Trial) -> float:
        params = {name: suggest_fn(trial) for name, suggest_fn in param_space.items()}

        scores = []
        for train_idx, test_idx in tscv.split(df):
            train, test = df.iloc[train_idx], df.iloc[test_idx]
            x_test = test.drop(columns=model.target_cols)
            y_test = np.array(test[target_col])
 
            _set_model_params(params)
            fit_params = _fit_params(params)
            model.model.set_params(**fit_params)
            model.fit(train)

            exog_test = x_test if x_test.shape[1] > 0 else None
            y_pred = model.forecast(len(y_test), exog_test)[target_col]
 
            if eval_metric.__name__ in ("MASE", "SMAE", "SRMSE", "RMSSE"):
                score = eval_metric(y_test, y_pred, train[target_col])
            else:
                score = eval_metric(y_test, y_pred)
            scores.append(score)
 
        mean_score = float(np.mean(scores))
        if verbose:
            if trial.number > 0:
                print(f"Trial {trial.number:>4d} | score={mean_score:.6f} | {params} | best trial={trial.study.best_trial.number} | best_score={trial.study.best_value:.4f}")
            else:
                print(f"Trial {trial.number:>4d} | score={mean_score:.6f} | {params}")
        return mean_score
    
    # Optuna internal logging control
    old_verbosity = optuna.logging.get_verbosity()
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # or ERROR / CRITICAL

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=eval_num, show_progress_bar=False)
    finally:
        optuna.logging.set_verbosity(old_verbosity)

    best = study.best_params
    if "lags" in param_space:
        best_lags = best.pop("lags")
        best_lags = {trgt: best_lags for trgt in model.target_cols}
        model_parameters = best

    else:
        best_lags = None
        model_parameters = best
 
    return model_parameters, best_lags
```
:::


::: {#bfe2cd32 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
lgb_param_space = {
    "learning_rate":     lambda t: t.suggest_float("learning_rate", 0.001, 0.6),
    "num_leaves":        lambda t: t.suggest_int("num_leaves", 10, 200),
    "max_depth":         lambda t: t.suggest_int("max_depth", 2, 18),
    "bagging_fraction":  lambda t: t.suggest_float("bagging_fraction", 0.5, 1.0),
    "feature_fraction":  lambda t: t.suggest_float("feature_fraction", 0.5, 1.0),
    "min_data_in_leaf":  lambda t: t.suggest_int("min_data_in_leaf", 5, 100),
    "lambda_l2":         lambda t: t.suggest_float("lambda_l2", 0.0, 10.0),
    "lambda_l1":         lambda t: t.suggest_float("lambda_l1", 0.0, 10.0),
    "min_gain_to_split": lambda t: t.suggest_float("min_gain_to_split", 0.0, 20.0),
    "max_bin":           lambda t: t.suggest_int("max_bin", 100, 350),
    "top_rate":          lambda t: t.suggest_float("top_rate", 0.05, 0.4),
    "other_rate":        lambda t: t.suggest_float("other_rate", 0.05, 0.3),
    "num_iterations":    lambda t: t.suggest_int("num_iterations", 30, 700),
    "top_k":             lambda t: t.suggest_int("top_k", 8, 30),
    "seed":              lambda t: 0,   # fixed, not sampled
    'lags': lambda t: t.suggest_categorical(
                             "lags", [
                                 [1,4,7],
                                 [1,2,3,4,5,6,7],
                                 [1,2,3,4,5,6,7,14]
                             ]),    
}
best_params, best_lags = mv_optuna_tune(model=ml_linear, df=train, target_col= "admissions", cv_split=5, step_size=10,
                                        test_size=30, eval_metric=RMSE, eval_num=4,
                                        param_space=lgb_param_space)
```
:::


## Hyperparameter tuning for Multi-Series Interdependent Machine Learning models (ml_multi_forecaster)

::: {#optuna_tune_multi_export .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def optuna_tune_multi(
    model: Any,
    df: pd.DataFrame,
    cv_split: int,
    test_size: int,
    eval_metric: Callable,
    param_space: Dict[str, Any],
    target_series: Optional[Union[str, List[str]]] = None,
    step_size: int = 1,
    eval_num: int = 100,
    multivariate: bool = False,
    warm_up_steps: int = 10,
    startup_trials: int = 5,
    ref_series_id: Optional[str] = None,
    transform_back: bool = True,
    verbose: bool = False,
    random_state: Optional[int] = 42
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """
    Tune multi-series interdependent forecasting model (ml_multi_forecaster) hyperparameters
    using panel time-series cross-validation and Optuna.

    Parameters
    ----------
    model : object
        An instance of `ml_multi_forecaster`.
    df : pd.DataFrame
        Long-format panel DataFrame with time index, id_col, target_col, and optional exogenous features.
    cv_split : int
        Number of cross-validation splits.
    test_size : int
        Number of time steps in each forecast evaluation test set.
    eval_metric : Callable
        Metric function to minimize (e.g. MAE, RMSE, MAPE, MASE).
    param_space : dict
        Dictionary where each value is a callable taking an Optuna `trial`.
    target_series : str, list of str, optional
        Specific series ID(s) to optimize the evaluation metric for.
        - If a single string, optimizes accuracy ONLY for that series ID.
        - If a list of strings, optimizes average accuracy across those specified series IDs.
        - If None (default), optimizes average accuracy across ALL series IDs in the panel.
    step_size : int, default 1
        Step size between CV folds.
    eval_num : int, default 100
        Number of Optuna trials.
    multivariate : bool, default False
        If True, uses TPE (Tree-structured Parzen Estimator) sampler for multivariate hyperparameter optimization. If False, uses default Independent sampler.
    warm_up_steps : int, default 10
        Number of initial CV folds before trial pruning begins.
    startup_trials : int, default 5
        Number of complete trials evaluated before enabling the pruner.
    ref_series_id : str, optional
        Series ID used for date-based time index splitting (defaults to series with shortest length).
    verbose : bool, default False
        If True, prints progress for every Optuna trial.
    transform_back : bool, default True
        If True, transforms the forecasted values back to the original scale. False is recommended when there are many series with different scales, as it may lead to large errors in the evaluation metric when target scaler is passed to model.

    Returns
    -------
    Tuple[Dict[str, Any], Any, Dict[str, Any]]
        (best_hparams, best_lags, other_forecaster_args)
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required for hyperparameter optimization. Install with `pip install optuna`.")

    dfc = df.copy()
    id_col = model.id_col
    target_col = model.target_col
    series_ids = sorted(dfc[id_col].unique().tolist())

    # Determine reference series for time index splitting
    if ref_series_id is None:
        ref_series_id = min(series_ids, key=lambda s: len(dfc[dfc[id_col] == s]))
    ref_df = dfc[dfc[id_col] == ref_series_id]

    # Resolve target_series filter
    if target_series is None:
        eval_series_list = series_ids
    elif isinstance(target_series, str):
        if target_series not in series_ids:
            raise ValueError(f"Target series '{target_series}' not found in DataFrame '{id_col}' column.")
        eval_series_list = [target_series]
    elif isinstance(target_series, list):
        for s in target_series:
            if s not in series_ids:
                raise ValueError(f"Target series '{s}' not found in DataFrame '{id_col}' column.")
        eval_series_list = target_series
    else:
        raise TypeError("target_series must be None, str, or list of str.")

    _forecaster_keys = {
        "lags", "lag_transform", "series_encoding", "difference", "seasonal_diff",
        "trend", "pol_degree", "ets_params", "change_points", "box_cox",
        "box_cox_biasadj", "target_scaler", "cat_variables", "categorical_encoder"
    }

    tscv = SplitTimeSeries(n_splits=cv_split, test_size=test_size, step_size=step_size)

    def objective(trial: optuna.Trial) -> float:
        params = {name: suggest_fn(trial) for name, suggest_fn in param_space.items()}
        
        forecaster_params = {k: v for k, v in params.items() if k in _forecaster_keys}
        base_model_params = {k: v for k, v in params.items() if k not in _forecaster_keys}

        model_ = model.copy()

        # Update forecaster wrapper parameters
        for k, v in forecaster_params.items():
            setattr(model_, k, v)

        # Update base regressor parameters
        if hasattr(model_.model, "set_params") and base_model_params:
            model_.model.set_params(**base_model_params)

        cv_scores = []

        for step_idx, (ref_train_idx, ref_test_idx) in enumerate(tscv.split(ref_df)):
            cutoff_date = ref_df.index[ref_train_idx[-1]]
            test_dates = ref_df.index[ref_test_idx]

            train_fold = dfc[dfc.index <= cutoff_date]
            test_fold = dfc[(dfc.index > cutoff_date) & (dfc.index <= test_dates[-1])]

            exog_cols = [c for c in dfc.columns if c not in [id_col, target_col]]
            exog_fold = test_fold.drop(columns=[target_col]) if len(exog_cols) > 0 else None

            model_.fit(train_fold)
            H_fold = len(test_dates)
            fc_dict = model_.forecast(H=H_fold, exog=exog_fold)

            fold_series_scores = []
            for s in eval_series_list:
                s_test_df = test_fold[test_fold[id_col] == s]
                y_true_s = s_test_df[target_col].values
                y_pred_s = fc_dict[s][:len(y_true_s)]
                if not transform_back:
                    scaler = model_.transform_meta.get(s, {}).get("scaler", None)
                    if scaler is not None:
                        y_true_s = scaler.transform(y_true_s.reshape(-1, 1)).flatten() # flatten because inverse_transform returns 2D array, we make    
                        y_pred_s = scaler.transform(y_pred_s.reshape(-1, 1)).flatten()
                if eval_metric.__name__ in ("MASE", "SMAE", "SRMSE", "RMSSE"):
                    s_train_df = train_fold[train_fold[id_col] == s]
                    y_train_s = s_train_df[target_col].values
                    score_s = eval_metric(y_true_s, y_pred_s, y_train_s)
                else:
                    score_s = eval_metric(y_true_s, y_pred_s)

                fold_series_scores.append(score_s)

            fold_mean = np.mean(fold_series_scores)
            cv_scores.append(fold_mean)

            trial.report(np.mean(cv_scores), step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=startup_trials, n_warmup_steps=warm_up_steps),
        sampler=optuna.samplers.TPESampler(n_startup_trials=startup_trials, multivariate=multivariate, seed=random_state)
    )

    if verbose:
        def optuna_print_callback(study, trial):
            if trial.state == optuna.trial.TrialState.PRUNED:
                print(f"Trial {trial.number:3} | Status: PRUNED | Pruned at step: {list(trial.intermediate_values.keys())[-1] if trial.intermediate_values else 'N/A'}")
            else:
                print(f"Trial {trial.number:3} | Status: {trial.state.name:8} | Value: {trial.value:.6f} | Best trial: {study.best_trial.number:3} | Best Value: {study.best_value:.6f}")
        study.optimize(objective, n_trials=eval_num, callbacks=[optuna_print_callback])
    else:
        study.optimize(objective, n_trials=eval_num)

    best_results = study.best_params.copy()
    best_lags = best_results.pop("lags", None)
    other_args = {k: best_results.pop(k) for k in list(best_results.keys()) if k in _forecaster_keys}
    best_hparams = best_results

    return best_hparams, best_lags, other_args

```
:::


::: {#optuna_tune_multi_test .cell}
``` {.python .cell-code}
from sklearn.preprocessing import OneHotEncoder
#| hide
import os
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Callable, Tuple, Union
from sklearn.linear_model import Ridge
from peshbeen.metrics import MAE
from peshbeen.model_selection import SplitTimeSeries
from peshbeen.models.ml_multi_forecaster import ml_multi_forecaster
from sklearn.preprocessing import StandardScaler

# Load sales dataset using peshbeen.datasets
from peshbeen.datasets import load_sales
df_sales = load_sales()

forecaster = ml_multi_forecaster(
    model=Ridge(),
    id_col='store_item',
    target_col='sales',
    lags=7,
    id_col_encoder=OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
    target_scaler=StandardScaler()
)

param_space = {
    "alpha": lambda trial: trial.suggest_float("alpha", 1e-2, 10.0, log=True),
    # "lags": lambda trial: trial.suggest_int("lags", 3, 7),
    # "difference": lambda trial: trial.suggest_categorical("difference", [0, 1])
}

# Run test tuning across ALL series
best_hp, best_lags, other_args = optuna_tune_multi(
    model=forecaster,
    df=df_sales,
    cv_split=2,
    test_size=7,
    eval_metric=MAE,
    param_space=param_space,
    eval_num=2,
    verbose=True,
    transform_back=False
)
assert isinstance(best_hp, dict), "best_hp must be a dict"
print("Optuna Multi-Series Tuning Test Passed!")

```

::: {.cell-output .cell-output-stdout}
```
Trial   0 | Status: COMPLETE | Value: 0.309817 | Best trial:   0 | Best Value: 0.309817
Trial   1 | Status: COMPLETE | Value: 0.309816 | Best trial:   1 | Best Value: 0.309816
Optuna Multi-Series Tuning Test Passed!
```
:::
:::


## Feature selection methods for univariate time series models

::: {#855c8f86 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
#------------------------------------------------------------------------------
# Feature Selection Algorithms
# ------------------------------------------------------------------------------
 
def forward_feature_selection(
    model: object,
    df: pd.DataFrame,
    cv_split: int,
    H: int,
    metric: Callable,
    step_size: Optional[int] = None,
    lags_to_consider: Optional[int] = None,
    candidate_features: Optional[List[str]] = None,
    transformations: Optional[List] = None,
    starting_lags: Optional[List[int]] = None,
    starting_transforms: Optional[List] = None,
    best_start_score: Optional[float] = None, # Changed typing to float
    improve_rate_threshold: float = 0.0,
    warm_up_steps: int = 10,
    verbose: bool = False,
):
    """
    Forward stepwise feature selection for `ml_forecaster` models.
 
    At each iteration every remaining candidate (lag, exogenous column, or
    lag-transform) is tested individually by adding it to the current best
    feature set.  The candidate that produces the largest cross-validation
    improvement is permanently added.  The loop continues until no remaining
    candidate improves any of the evaluation metrics.
 
    Parameters
    ----------
    model : object
        A *configured but unfitted* `ml_forecaster` instance.  The function works exclusively on deep copies and never mutates the object passed in.
    df : pd.DataFrame
        Full training DataFrame. Must contain the target column and any candidate exogenous columns.
    cv_split : int
        Number of time-series cross-validation folds.
    H : int
        Forecast horizon (test window size for each fold).
    step_size : int, optional
        Step size between consecutive CV folds.  If `None` (default) the step equals `H`, producing non-overlapping folds — consistent with the default behaviour of `ml_forecaster.cross_validate`.
    metric : callable or list of callable.
        One or more metric functions accepted by `ml_forecaster.cross_validate` (e.g. `[MAE, RMSE]`). Selection is driven by the **first** metric in the list; a candidate is only accepted when it improves **all** metrics simultaneously.
    lags_to_consider : int, optional
        Consider lags `1, 2, ..., lags_to_consider` as candidates.  If `None`, lag selection is skipped.
    candidate_features : list of str, optional
        Column names in `df` that are exogenous feature candidates.  The function never modifies this list.  If `None`, exogenous feature selection is skipped.
    transformations : list, optional
        Lag-transform objects to test as candidates (e.g. `[rolling_mean(3, 1), expanding_std(1)]`).  The function never modifies this list.  If `None`, transform selection is skipped.
    starting_lags : list of int, optional
        Lags to include in the initial feature set before the search begins. These are *not* candidates — they are always included.  Must be a list (e.g. `[1]` or `[1, 2, 3]`).
    starting_transforms : list, optional
        Lag-transform objects to include in the initial feature set before the search begins.  Must be a list.
    best_start_score : float, optional
        Initial best score for the metric. If not provided, the function will compute the baseline score using the model with the starting features (if any) before beginning the search.
    improve_rate_threshold : float, default 0.0
        Minimum improvement rate required for a candidate to be accepted. For example, `0.05` means the new score must be at least 5% better than the current best score to be accepted.
    warm_up_steps : int, default 10
        Number of CV folds to evaluate before pruning candidates that perform worse than the current best score. This allows the function to gather enough performance data for reliable pruning decisions.
    verbose : bool, default False
        Print a message each time a candidate is accepted.
 
    Returns
    -------
    dict
        A dictionary with keys `best_lags`, `best_exogs`, and `best_transforms` containing the selected features.
    """
 
    if metric is None:
        raise ValueError("metric must be provided (a callable).")
 
    _step_size = step_size if step_size is not None else H
 
    if starting_lags is not None and not isinstance(starting_lags, list):
        raise ValueError("starting_lags must be a list of integers, e.g. [1] or [1, 2, 3].")
    if starting_transforms is not None and not isinstance(starting_transforms, list):
        raise ValueError("starting_transforms must be a list of transformation instances.")
 
    # Current best lags / transforms — kept in sync as candidates are accepted
    current_lags       = list(starting_lags) if starting_lags is not None else []
    current_transforms = list(starting_transforms) if starting_transforms is not None else []

    # Build candidate pools
    remaining_lags = (
        [x for x in range(1, lags_to_consider + 1) if x not in current_lags]
        if lags_to_consider is not None else []
    )
    remaining_feats = list(candidate_features) if candidate_features is not None else []
    remaining_transforms = list(transformations) if transformations is not None else []
 
    # Working df — start without exog candidates, add back as selected
    if candidate_features is not None:
        df_work = df.drop(columns=candidate_features)
    else:
        df_work = df.copy()
 
    best_features = {
        "best_lags":       current_lags.copy(),
        "best_exogs":      [],
        "best_transforms": current_transforms.copy(),
    }
 
    def _scores_improved(new_score, ref_score):
        """True only when every metric strictly improves."""
        if new_score == float('inf'):
            return False
        if ref_score == 0:
            return new_score < 0 # Avoid division by zero
            
        # FIXED: Added parentheses around the subtraction to ensure correct math
        improvement_rate = (ref_score - new_score) / abs(ref_score)
        return improvement_rate > improve_rate_threshold
 
    tscv = SplitTimeSeries(n_splits=cv_split, test_size=H, step_size=_step_size)

    step_score_tracker = {idx: [] for idx in range(cv_split)}

    def _validate(m, df_test, ref_score=None):
        """Fit-and-score one candidate model via manual cross-validation with early pruning."""
        cv_scores = []

        if m.get_name() == "ms_arr":
            m.coeffs = None
            m.stds = None
        
        target_col = getattr(m, "target_col", None)
        if target_col is None and hasattr(m, "models"):
            target_col = list(m.models.values())[0].target_col

        for step_idx, (train_idx, test_idx) in enumerate(tscv.split(df_test)):
            train, test = df_test.iloc[train_idx], df_test.iloc[test_idx]
            x_test = test.drop(columns=[target_col], errors="ignore")
            y_test = np.array(test[target_col])

            m.fit(train)
            exog_t = x_test if x_test.shape[1] > 0 else None
            y_pred = m.forecast(H=len(test), exog=exog_t)

            if isinstance(y_pred, pd.DataFrame):
                if m.get_name() == "pesh" and "pesh" in y_pred.columns:
                    y_pred = y_pred["pesh"].values
                elif target_col in y_pred.columns:
                    y_pred = y_pred[target_col].values
                else:
                    y_pred = y_pred.iloc[:, -1].values

            if metric.__name__ in ("MASE", "SMAE", "SRMSE", "RMSSE"):
                score = metric(y_test, y_pred, train[target_col])
            else:
                score = metric(y_test, y_pred)
            cv_scores.append(score)

            current_mean = np.mean(cv_scores)
            step_score_tracker[step_idx].append(current_mean)
            
            if ref_score is not None and ref_score != float('inf'):
                if (step_idx + 1) >= warm_up_steps:
                    if len(step_score_tracker[step_idx]) > warm_up_steps:  # Ensure we have enough data points to calculate a reliable median
                        if current_mean > np.median(step_score_tracker[step_idx][:-1]):
                            return float('inf')

        return np.mean(cv_scores)
 
    def _make_candidate_model(lags, transforms, active_exogs):
        """Return a deep copy of the base model configured with the given lags and transforms."""
        m = model.copy()
        if lags_to_consider is not None:
            # Safely assign to the correct attribute
            if hasattr(m, "lags"):
                m.lags = sorted(lags) if lags else None
            else:
                m.n_lag = sorted(lags) if lags else None
        if transformations is not None:
            m.lag_transform = list(transforms) if transforms else None
        return m
 
    # FIXED: Extract list value if user accidentally passed a list instead of float
    if best_start_score is not None:
        best_score = best_start_score[0] if isinstance(best_start_score, list) else best_start_score
    else:
        best_score = float('inf')

    # Baseline score with starting features (if any and if best_start_score isn't explicitly provided)
    if best_start_score is None and (current_lags or current_transforms or (candidate_features and len(candidate_features) > 0)):
        m_start = _make_candidate_model(current_lags, current_transforms, best_features["best_exogs"])
        best_score = _validate(m_start, df_work)
        if verbose:
            print(f"Baseline score with starting features: {best_score}")
 
    # ── Stepwise forward loop ─────────────────────────────────────────────────
    while True:
        improvement    = False
        best_candidate = {'type': None, 'value': None}
        # FIXED: Set running score to best_score at the start of loop iteration
        running_score  = best_score
 
        # ── Test lag candidates ───────────────────────────────────────────────
        if remaining_lags and lags_to_consider is not None:
            for lag in remaining_lags:
                m = _make_candidate_model(current_lags + [lag], current_transforms, best_features["best_exogs"])
                score = _validate(m, df_work, running_score)
                if _scores_improved(score, running_score):
                    running_score  = score
                    best_candidate = {'type': 'lag', 'value': lag}
                    improvement    = True
 
        # ── Test exogenous feature candidates ─────────────────────────────────
        if remaining_feats and candidate_features is not None:
            for feat in remaining_feats:
                # OPTIMIZED: Temporarily add the column to avoid full DataFrame copying
                df_work[feat] = df[feat]
                m = _make_candidate_model(current_lags, current_transforms, best_features["best_exogs"] + [feat])
                
                score = _validate(m, df_work, running_score)
                
                # Drop immediately after validation
                df_work.drop(columns=[feat], inplace=True)
                
                if _scores_improved(score, running_score):
                    running_score  = score
                    best_candidate = {'type': 'exog', 'value': feat}
                    improvement    = True
 
        # ── Test lag-transform candidates ─────────────────────────────────────
        if remaining_transforms and transformations is not None:
            for trans in remaining_transforms:
                m = _make_candidate_model(current_lags, current_transforms + [trans], best_features["best_exogs"])
                score = _validate(m, df_work, running_score)
                if _scores_improved(score, running_score):
                    running_score  = score
                    best_candidate = {'type': 'transform', 'value': trans}
                    improvement    = True
 
        # ── Accept the best candidate found this round ────────────────────────
        if improvement:
            best_score = running_score
            ctype = best_candidate['type']
            cval  = best_candidate['value']
 
            if ctype == 'lag':
                current_lags.append(cval)
                current_lags.sort()
                best_features["best_lags"].append(cval)
                best_features["best_lags"].sort()
                remaining_lags.remove(cval)
 
            elif ctype == 'exog':
                best_features["best_exogs"].append(cval)
                remaining_feats.remove(cval)
                # OPTIMIZED: Permanently apply the winning column without copying
                df_work[cval] = df[cval]
 
            elif ctype == 'transform':
                current_transforms.append(cval)
                best_features["best_transforms"].append(cval)
                remaining_transforms.remove(cval)
 
            if verbose:
                label = cval.get_name() if hasattr(cval, 'get_name') else str(cval)
                print(f"Added {ctype}: {label} | score: {best_score}")
 
        else:
            break  # No candidate improved any metric — search is complete
 
    # ── Finalise output ───────────────────────────────────────────────────────
    best_features["best_transforms"] = [
        t.get_name() if hasattr(t, 'get_name') else str(t)
        for t in best_features["best_transforms"]
    ]
 
    return best_features
```
:::


::: {#3129406c .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
ml_linear = ml_forecaster(model=LinearRegression(),
              target_col='admissions', lags = 30, difference=1, categorical_encoder=ohe,
              cat_variables=cat_variables)
feats = forward_feature_selection(
    df=train,
    cv_split=5,
    H=30,
    model=ml_linear,
    metric=MAE,
    lags_to_consider=15,
    transformations=None,
    starting_lags=None,
    starting_transforms=None,
    verbose=True
)
```
:::


::: {#62f1fb3f .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def backward_feature_selection(
    model: object,
    df: pd.DataFrame,
    cv_split: int,
    H: int,
    step_size: Optional[int] = None,
    metrics: Union[Callable, List[Callable]] = None,
    lags_to_consider: Optional[List[int]] = None,
    candidate_features: Optional[List[str]] = None,
    transformations: Optional[List] = None,
    verbose=False,
):
    """
    Backward stepwise feature selection for `ml_forecaster` models.

    Starts with the full feature set (all provided lags, exogenous columns, and
    lag-transforms) and at each iteration tries removing each current feature
    individually.  The feature whose removal produces the largest cross-validation
    improvement is permanently dropped.  The loop continues until no remaining
    feature can be removed without hurting any of the evaluation metrics.

    Parameters
    ----------
    model : ml_forecaster
        A *configured but unfitted* `ml_forecaster` instance.  The function works exclusively on deep copies and never mutates the object passed in.
    df : pd.DataFrame
        Full training DataFrame. Must contain the target column and any candidate exogenous columns.
    cv_split : int
        Number of time-series cross-validation folds.
    H : int
        Forecast horizon (test window size for each fold).
    step_size : int, optional
        Step size between consecutive CV folds.  If `None` (default) the step equals `H`, producing non-overlapping folds.
    metrics : callable or list of callable
        One or more metric functions accepted by `ml_forecaster.cross_validate` (e.g. `[MAE, RMSE]`). Selection is driven by the **first** metric in the list; a feature is only removed when doing so improves **all** metrics simultaneously.
    lags_to_consider : list of int, optional
        Lags to include in the initial feature set and test for removal (e.g. `[1, 2, 3, 4]`).  If `None`, no lag removal is attempted.
    candidate_features : list of str, optional
        Column names in `df` that start in the model and are tested for removal.  If `None`, exogenous feature removal is skipped.
    transformations : list, optional
        Lag-transform objects that start in the model and are tested for removal (e.g. `[rolling_mean(3, 1), expanding_std(1)]`).  If `None`, transform removal is skipped.
    verbose : bool, default False
        Print a message each time a feature is removed.

    Returns
    -------
    dict
        A dictionary with keys `best_lags`, `best_exogs`, and `best_transforms` containing the surviving features after backward selection.
    """

    # ── Normalise metrics to always be a list ─────────────────────────────────
    if metrics is None:
        raise ValueError("metrics must be provided (a callable or list of callables).")
    if callable(metrics):
        metrics = [metrics]

    # ── step_size default: non-overlapping folds (step = H) ───────────────────
    _step_size = step_size if step_size is not None else H

    # ── Build initial feature sets (local copies — never mutate caller's lists) ─
    if lags_to_consider is None:
        current_lags = []
    elif isinstance(lags_to_consider, int):
        current_lags = list(range(1, lags_to_consider + 1))
    else:
        current_lags = sorted(lags_to_consider)
    current_exogs      = list(candidate_features)  if candidate_features is not None else []
    current_transforms = list(transformations)      if transformations    is not None else []


    # ── Working df includes all exog candidates from the start ────────────────
    df_work = df.copy()

    # ── Columns in candidate_features that are also cat_variables on the model ─
    _cat_exog_candidates = set()
    if candidate_features is not None and model.cat_variables is not None:
        _cat_exog_candidates = set(candidate_features) & set(model.cat_variables)

    # ── Inner helpers ─────────────────────────────────────────────────────────

    def _scores_improved(new_score, ref_score):
        """True only when every metric strictly improves."""
        return all(n < r for n, r in zip(new_score, ref_score))

    def _make_candidate_model(lags, transforms, active_exogs=None):
        """
        Return a deep copy of the base model configured with the given lags and
        transforms.  Cat_variables that are no longer in active_exogs are stripped
        from the copy so fit() does not look for missing columns.
        """
        m = model.copy()
        m.n_lag         = sorted(lags)     if lags       else None
        m.lag_transform = list(transforms) if transforms else None

        if m.cat_variables is not None and _cat_exog_candidates:
            active = set(active_exogs) if active_exogs else set()
            present_cats = [
                c for c in m.cat_variables
                if c not in _cat_exog_candidates or c in active]
            m.cat_variables = present_cats if present_cats else None

        return m

    def _validate(m, df_test):
        """Fit-and-score one candidate model via cross-validation."""
        m.cross_validate(
            df=df_test,
            cv_split=cv_split,
            test_size=H,
            metrics=metrics,
            step_size=_step_size,
        )
        result_df = m.cv_summary
        return result_df["overall_score"].tolist()

    # ── Baseline score with all features ──────────────────────────────────────
    m_start = _make_candidate_model(current_lags, current_transforms, current_exogs)
    best_score = _validate(m_start, df_work)
    if verbose:
        print(f"Baseline score with all features: {best_score}")

    # ── Stepwise backward loop ────────────────────────────────────────────────
    while True:
        improvement    = False
        best_candidate = {'type': None, 'value': None}
        running_score  = best_score

        # ── Test lag removals ─────────────────────────────────────────────────
        if current_lags:
            for lag in current_lags:
                reduced_lags = [l for l in current_lags if l != lag]
                m = _make_candidate_model(reduced_lags, current_transforms, current_exogs)
                score = _validate(m, df_work)
                if _scores_improved(score, running_score):
                    running_score  = score
                    best_candidate = {'type': 'lag', 'value': lag}
                    improvement    = True

        # ── Test exogenous feature removals ───────────────────────────────────
        if current_exogs:
            for feat in current_exogs:
                reduced_exogs = [f for f in current_exogs if f != feat]
                df_test = df_work.drop(columns=[feat])
                m = _make_candidate_model(current_lags, current_transforms, reduced_exogs)
                score = _validate(m, df_test)
                if _scores_improved(score, running_score):
                    running_score  = score
                    best_candidate = {'type': 'exog', 'value': feat}
                    improvement    = True

        # ── Test lag-transform removals ───────────────────────────────────────
        if current_transforms:
            for trans in current_transforms:
                reduced_transforms = [t for t in current_transforms if t is not trans]
                m = _make_candidate_model(current_lags, reduced_transforms, current_exogs)
                score = _validate(m, df_work)
                if _scores_improved(score, running_score):
                    running_score  = score
                    best_candidate = {'type': 'transform', 'value': trans}
                    improvement    = True

        # ── Accept the best removal found this round ──────────────────────────
        if improvement:
            best_score = running_score
            ctype = best_candidate['type']
            cval  = best_candidate['value']

            if ctype == 'lag':
                current_lags.remove(cval)
                if verbose:
                    print(f"Removed lag: {cval} | score: {best_score}")

            elif ctype == 'exog':
                current_exogs.remove(cval)
                df_work = df_work.drop(columns=[cval])
                if verbose:
                    print(f"Removed exog: {cval} | score: {best_score}")

            elif ctype == 'transform':
                current_transforms = [t for t in current_transforms if t is not cval]
                if verbose:
                    label = cval.get_name()
                    print(f"Removed transform: {label} | score: {best_score}")

        else:
            break  # No removal improved any metric — search is complete

    # ── Finalise output ───────────────────────────────────────────────────────
    return {
        "best_lags":       current_lags,
        "best_exogs":      current_exogs,
        "best_transforms": [t.get_name() for t in current_transforms],
    }

```
:::


::: {#42dbe094 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
ml_linear = ml_forecaster(model=LinearRegression(),
              target_col='admissions', lags = 30, difference=1, categorical_encoder=ohe,
              cat_variables=cat_variables)
feats = backward_feature_selection(
    df=train,
    cv_split=5,
    H=30,
    model=ml_linear,
    metrics=[MAE, RMSE],
    lags_to_consider=10,
    candidate_features=["month"],
    transformations=None,
    verbose=True
)
```

::: {.cell-output .cell-output-stdout}
```
Baseline score with all features: [507.57446530410033, 579.4020476650461]
Removed exog: month | score: [306.76607444762516, 361.22458411154975]
Removed lag: 6 | score: [297.93939946966464, 351.9954499886267]
Removed lag: 2 | score: [294.1369942216585, 348.75752303462014]
Removed lag: 5 | score: [292.161501826388, 346.88304443973703]
Removed lag: 10 | score: [289.72402228318884, 344.82523600153763]
Removed lag: 7 | score: [288.0925883221115, 343.65266888864215]
Removed lag: 8 | score: [287.3327517919541, 342.9666419408003]
Removed lag: 9 | score: [278.02568538895446, 332.3523248898411]
Removed lag: 1 | score: [277.76096946580094, 331.9474715817777]
```
:::
:::


## Feature selection methods for multivariate time series models

::: {#46039723 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``````````` {.python .cell-code}
def mv_forward_feature_selection(
    model: object,
    df: pd.DataFrame,
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
    verbose=False,
):
    """
    Forward stepwise feature selection for `ml_mv_forecaster`.

    Parameters
    ----------
    model : ml_mv_forecaster
        Template model — never mutated.
    df : pd.DataFrame
        DataFrame containing the target variable and any candidate features.
    target_col : str
        Target variable used to evaluate cross-validation score.
    cv_split : int
        Number of time-series cross-validation folds.
    H : int
        Forecast horizon / test size per fold.
    step_size : int, optional
        Rolling-window step size (defaults to H).
    metrics : callable or list of callable
        One or more metric functions (e.g. `[MAE, RMSE]`). Selection is driven by the **first** metric in the list; a candidate is only accepted when it improves **all** metrics simultaneously.
    lags_to_consider : dict, optional
        ``{col: max_lag}`` — lags 1..max_lag are candidates.
    candidate_features : list of str, optional
        Exogenous columns to consider adding.
    transformations : dict, optional
        ``{col: [transform_objects]}`` — transform candidates per target.
    starting_lags : dict, optional
        Lags already included before search begins.
    starting_transforms : dict, optional
        Transforms already included before search begins.
    verbose : bool, default False

    Returns
    -------
    dict
        `{"best_lags": {col: [...]}, "best_exogs": [...],
           "best_transforms": {col: [name_str, ...]}}`
    """
    if metrics is None:
        raise ValueError("metrics must be provided.")
    if callable(metrics):
        metrics = [metrics]

    _step_size = step_size if step_size is not None else H

    if lags_to_consider is None:
        lags_to_consider = {}
    if transformations is None:
        transformations = {}

    remaining_lags = {col: list(range(1, ml + 1)) for col, ml in lags_to_consider.items()}
    remaining_transforms = {col: list(tl) for col, tl in transformations.items()}
    remaining_feats = list(candidate_features) if candidate_features is not None else []

    best_lags = {col: [] for col in lags_to_consider}
    if starting_lags is not None:
        for col, lags in starting_lags.items():
            best_lags[col] = list(lags)
            remaining_lags[col] = [x for x in remaining_lags.get(col, []) if x not in lags]

    best_transforms = {col: [] for col in transformations}
    if starting_transforms is not None:
        for col, tl in starting_transforms.items():
            best_transforms[col] = list(tl)
            remaining_transforms[col] = [t for t in remaining_transforms.get(col, []) if t not in tl]

    best_features = {
        "best_lags":       best_lags,
        "best_exogs":      [],
        "best_transforms": best_transforms,
    }

    if candidate_features is not None:
        df_work = df.drop(columns=candidate_features)
        df_orig = df.copy()
    else:
        df_work = df.copy()
        df_orig = df.copy()

    best_score = [float('inf')] * len(metrics)

    _cat_exog_candidates = set()
    if candidate_features is not None and model.cat_variables is not None:
        _cat_exog_candidates = set(candidate_features) & set(model.cat_variables)

    def _scores_improved(new_score, ref_score):
        return all(n < r for n, r in zip(new_score, ref_score))

    def _validate(m, df_test):
        # cross_validate returns (metrics_df, cv_df_); metrics_df has columns
        # ["eval_metric", "score"] — one row per metric.
        m.cross_validate(
            df=df_test, target_col=target_col, cv_split=cv_split,
            test_size=H, metrics=metrics, step_size=_step_size,
        )
        result_df = m.cv_summary
        return result_df['overall_score'].tolist()

    def _make_candidate_model(lags_dict, transforms_dict, active_exogs=None):
        m = model.copy()
        # Use {} (not None) when no lags — ml_mv_forecaster.data_prep iterates m.n_lag
        m.n_lag = {col: sorted(v) for col, v in lags_dict.items() if v}
        at = {col: list(v) for col, v in transforms_dict.items() if v}
        m.lag_transform = at if at else None
        if m.cat_variables is not None and _cat_exog_candidates:
            active = set(active_exogs) if active_exogs else set()
            present = [c for c in m.cat_variables
                       if c not in _cat_exog_candidates or c in active]
            m.cat_variables = present if present else None
        return m

    # Evaluate starting point if warm-start provided
    if starting_lags is not None or starting_transforms is not None:
        m_start = _make_candidate_model(
            best_features["best_lags"],
            best_features["best_transforms"],
            best_features["best_exogs"],
        )
        best_score = _validate(m_start, df_work)
        if verbose:
            print(f"Baseline score: {best_score}")

    while True:
        improvement    = False
        best_candidate = {'target': None, 'type': None, 'value': None}
        running_score  = best_score

        # --- try adding each candidate lag ---
        for col, lags in remaining_lags.items():
            for lg in lags:
                trial_lags = {c: list(v) for c, v in best_features["best_lags"].items()}
                trial_lags[col] = sorted(trial_lags[col] + [lg])
                m = _make_candidate_model(trial_lags, best_features["best_transforms"],
                                          best_features["best_exogs"])
                score = _validate(m, df_work)
                if _scores_improved(score, running_score):
                    running_score  = score
                    best_candidate = {'target': col, 'type': 'lag', 'value': lg}
                    improvement    = True

        # --- try adding each candidate exog feature ---
        for feat in remaining_feats:
            df_test = df_work.copy()
            df_test[feat] = df_orig[feat]
            active_exogs = best_features["best_exogs"] + [feat]
            m = _make_candidate_model(best_features["best_lags"],
                                      best_features["best_transforms"], active_exogs)
            score = _validate(m, df_test)
            if _scores_improved(score, running_score):
                running_score  = score
                best_candidate = {'target': None, 'type': 'exog', 'value': feat}
                improvement    = True

        # --- try adding each candidate transform ---
        for col, tl in remaining_transforms.items():
            for trans in tl:
                trial_trans = {c: list(v) for c, v in best_features["best_transforms"].items()}
                trial_trans[col] = trial_trans[col] + [trans]
                m = _make_candidate_model(best_features["best_lags"], trial_trans,
                                          best_features["best_exogs"])
                score = _validate(m, df_work)
                if _scores_improved(score, running_score):
                    running_score  = score
                    best_candidate = {'target': col, 'type': 'transform', 'value': trans}
                    improvement    = True

        if improvement:
            best_score = running_score
            ctype  = best_candidate['type']
            cval   = best_candidate['value']
            ctargt = best_candidate['target']
            if ctype == 'lag':
                best_features["best_lags"][ctargt].append(cval)
                best_features["best_lags"][ctargt].sort()
                remaining_lags[ctargt].remove(cval)
            elif ctype == 'exog':
                best_features["best_exogs"].append(cval)
                remaining_feats.remove(cval)
                df_work[cval] = df_orig[cval]
            elif ctype == 'transform':
                best_features["best_transforms"][ctargt].append(cval)
                remaining_transforms[ctargt].remove(cval)
            if verbose:
                label = cval.get_name() if ctype == 'transform' else cval
                tgt_str = f" [{ctargt}]" if ctargt else ""
                print(f"Added {ctype}{tgt_str}: {label} | score: {best_score}")
        else:
            break

    for col in best_features["best_lags"]:
        best_features["best_lags"][col].sort()
    best_features["best_transforms"] = {
        col: [t.get_name() for t in tl]
        for col, tl in best_features["best_transforms"].items()
    }
    return best_features
```````````
:::


::: {#ecd237a4 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
from peshbeen.datasets import load_admission_calls

from peshbeen.models import ml_mv_forecaster

admission_calls = load_admission_calls()
## get day of week and month as features from the date index
admission_calls["day_of_week"] = admission_calls.index.dayofweek
admission_calls["month"] = admission_calls.index.month
train = admission_calls[:-30]
test = admission_calls[-30:]

cat_variables = ["day_of_week", "month"]
ml_linear = ml_mv_forecaster(model=LinearRegression(),
              target_cols=['admissions', "calls"], lags = {"admissions": 7, "calls": 7}, categorical_encoder=ohe,
                cat_variables=cat_variables,
                trend={"admissions": "linear", "calls": "linear"}, change_points={"admissions": [100], "calls": [130]}, pol_degree={"admissions": 1, "calls": 1})
# ml_linear.fit(train)
# forecasts = ml_linear.forecast(H=30, exog=test[cat_variables])
```
:::


::: {#cb29ade8 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
feats_mvf = mv_forward_feature_selection(
    model=ml_linear,
    df=train,
    target_col='admissions',
    cv_split=3,
    H=30,
    metrics=[MAE, RMSE],
    lags_to_consider={"admissions": 7, "calls": 7},
    candidate_features=cat_variables,
    verbose=True
)
```

::: {.cell-output .cell-output-stdout}
```
Added exog: month | score: [166.74875519029243, 215.17550849515987]
Added lag [admissions]: 1 | score: [128.66697431600946, 159.75541150704558]
Added exog: day_of_week | score: [110.8232632739758, 141.59466251840496]
Added lag [calls]: 1 | score: [103.28304873209088, 133.95230334295243]
Added lag [calls]: 2 | score: [102.08224996483074, 132.48442870101943]
```
:::
:::


::: {#8f3f6822 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``````````` {.python .cell-code}
def mv_backward_feature_selection(
    model: object,
    df: pd.DataFrame,
    target_col: str,
    cv_split: int,
    H: int,
    step_size=None,
    metrics=None,
    lags_to_consider=None,
    candidate_features=None,
    transformations=None,
    verbose=False,
):
    """
    Backward stepwise feature selection for ``ml_mv_forecaster``.

    Starts with all candidate features included and iteratively removes
    the one whose removal most improves cross-validation score.

    Parameters
    ----------
    model : ml_mv_forecaster
        Template model — never mutated.
    df : pd.DataFrame
        All candidate exog columns must already be present.
    target_col : str
        Target variable used to evaluate cross-validation score.
    cv_split : int
    H : int
        Forecast horizon / test size per fold.
    step_size : int, optional
        Rolling-window step size (defaults to H).
    metrics : callable or list of callable
        One or more metric functions (e.g. `[MAE, RMSE]`). A feature is only removed when its removal improves **all** metrics simultaneously.
    lags_to_consider : dict, optional
        ``{col: max_lag}`` — all lags 1..max_lag start as selected.
    candidate_features : list of str, optional
        Exogenous columns that start as selected.
    transformations : dict, optional
        ``{col: [transform_objects]}`` — all transforms start as selected.
    verbose : bool, default False

    Returns
    -------
    dict
        ``{"best_lags": {col: [...]}, "best_exogs": [...],
           "best_transforms": {col: [name_str, ...]}}``
    """
    if metrics is None:
        raise ValueError("metrics must be provided.")
    if callable(metrics):
        metrics = [metrics]

    _step_size = step_size if step_size is not None else H

    best_features = {
        "best_lags": {col: list(range(1, ml + 1))
                      for col, ml in (lags_to_consider or {}).items()},
        "best_exogs": list(candidate_features) if candidate_features is not None else [],
        "best_transforms": {col: list(tl) for col, tl in (transformations or {}).items()},
    }

    df_work = df.copy()

    _cat_exog_candidates = set()
    if candidate_features is not None and model.cat_variables is not None:
        _cat_exog_candidates = set(candidate_features) & set(model.cat_variables)

    def _scores_improved(new_score, ref_score):
        return all(n < r for n, r in zip(new_score, ref_score))

    def _validate(m, df_test):
        m.cross_validate(
            df=df_test, target_col=target_col, cv_split=cv_split,
            test_size=H, metrics=metrics, step_size=_step_size,
        )
        result_df = m.cv_summary
        return result_df['overall_score'].tolist()

    def _make_candidate_model(lags_dict, transforms_dict, active_exogs=None):
        m = model.copy()
        m.n_lag = {col: sorted(v) for col, v in lags_dict.items() if v}
        at = {col: list(v) for col, v in transforms_dict.items() if v}
        m.lag_transform = at if at else None
        if m.cat_variables is not None and _cat_exog_candidates:
            active = set(active_exogs) if active_exogs else set()
            present = [c for c in m.cat_variables
                       if c not in _cat_exog_candidates or c in active]
            m.cat_variables = present if present else None
        return m

    # Evaluate baseline (all features included) before starting removal
    m_baseline = _make_candidate_model(
        best_features["best_lags"],
        best_features["best_transforms"],
        best_features["best_exogs"],
    )
    best_score = _validate(m_baseline, df_work)
    if verbose:
        print(f"Baseline score (all features): {best_score}")

    while True:
        improvement    = False
        best_candidate = {'target': None, 'type': None, 'value': None}
        running_score  = best_score

        # --- try removing each selected lag ---
        for targ_col, lags in best_features["best_lags"].items():
            for lg in lags:
                trial_lags = {col: list(v) for col, v in best_features["best_lags"].items()}
                trial_lags[targ_col] = sorted([x for x in lags if x != lg])
                m = _make_candidate_model(trial_lags, best_features["best_transforms"],
                                          best_features["best_exogs"])
                score = _validate(m, df_work)
                if _scores_improved(score, running_score):
                    running_score  = score
                    best_candidate = {'target': targ_col, 'type': 'lag', 'value': lg}
                    improvement    = True

        # --- try removing each selected transform ---
        for targ_col, tl in best_features["best_transforms"].items():
            for tr in tl:
                trial_trans = {col: list(v) for col, v in best_features["best_transforms"].items()}
                trial_trans[targ_col] = [x for x in tl if x is not tr]
                m = _make_candidate_model(best_features["best_lags"], trial_trans,
                                          best_features["best_exogs"])
                score = _validate(m, df_work)
                if _scores_improved(score, running_score):
                    running_score  = score
                    best_candidate = {'target': targ_col, 'type': 'transform', 'value': tr}
                    improvement    = True

        # --- try removing each selected exog feature ---
        for feat in best_features["best_exogs"]:
            df_test      = df_work.drop(columns=[feat])
            active_after = [x for x in best_features["best_exogs"] if x != feat]
            m = _make_candidate_model(best_features["best_lags"],
                                      best_features["best_transforms"], active_after)
            score = _validate(m, df_test)
            if _scores_improved(score, running_score):
                running_score  = score
                best_candidate = {'target': None, 'type': 'exog', 'value': feat}
                improvement    = True

        if improvement:
            best_score = running_score
            ctype  = best_candidate['type']
            cval   = best_candidate['value']
            ctargt = best_candidate['target']
            if ctype == 'lag':
                best_features["best_lags"][ctargt].remove(cval)
            elif ctype == 'exog':
                best_features["best_exogs"].remove(cval)
                df_work = df_work.drop(columns=[cval])
            elif ctype == 'transform':
                best_features["best_transforms"][ctargt].remove(cval)
            if verbose:
                label = cval.get_name() if ctype == 'transform' else cval
                tgt_str = f" [{ctargt}]" if ctargt else ""
                print(f"Removed {ctype}{tgt_str}: {label} | score: {best_score}")
        else:
            break

    for col in best_features["best_lags"]:
        best_features["best_lags"][col].sort()
    best_features["best_transforms"] = {
        col: [t.get_name() for t in tl]
        for col, tl in best_features["best_transforms"].items()
    }
    return best_features
```````````
:::


::: {#806f5543 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
feats_mvb = mv_backward_feature_selection(
    model=ml_linear,
    df=train,
    target_col='admissions',
    cv_split=3,
    H=30,
    metrics=[MAE, RMSE],
    lags_to_consider={"admissions": 7, "calls": 7},
    candidate_features=cat_variables,
    verbose=True
)
```

::: {.cell-output .cell-output-stdout}
```
Baseline score (all features): [109.29717677964436, 148.29831037859776]
Removed lag [admissions]: 1 | score: [105.91419788677575, 146.73082839224705]
Removed lag [admissions]: 7 | score: [104.52863353888428, 142.8577988679976]
Removed lag [calls]: 4 | score: [103.42189903529406, 141.93740105885334]
Removed lag [calls]: 7 | score: [102.46152777328761, 140.19391221941834]
Removed lag [calls]: 6 | score: [101.56647138608987, 139.30555349890432]
Removed lag [calls]: 5 | score: [100.81916910153232, 138.0052305954924]
Removed lag [admissions]: 6 | score: [100.7880710238256, 137.00717747155758]
Removed lag [admissions]: 4 | score: [100.62239607840796, 136.99786144919938]
```
:::
:::


::: {#b76b27c1 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
from peshbeen.models import var
var_model = var(target_cols=['admissions', "calls"], lags={'admissions': 7, "calls": 7}, trend={'admissions': "linear", "calls": "linear"},
                cat_variables=cat_variables, change_points={'admissions': [100], "calls": [130]},
                categorical_encoder=ohe)
feats_mvf_var = mv_forward_feature_selection(
    model=var_model,
    df=train,
    target_col='admissions',
    cv_split=3,
    H=30,
    metrics=[MAE, RMSE],
    lags_to_consider={"admissions": 7, "calls": 7},
    candidate_features=cat_variables,
    verbose=True
)
feats_mvf_var
```

::: {.cell-output .cell-output-stdout}
```
Added exog: month | score: [166.748755190292, 215.17550849515956]
Added lag [admissions]: 1 | score: [128.66697431601406, 159.75541150704734]
Added exog: day_of_week | score: [110.82326327397881, 141.5946625184181]
Added lag [calls]: 1 | score: [103.28304873205606, 133.95230334290463]
Added lag [calls]: 2 | score: [102.08224996475475, 132.48442870091307]
```
:::

::: {.cell-output .cell-output-display}
```
{'best_lags': {'admissions': [1], 'calls': [1, 2]},
 'best_exogs': ['month', 'day_of_week'],
 'best_transforms': {}}
```
:::
:::


::: {#8cf3a51d .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
feats_mvb_var = mv_backward_feature_selection(
    model=var_model,
    df=train,
    target_col='admissions',
    cv_split=3,
    H=30,
    metrics=[MAE, RMSE],
    lags_to_consider={"admissions": 15, "calls": 15},
    verbose=True
)
feats_mvb_var
```

::: {.cell-output .cell-output-stdout}
```
Baseline score (all features): [153.61836133398228, 200.9696238804104]
Removed lag [calls]: 13 | score: [148.40591785644847, 195.15154160184457]
Removed lag [calls]: 14 | score: [146.5362269280274, 190.33325164086145]
Removed lag [admissions]: 9 | score: [145.3263248960864, 189.13277347195526]
Removed lag [admissions]: 8 | score: [143.99157468800897, 187.86488735637678]
Removed lag [admissions]: 6 | score: [143.0021704871917, 186.88430243231878]
Removed lag [admissions]: 5 | score: [141.4711131351589, 186.14884612639602]
Removed lag [calls]: 7 | score: [140.37455534940875, 185.25906199525153]
Removed lag [calls]: 5 | score: [139.5678276274476, 184.72631648710572]
Removed lag [calls]: 10 | score: [138.76482507320432, 183.45321039355363]
Removed lag [admissions]: 14 | score: [138.63006459724355, 183.4166970136962]
```
:::

::: {.cell-output .cell-output-display}
```
{'best_lags': {'admissions': [1, 2, 3, 4, 7, 10, 11, 12, 13, 15],
  'calls': [1, 2, 3, 4, 6, 8, 9, 11, 12, 15]},
 'best_exogs': [],
 'best_transforms': {}}
```
:::
:::


