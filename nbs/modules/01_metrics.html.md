---
title: Metrics
---





::: {#6956d1df .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
from __future__ import annotations
import numpy as np
import pandas as pd
ArrayLike = np.ndarray|list|pd.Series

def MAPE(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between actual and predicted values.

    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.

    Returns
    -------
    float
        The MAPE value rounded to 2 decimal places.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return round(np.mean(np.abs((y_true - y_pred) / y_true)), 2)
```
:::


::: {#723ae538 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
ArrayLike = np.ndarray|list|pd.Series
def SMAPE(
    y_true:ArrayLike,
    y_pred:ArrayLike,
) -> float:
    
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between actual and predicted values.
    
    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.

    Returns
    -------
    float
        The SMAPE value rounded to 2 decimal places.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1 / len(y_true) * np.sum(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)) * 100
    )
```
:::


::: {#d40468fe .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
ArrayLike = np.ndarray|list|pd.Series
def WMAPE(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    
    """"
    Calculate the Weighted Mean Absolute Percentage Error (WMAPE) between actual and predicted values.
    
    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.

    Returns
    -------
    float
        The WMAPE value.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
```
:::


::: {#904eef9e .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
ArrayLike = np.ndarray|list|pd.Series
def MAE(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    """
    Calculate mean absolute error (MAE).

    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.

    Returns
    -------
    float
        The MAE value.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))
```
:::


::: {#fb0bb535 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
ArrayLike = np.ndarray|list|pd.Series
def MSE(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    
    """
    Calculate mean squared error (MSE).

    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.

    Returns
    -------
    float
        The MSE value.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)
```
:::


::: {#e4e60878 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
ArrayLike = np.ndarray|list|pd.Series
def RMSE(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    
    """
    Calculate Root Mean Square Error (RMSE).
    
    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.
        
    Returns
    -------
    float
        The RMSE value.
    """
    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))
```
:::


::: {#453cc8ff .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
ArrayLike = np.ndarray|list|pd.Series
def SRMSE(y_true: ArrayLike,
          y_pred: ArrayLike,
          y_train: ArrayLike
          ) -> float:
    """
    Calculate Scaled Root Mean Square Error (SRMSE).

    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.
    y_train : ArrayLike
        The training values used for scaling.
        
    Returns
    -------
    float
        The SRMSE value.
    """
    # Ensure all arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")

    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)


    return np.sqrt(np.mean((y_true - y_pred) ** 2))/np.mean(y_train)

```
:::


::: {#1d873c4c .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
ArrayLike = np.ndarray|list|pd.Series
def RMSSE(y_true: ArrayLike,
          y_pred: ArrayLike,
          y_train: ArrayLike
          ) -> float:
    
    """
    Calculate Root Mean Squared Scaled Error (RMSSE).

    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.
    y_train : ArrayLike
        The training values used for scaling.

    Returns
    -------
    float
        The RMSSE value.
    """
    # Ensure all arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")

    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)

    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse / np.mean(np.diff(y_train) ** 2))
```
:::


::: {#848d0a05 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
ArrayLike = np.ndarray|list|pd.Series
def MASE(y_true: ArrayLike,
         y_pred: ArrayLike,
         y_train: ArrayLike
         ) -> float:
    
    """
    Calculate Mean Absolute Scaled Error (MASE)
    
    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.
    y_train : ArrayLike
        The training values used for scaling.
    
    Returns
    -------
    float
        The MASE value.
    """

    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)
    # Calculate the mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate the scaled error
    scaled_error = np.mean(np.abs(np.diff(y_train)))
    
    # Calculate MASE

    return mae / scaled_error

```
:::


::: {#d2752e7f .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
ArrayLike = np.ndarray|list|pd.Series
def CFE(y_true: ArrayLike,
        y_pred: ArrayLike
        ) -> float:
    
    """
    Calculate Cumulative Forecast Error (CFE). It is the cumulative sum of the differences between actual and predicted values.

    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.
    Returns
    -------
    float
        The CFE value.
    """
    return np.cumsum([a - f for a, f in zip(y_true, y_pred)])[-1]
```
:::


::: {#2b9c7ad0 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
ArrayLike = np.ndarray|list|pd.Series

def CFE_ABS(y_true: ArrayLike,
            y_pred: ArrayLike
            ) -> float:
    
    """
    Calculate Absolute Cumulative Forecast Error (CFE_ABS). It is the absolute value of the cumulative sum of the differences between actual and predicted values.
    
    Parameters
    ----------
    y_true : ArrayLike
        The actual values.
    y_pred : ArrayLike
        The predicted values.
        
    Returns
    -------
    float
        The absolute CFE value.
    """
    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Calculate cumulative forecast error
    cfe_t = np.cumsum([a - f for a, f in zip(y_true, y_pred)])
    return np.abs(cfe_t[-1])
```
:::


