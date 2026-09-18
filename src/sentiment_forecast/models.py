import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ridge_model(alpha):
    # All learned imputation/scaling statistics are fit inside each origin's training data.
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def baseline_predictions(trends, origin, horizon, issue):
    history = trends[
        (trends.index <= origin) & trends.available_at.le(issue)
    ].value.dropna()
    if history.empty:
        return {}
    target = origin + pd.Timedelta(weeks=horizon)
    seasonal_date = target - pd.Timedelta(weeks=52)
    predictions = {"naive": float(history.iloc[-1])}
    if seasonal_date in history.index:
        predictions["seasonal_naive"] = float(history.loc[seasonal_date])
    if len(history) > 1:
        span = (history.index[-1] - history.index[0]).days / 7
        distance = (target - history.index[-1]).days / 7
        predictions["drift"] = float(
            history.iloc[-1] + distance * (history.iloc[-1] - history.iloc[0]) / span
        )
    return predictions
