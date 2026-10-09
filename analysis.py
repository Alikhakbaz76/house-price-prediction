import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "train.csv"
train = pd.read_csv(DATA_PATH)
y = train['SalePrice']
X = train.drop(columns=['SalePrice', 'Id'])
X['MSSubClass'] = X['MSSubClass'].astype(str)
X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size=0.2 , random_state=42)
cv_strategy = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
numerical_features = X_train.select_dtypes(include='number').columns
categorical_features = X_train.select_dtypes(
    include=['object', 'string']
).columns
def make_preprocessor():

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(
            strategy='constant',
            fill_value='Missing'
        )),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

def evaluate_regression(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }    

full_pipeline = Pipeline([
    ('preprocessor', make_preprocessor()),
    ('regressor', LinearRegression())
])
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
linear_test_metrics = evaluate_regression(y_test, y_pred)

y_train_pred = full_pipeline.predict(X_train)

linear_train_metrics = evaluate_regression(
    y_train, y_train_pred
)
baseline_model = DummyRegressor(strategy='mean')
baseline_model.fit(X_train, y_train)
baseline_test_pred = baseline_model.predict(X_test)
baseline_train_pred = baseline_model.predict(X_train)

baseline_test_metrics = evaluate_regression(
    y_test, baseline_test_pred
)

baseline_train_metrics = evaluate_regression(
    y_train, baseline_train_pred
)
cv_linear_scores = cross_val_score(
    full_pipeline,
    X_train,
    y_train,
    cv=cv_strategy,
    scoring='r2'
)
rf_pipeline = Pipeline([('preprocessor', make_preprocessor()),
                        ('regressor',
                         RandomForestRegressor(
                             n_estimators=200,
                             random_state=42,
                             n_jobs=-1
                             ))])
rf_pipeline.fit(X_train , y_train)
rf_pred = rf_pipeline.predict(X_test)
rf_train_pred = rf_pipeline.predict(X_train)
rf_test_metrics = evaluate_regression(y_test, rf_pred)

rf_train_metrics = evaluate_regression(
    y_train, rf_train_pred
)
cv_rf_scores = cross_val_score(
    rf_pipeline,
    X_train,
    y_train,
    cv=cv_strategy,
    scoring='r2'
)

results = pd.DataFrame([
    {
        'Model': 'Baseline',
        'Train_MAE': baseline_train_metrics['MAE'],
        'Test_MAE': baseline_test_metrics['MAE'],
        'Train_RMSE': baseline_train_metrics['RMSE'],
        'Test_RMSE': baseline_test_metrics['RMSE'],
        'Train_R2': baseline_train_metrics['R2'],
        'Test_R2': baseline_test_metrics['R2'],
        'CV_R2_Mean': np.nan,
        'CV_R2_Std': np.nan
    },
    {
        'Model': 'Linear Regression',
        'Train_MAE': linear_train_metrics['MAE'],
        'Test_MAE': linear_test_metrics['MAE'],
        'Train_RMSE': linear_train_metrics['RMSE'],
        'Test_RMSE': linear_test_metrics['RMSE'],
        'Train_R2': linear_train_metrics['R2'],
        'Test_R2': linear_test_metrics['R2'],
        'CV_R2_Mean': cv_linear_scores.mean(),
        'CV_R2_Std': cv_linear_scores.std()
    },
    {
        'Model': 'Random Forest',
        'Train_MAE': rf_train_metrics['MAE'],
        'Test_MAE': rf_test_metrics['MAE'],
        'Train_RMSE': rf_train_metrics['RMSE'],
        'Test_RMSE': rf_test_metrics['RMSE'],
        'Train_R2': rf_train_metrics['R2'],
        'Test_R2': rf_test_metrics['R2'],
        'CV_R2_Mean': cv_rf_scores.mean(),
        'CV_R2_Std': cv_rf_scores.std()
    }
])

print('\nModel Comparison')
print(results.round(4).to_string(index=False))
OUTPUT_PATH = PROJECT_DIR / "model_comparison.csv"
results.to_csv(OUTPUT_PATH, index=False)


