# House Price Prediction
This project uses Python and machine learning to predict house sale prices.


## Tools & Technologies
- **Python** — Programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical calculations
- **Scikit-learn** — Machine learning and model evaluation
- **SimpleImputer** — Handling missing values
- **OneHotEncoder** — Encoding categorical features
- **Pipeline and ColumnTransformer** — Building preprocessing workflows
- **Linear Regression and Random Forest** — Regression models
- **MAE, RMSE, and R²** — Model evaluation metrics
- **Cross-Validation** — Evaluating model performance across different data splits

## Project Workflow
1. Data Loading — Loaded the house price dataset using Pandas.
2. Feature and Target Selection — Selected SalePrice as the target and removed Id from the input features.
3. Train-Test Split — Split the dataset into training and testing sets using an 80/20 ratio.
4. Data Preprocessing — Handled missing numerical values using the median and missing categorical values using a placeholder. Applied One-Hot Encoding to categorical features.
5. Model Training — Trained three models:
  - Dummy Regressor as a baseline
  - Linear Regression
  - Random Forest Regressor
6. Model Evaluation — Evaluated predictions using MAE, RMSE, and R².
7. Cross-Validation — Used 5-fold cross-validation to evaluate Linear Regression and Random Forest.
8. Model Comparison — Compared training, testing, and cross-validation results to identify the best-performing model.

## Results
Three regression models were evaluated using MAE, RMSE, and R². Linear Regression and Random Forest were also evaluated using 5-fold cross-validation.

| Model                      |  Test MAE | Test RMSE | Test R² | Mean CV R² | CV R² Std |
| -------------------------- | --------  | --------  | ------  | ---------  | --------  |
| Dummy Regressor (Baseline) | 62,575.93 | 87,619.03 | -0.0009 |     —      |     —     |
|     Linear Regression      | 20,823.63 | 31,456.53 |  0.8710 |   0.8086   |  0.0600   |
|       Random Forest        | 17,591.96 | 28,992.73 |  0.8904 |   0.8428   |  0.0411   |



## Key Findings
* **Best-performing model:** Random Forest achieved the lowest test MAE and RMSE, and the highest test R² among the evaluated models.
* **Model performance:** Random Forest achieved a test R² of 0.8904, while Linear Regression achieved 0.8710.
* **Cross-validation:** Random Forest achieved a mean R² of 0.8428 across five folds, compared with 0.8086 for Linear Regression.
* **Baseline comparison:** Both regression models performed substantially better than the Dummy Regressor baseline.
* **Overfitting observation:** Random Forest achieved a training R² of 0.9799 compared with a test R² of 0.8904. This gap suggests some overfitting and indicates that further model tuning may be useful.

The evaluation results are saved in `model_comparison.csv`.

## Project Structure

```text
house-price-prediction/
├── analysis.py
├── README.md
├── requirements.txt
├── .gitignore
├── model_comparison.csv
└── data/
    └── train.csv (download separately)
```

- **`analysis.py`** — Main script for data preprocessing, model training, cross-validation, and evaluation.
- **`model_comparison.csv`** — Contains the model evaluation results.
- **`requirements.txt`** — Lists the required Python libraries.
- **`.gitignore`** — Specifies files and folders Git should ignore.
- **`data/train.csv`** — Dataset file that must be downloaded separately.
- **`README.md`** — Project overview, workflow, tools, and results.

## Dataset

This project uses the `train.csv` dataset from Kaggle's House Prices - Advanced Regression Techniques competition.

- **Source:** [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- **Target variable:** `SalePrice`
- **Expected local path:** `data/train.csv`

The dataset is not included in this repository. Download it from the competition page, follow the applicable terms, and place `train.csv` inside the `data` folder before running `analysis.py`.



