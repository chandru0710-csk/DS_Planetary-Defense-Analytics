# Planetary Defense Analytics

Machine learning on the NASA JPL Small-Body Database (958,524 asteroids). The project:

1. **Classifies** Potentially Hazardous Asteroids (PHA)
2. **Predicts** asteroid diameter
3. **Clusters** asteroids into orbital groups

**Live app:** _add your Streamlit link here_

## Cases completed

| Case | Task | Best model | Result (test set) |
|---|---|---|---|
| 1 | Classification (`pha`) | XGBoost + SMOTE | Recall 95.88%, Precision 35.33%, F1 51.63%, ROC-AUC 0.9987 |
| 2 | Regression (`diameter`) | XGBoost | R² 0.9637 (log diameter), RMSE 0.57 km, MAE 0.35 km |
| 3 | Clustering (a, e, i, per, moid_ld) | K-Means, k = 3 | Silhouette 0.72 (sample) |

- Case 1 compares 3 models (Logistic Regression, Random Forest, XGBoost) × 3 balancing options (none, class_weight, SMOTE). The leakage columns (moid, q, ad, neo, class) are excluded.
- Case 2 compares Linear Regression, Random Forest and XGBoost. 5-fold CV R² is 0.9636 ± 0.0009.
- Case 3 uses K-Means, with k chosen by the elbow method and silhouette score for k = 2–10. The three clusters are a near-Earth group (contains all PHAs), a main belt group and an outer solar system group.
- SHAP explanations for Cases 1 and 2. MLflow tracking for all 12 runs.

## Project structure

```
1.DataCleaning.ipynb        load, clean, EDA
2.Data_preprocessing.ipynb  encoding, skewness, outliers, feature selection
3.Classification.ipynb      Case 1 + MLflow + SHAP
4.Regression.ipynb          Case 2 + MLflow + SHAP
5.Clustering.ipynb          Case 3
app.py                      Streamlit app
models/                     saved models (.pkl)
results/                    comparison tables (.csv)
images/                     plots
requirements.txt
```

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

To rerun the notebooks, put `pha_dataset.csv` in the project folder and run notebooks 1 to 5 in order. The notebooks also need `imbalanced-learn`, `mlflow`, `seaborn` and `scipy`.

To view the MLflow runs (the repo includes `mlflow.db` with all parameters and metrics; the model artifacts in `mlruns/` are not pushed because of their size):

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
