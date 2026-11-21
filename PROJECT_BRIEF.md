# Planetary Defense Analytics: Classifying, Predicting and Clustering Asteroids

Source: `C:\Users\Chandrasekar A\Downloads\Planetary Defense Analytics.docx`
Domain: Astronomy, Space Science, Planetary Defense
Deadline: 12 days from the assigned date.
Dataset: NASA Small-Body Database export, 958,524 rows x 45 columns, one row per asteroid.

## Skills
EDA, cleaning, preprocessing, class imbalance, feature selection (RFE / tree importance / statistical tests), classification, regression, clustering, model evaluation, MLflow, Matplotlib/Seaborn/Plotly, SHAP, Streamlit, GitHub.

## Column roles
| Columns | Role |
|---|---|
| id, spkid, pdes, full_name, name, orbit_id | Identifiers. Drop before modeling. |
| neo | Existing label (q < 1.3 AU). Not a target. |
| pha | Case 1 target (MOID <= 0.05 AU and H <= 22.0). |
| H | Absolute magnitude. Predictor for diameter. |
| diameter (km) | Case 2 target. |
| albedo | Predictor for diameter. |
| e, a, i | Genuine predictors. |
| q | Defines NEO directly. Near-duplicate of moid. Leakage risk. |
| om, w, ma | Orbit angles, cyclic (0-360). Consider sin/cos encoding. |
| ad | ad = a(1+e). Can indirectly leak q. |
| n, per, per_y | Derived from a. |
| moid | Strong leakage for Case 1. Exclude when predicting pha. |
| class | Orbital family. Use to interpret clusters. Do not use as a feature. |

## Cases (all compulsory)
1. **Classification (pha):** at least 3 models (e.g. Logistic Regression, Random Forest, GB/SVM/KNN). Handle imbalance (class_weight and/or SMOTE) and compare with vs without.
2. **Regression (diameter):** at least 3 models (e.g. Linear, Random Forest Regressor, GB/SVR/KNN).
3. **Clustering (orbital elements):** K-Means (justify k via elbow/silhouette) and DBSCAN (optional, tune eps/min_samples via k-distance graph). Compare silhouette scores. Compare clusters with existing labels.

## Workflow
1. Dataset understanding
2. Cleaning (missing values, duplicates, invalid values)
3. EDA (univariate, bivariate, correlation heatmap, skew/outliers)
4. Preprocessing (skew: sqrt/cbrt/inverse/boxcox/log; outliers: IQR or Z-score; scaling; encoding)
5. Feature selection (compare at least 2 approaches, justify keep/drop)
6. Model building
7. Evaluation (precision/recall/F1 for imbalance, R2/RMSE/MAE, silhouette)
8. Model saving (.pkl/joblib) and MLflow tracking for Cases 1 and 2 (optional). Clustering: comparison table only.
9. SHAP for Cases 1 and 2
10. Streamlit app: inputs, PHA class, diameter, cluster, SHAP plot per prediction. Deploy to Streamlit Cloud.
11. GitHub repo with README (what, cases done, run locally, app link)

## Reference benchmarks
| Case | Metrics |
|---|---|
| 1 Classification | Accuracy 98.9%, Precision 15.8%, Recall 89.4%, F1 26.9% |
| 2 Regression | R2 0.96, RMSE 0.43, MAE 0.28 |
| 3 Clustering (a, e, i, per, moid_ld) | Silhouette about 0.35 |

## Deliverables
- Documented scripts/notebooks for all three cases
- EDA, feature selection and evaluation visuals
- Saved models for Cases 1 and 2, MLflow logs, Case 3 comparison table
- SHAP plots (shown in Streamlit)
- Deployed Streamlit app with shareable link
- GitHub repo with README
