# Dataset

The project downloads the **UCI Cleveland Heart Disease** processed file from the UCI
Machine Learning Repository. Do not manually edit generated CSV files.

```bash
python -m heart_disease_mlops.data
```

Generated files:

- `raw/processed.cleveland.data`: cached source file.
- `processed/heart_disease_cleaned.csv`: typed, domain-checked, binary-target dataset.
- `processed/data_summary.json`: source URL, UTC time, SHA-256, quality counts, and class balance.

The original diagnosis values 1 through 4 are mapped to `target=1`; value 0 maps to
`target=0`. Question marks and invalid domains become missing values. Imputation is fitted
inside the model pipeline and cross-validation folds to prevent data leakage.

| Column | Meaning | Treatment |
|---|---|---|
| `age` | Age in years | Median imputation, scaling |
| `sex` | Sex code | Mode imputation, one-hot encoding |
| `cp` | Chest pain type, 1-4 | Mode imputation, one-hot encoding |
| `trestbps` | Resting blood pressure | Median imputation, scaling |
| `chol` | Serum cholesterol | Median imputation, scaling |
| `fbs` | Fasting blood sugar indicator | Mode imputation, one-hot encoding |
| `restecg` | Resting ECG code | Mode imputation, one-hot encoding |
| `thalach` | Maximum heart rate | Median imputation, scaling |
| `exang` | Exercise-induced angina | Mode imputation, one-hot encoding |
| `oldpeak` | Exercise ST depression | Median imputation, scaling |
| `slope` | Peak exercise ST slope | Mode imputation, one-hot encoding |
| `ca` | Major vessel count | Mode imputation, one-hot encoding |
| `thal` | Thalassemia code | Mode imputation, one-hot encoding |
| `target` | Heart disease absent/present | Binary outcome |
