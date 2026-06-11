# SaakhSetu - Yogyank Audit Baseline

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The local assessment environment already had the needed packages installed. The
fixed script does not require XGBoost.

## Run

Train the safer baseline and write artifacts:

```powershell
python fixed_yogyank_training.py
```

The script automatically uses `farmer_scoring_sample_yogyank_round1.csv` if it
exists, otherwise it uses the provided
`farmer_scoring_sample_yogyank_round1_final.csv`.

Outputs are written to `artifacts/`:

- `model_pipeline.joblib`: fitted preprocessing plus model pipeline
- `feature_schema.json`: scoring schema, feature list, excluded columns
- `model_version.json`: model version, data/script hash, package versions
- `validation_metrics.json`: out-of-time validation metrics and leakage check
- `slice_metrics.json`: validation residuals by monitoring slice
- `training_metadata.json`: training assumptions and validation setup
- `reason_code_reference.json`: training baselines for local reason codes
- `example_reason_codes.json`: sample score with top 3 reason codes

Score a single farmer record from JSON after training:

```powershell
python fixed_yogyank_training.py --score-json path\to\farmer_record.json
```

## Assumptions

- `defaulted_in_next_12_months` is a future outcome and is never used as a
  scoring feature.
- `application_year` is used only for out-of-time validation, not as a model
  feature.
- PM-Kisan status is treated as an observed model input in this baseline, but it
  is governance-sensitive and is explicitly monitored by slice.
- The model score is not a probability of default, credit decision, cutoff,
  grade, or bank eligibility rule.

## What Was Fixed

- Removed future-outcome leakage from the feature set.
- Removed the hard-coded PM-Kisan target penalty from model training.
- Replaced pre-split label encoding with a single sklearn `Pipeline` and
  `ColumnTransformer`.
- Used one-hot encoding with `handle_unknown="ignore"` for categorical features.
- Switched validation from random shuffle split to an out-of-time split:
  train on 2022-2023, validate on 2024.
- Saved the full pipeline, schema, version metadata, metrics, slice diagnostics,
  and reason-code reference artifacts.
- Added local top-3 reason codes using deterministic feature-baseline
  perturbations.

## Validation Approach

The fixed run uses the latest `application_year` as validation. On the provided
data this means:

- Train rows: 3,606 from 2022-2023
- Validation rows: 1,394 from 2024
- Validation R2: 0.2763
- Validation RMSE: 127.1105
- Validation MAE: 80.6470

The lower validation score is expected after removing leakage. It is more
credible than the original near-perfect result because preprocessing is fit only
on training rows and validation simulates future applications.

## Skipped Due To Time

- No hyperparameter search or model comparison.
- No causal or legally validated explainability method.
- No production monitoring service, model registry, or CI/CD checks.
- No independent label-quality review of the synthetic target.
- No policy-layer implementation for bank-specific cutoffs or grade mappings.
