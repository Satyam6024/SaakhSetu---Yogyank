# Yogyank Baseline Audit Memo

The original script was dangerous because it trained on information that would
not exist when scoring a farmer in the future. Most critically, it allowed the
future default outcome `defaulted_in_next_12_months` to influence model design
and that field has a correlation of about -0.89 with the target score in the
provided sample. A high validation R2 from that setup is not trustworthy because
the model can learn the answer from a future event. The script also changed the
training target by subtracting 150 points for farmers without PM-Kisan status,
which mixes business policy into the model target. In addition, it fit label
encoders before splitting the data, reused one encoder across multiple
categorical fields, used a random split despite having `application_year`, and
saved only a model file without the preprocessing, schema, metrics, or version
metadata needed for reproducible scoring.

I replaced the draft with `fixed_yogyank_training.py`, a leakage-conscious
baseline built as a single sklearn pipeline. The fixed version excludes
`defaulted_in_next_12_months`, uses `application_year` only for an out-of-time
validation split, and does not alter the target with PM-Kisan or any other
policy rule. Preprocessing is fit only on training rows through a
`ColumnTransformer`: numeric fields get median imputation and categorical fields
get one-hot encoding with unknown-category handling. The run trains on 2022-2023
applications and validates on 2024 applications. It saves the full pipeline,
feature schema, version stamp, data and script hashes, validation metrics,
fairness/stability slice diagnostics, and reason-code baselines. Scoring remains
a bank-agnostic entitlement score; cutoffs, grades, and eligibility decisions are
left for a separate versioned policy layer.

## Limitations

I would not yet trust this model for production credit-impacting decisions. The
out-of-time R2 is only 0.2763 with MAE around 80.65 points, so the baseline is
useful for audit review but not strong enough for deployment. With more time, I
would review target construction with domain owners, test feature availability
at real scoring time, compare simpler and more robust models, add confidence
intervals, and validate reason codes with credit and compliance reviewers.

## Monitoring

After deployment, I would monitor fairness and stability at minimum by crop
type, district, PM-Kisan status, irrigation type, and landholding band. For each
slice I would track score distributions, prediction error once outcomes or
review labels mature, approval/decline rates from the separate policy layer,
missing-value rates, unknown-category rates, and drift against the training
baseline.
