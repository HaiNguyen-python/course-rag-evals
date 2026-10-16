# MLOps Basics

MLOps applies software engineering and DevOps practices to machine learning, so that models can be trained, deployed and maintained reliably.

## Experiment tracking

Every training run should record its parameters, data version, metrics and resulting model. Tools such as MLflow store this information so that results can be compared and reproduced later.

## Model registry

A model registry keeps versions of trained models and marks which version is used in production. In MLflow, an alias such as champion can point to the production version. Rolling back means moving the alias to an earlier version.

## Validation before deployment

A new model should be compared with a simple baseline and with the current production model on the same held-out data before it is promoted. For time series, the holdout must come after the training period in time, otherwise the evaluation leaks information from the future.

## Monitoring in production

After deployment, teams monitor input data drift and prediction quality. The Population Stability Index (PSI) is a common way to measure how much the distribution of a feature has shifted. As a rule of thumb, a PSI below 0.1 means little change and above 0.25 means a significant shift. When true outcomes become available, the error metrics should be tracked over time and an alert raised when they pass a threshold.
