[← Back to Root Documentation](../README.md)

# `src`

This directory contains the project's model related logic; dataset preprocessing, hyperparameter optimization, experiment tracking and model registration.

## Structure

```text
.
├── config.py          # Configuration variables
├── dataload.py        # Dataset loading and processing
├── metrics.py         # Metric calculation and related figures
├── train.py           # Model training, optuna search, MLflow logging
├── tracking.py        # MLflow logging (features and figures)
├── registry.py        # MLflow model registry (alias assignment)
├── main.py            # Main execution - run from root.
└── README.md
```