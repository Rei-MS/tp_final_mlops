[← Back to Root Documentation](../README.md)

# `api`

This directory contains the FastAPI microservice that serves passenger satisfaction predictions. The service dynamically fetches the latest champion model directly from the MLflow Model Registry and enforces input feature validation against the training scheme.

## Structure

```text
.
├── main.py             # FastAPI application, pre-processing, and endpoints
├── Dockerfile          # Container build definition
├── requirements.txt    # Python dependencies
└── README.md
```