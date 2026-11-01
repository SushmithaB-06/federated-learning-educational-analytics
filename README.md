
# Federated Learning for Educational Analytics

## Project Overview

This project explores the application of Federated Learning
in educational analytics.

The objective is to develop a federated deep learning
framework where multiple educational institutions train
local models using their own student datasets without
sharing raw student records with a central server.

## Objectives

1. Develop a federated deep learning framework.
2. Investigate the impact of non-IID student data.
3. Compare federated aggregation strategies.
4. Evaluate model performance using standard metrics.

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Flower
- Matplotlib
- Streamlit

## Architecture

Multiple simulated educational clients independently
train local models and communicate model updates
to a central federated server.

The server aggregates model updates using Federated
Averaging (FedAvg) and distributes the global model.

## Project Status

Initial development and environment setup.


@'

## Development Progress

### Day 1 - Project Setup and Data Preprocessing

Completed:
- Initialized GitHub repository and project structure.
- Downloaded UCI Student Performance dataset.
- Explored 395 student records.
- Analyzed 33 original dataset columns.
- Created binary Pass/Fail target using G3 >= 10.
- Split dataset into 80% training and 20% testing.
- Applied One-Hot Encoding to categorical features.
- Applied StandardScaler to numerical features.
- Prepared 58 numerical input features.
- Verified dataset contains no missing values.

Dataset Summary:
- Total records: 395
- Training records: 316
- Testing records: 79
- Processed features: 58
- Passing students: 265
- Failing students: 130

Next:
- Simulate three educational clients.
- Implement IID and non-IID data partitioning.
'@ | Add-Content README.md