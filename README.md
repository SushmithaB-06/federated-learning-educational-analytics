
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
