
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


## Project Progress

### 1. Dataset
- Dataset: UCI Student Performance Dataset (`student-mat.csv`)
- Total records: 395
- Prediction task: Student pass/fail classification
- Target variable: `pass`
- Pass: G3 >= 10
- Fail: G3 < 10
- Training samples: 316
- Testing samples: 79
- Encoded input features: 58

### 2. Data Preprocessing
- Explored the dataset using Pandas and NumPy.
- Checked for missing values.
- Created a binary pass/fail target.
- Used stratified train-test splitting.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Saved the processed dataset for model training.

### 3. Federated Client Simulation

Simulated three clients representing different educational institutions or student groups.

**IID partitioning**

| Client | Students | Pass | Fail |
|---|---:|---:|---:|
| Client 0 | 106 | 71 | 35 |
| Client 1 | 105 | 67 | 38 |
| Client 2 | 105 | 74 | 31 |

**Non-IID partitioning**

| Client | Students | Pass | Fail |
|---|---:|---:|---:|
| Client 0 | 106 | 90 | 16 |
| Client 1 | 105 | 70 | 35 |
| Client 2 | 105 | 52 | 53 |

IID clients have relatively similar class distributions, while Non-IID clients have deliberately different pass/fail distributions.

### 4. Neural Network Model

Implemented a feedforward neural network using PyTorch.

Architecture:
- Input layer: 58 features
- Hidden layer 1: 32 neurons with ReLU
- Hidden layer 2: 16 neurons with ReLU
- Output layer: 1 neuron
- Loss function: BCEWithLogitsLoss
- Optimizer: Adam
- Learning rate: 0.001

### 5. Federated Learning Implementation

Implemented Federated Averaging (FedAvg).

Training configuration:
- Number of clients: 3
- Federated rounds: 10
- Local epochs per round: 5
- Aggregation: Sample-weighted FedAvg

Each client trains its local model, and the server aggregates the client model parameters to update the global model.

### 6. Experimental Results

Evaluation on the same held-out test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Centralized | 84.81% | 91.84% | 84.91% | 88.24% |
| IID FedAvg | 87.34% | 93.88% | 86.79% | 90.20% |
| Non-IID FedAvg | 87.34% | 93.88% | 86.79% | 90.20% |

The IID and Non-IID FedAvg models produced identical predictions on this test set in the recorded run.

### 7. Convergence Analysis

Tracked the weighted average of clients' final local training losses over 10 federated rounds.

- IID FedAvg: Loss decreased from 0.6320 to 0.1443.
- Non-IID FedAvg: Loss decreased from 0.5918 to 0.1265.

The convergence graph is saved in `results/graphs/`.

Note: These values represent weighted final local training loss, not global test loss.

### 8. Current Implementation Status

Completed:
- Dataset exploration and preprocessing
- IID and Non-IID client partitioning
- PyTorch neural network
- Local client training
- FedAvg aggregation
- Federated training
- Convergence tracking and visualization
- Centralized vs federated evaluation
- Model comparison results

Planned:
- Additional aggregation strategies
- Multiple-run experiments
- Client-level performance analysis
- Communication overhead analysis
- Further evaluation and visualization
