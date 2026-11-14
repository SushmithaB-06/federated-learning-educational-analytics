
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    X,
    y,
    model=None,
    epochs=20,
    learning_rate=0.001,
    return_history=False
):

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    if model is None:
        from src.models.neural_network import StudentPerformanceModel
        model = StudentPerformanceModel(input_size=58)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        history.append(average_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}")

    if return_history:
        return model, history

    return model
