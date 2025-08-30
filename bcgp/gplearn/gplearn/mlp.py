import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(3, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.relu(self.layer1(x)) # relu
        return self.layer2(x)

class MLPTrainer:
    def __init__(self, device="cpu", gpu_id=1):
        # self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        if torch.cuda.is_available() and device == 'cuda':
            self.device = torch.device(f"cuda:{gpu_id}")  # use specific GPU
        else:
            self.device = torch.device('cpu')  # Use CPU if CUDA is unavailable or device is not 'cuda'
        self.model = MLPModel().to(self.device)

    def train(self, X, y, epochs=10):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"[MLP] Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader):.6f}")

    def predict(self, X_input):
        self.model.eval()
        input_tensor = torch.tensor(X_input, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            return output.cpu().numpy()
