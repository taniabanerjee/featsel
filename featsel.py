import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class PairwiseCouplingLayer(nn.Module):
    def __init__(self, p):
        super(PairwiseCouplingLayer, self).__init__()
        # Each original feature and its knockoff has a pair of filter weights
        self.weights = nn.Parameter(torch.Tensor(p, 2))  # Z_j for original and Z̃_j for knockoff
        nn.init.constant_(self.weights, 0.1)  # Initialize filter weights to constant value 0.1

    def forward(self, X, X_knockoff):
        coupled = self.weights[:, 0] * X + self.weights[:, 1] * X_knockoff
        return coupled

class DeepPINK(nn.Module):
    def __init__(self, p):
        super(DeepPINK, self).__init__()

        # Plugin Pairwise Coupling Layer
        self.pairwise_layer = PairwiseCouplingLayer(p)

        # Multilayer Perceptron (MLP)
        self.fc1 = nn.Linear(p, p)  # First fully connected layer
        self.fc2 = nn.Linear(p, p)  # Second fully connected layer
        self.output_layer = nn.Linear(p, 1)  # Output layer, predicting y

    def forward(self, X, X_knockoff):
        coupled_features = self.pairwise_layer(X, X_knockoff)
        x = F.relu(self.fc1(coupled_features))
        x = F.relu(self.fc2(x))
        output = self.output_layer(x)
        return output

# Callback function to be called at the end of each epoch
def on_epoch_end(epoch, model, pVal, outputDir):
    if (epoch + 1) % 100 != 0:
        return

    with torch.no_grad():
        h_local1_weight = model.pairwise_layer.weights.detach().cpu().numpy()
        h0 = np.zeros((pVal, 2))
        h0_abs = np.zeros((pVal, 2))

        # Pairwise weight calculations
        for pIdx in range(pVal):
            h0[pIdx, :] = h_local1_weight[pIdx, :].flatten()
            h0_abs[pIdx, :] = np.fabs(h_local1_weight[pIdx, :]).flatten()

        # Weights of the fully connected layers
        h1 = model.fc1.weight.detach().cpu().numpy()
        h2 = model.fc2.weight.detach().cpu().numpy()
        h3 = model.output_layer.weight.detach().cpu().numpy()

        # Calculate feature importance based on layer weights
        W1 = h1
        W_curr = np.matmul(W1, h2)
        W3 = np.matmul(W_curr, h3.T)

        v0_h0 = h0[:, 0].reshape((pVal, 1))
        v1_h0 = h0[:, 1].reshape((pVal, 1))
        v0_h0_abs = h0_abs[:, 0].reshape((pVal, 1))
        v1_h0_abs = h0_abs[:, 1].reshape((pVal, 1))

        # Feature importance calculation
        v3 = np.vstack((np.sum((v0_h0_abs * np.fabs(W3)), axis=1).reshape((pVal, 1)),
                        np.sum((v1_h0_abs * np.fabs(W3)), axis=1).reshape((pVal, 1)))).T

        v5 = np.vstack((np.sum(v0_h0 * W3, axis=1).reshape((pVal, 1)),
                        np.sum(v1_h0 * W3, axis=1).reshape((pVal, 1)))).T

        # Save results to files
        with open(os.path.join(outputDir, f'result_epoch{epoch+1}_featImport.csv'), "a+") as myfile:
            myfile.write(','.join([str(x) for x in v3.flatten()]) + '\n')
        with open(os.path.join(outputDir, f'result_epoch{epoch+1}_featWeight.csv'), "a+") as myfile:
            myfile.write(','.join([str(x) for x in v5.flatten()]) + '\n')

def calc_selectedfeat(model, pVal, q_thres):
    h_local1_weight = model.pairwise_layer.weights.detach().cpu().numpy()
    h0 = np.zeros((pVal, 2))
    h0_abs = np.zeros((pVal, 2))

    # Pairwise weight calculations
    for pIdx in range(pVal):
        h0[pIdx, :] = h_local1_weight[pIdx, :].flatten()
        h0_abs[pIdx, :] = np.fabs(h_local1_weight[pIdx, :]).flatten()

    # Weights of the fully connected layers
    h1 = model.fc1.weight.detach().cpu().numpy()
    h2 = model.fc2.weight.detach().cpu().numpy()
    h3 = model.output_layer.weight.detach().cpu().numpy()

    # Calculate feature importance based on layer weights
    W1 = h1
    W_curr = np.matmul(W1, h2)
    W3 = np.matmul(W_curr, h3.T)

    v0_h0 = h0[:, 0].reshape((pVal, 1))
    v1_h0 = h0[:, 1].reshape((pVal, 1))
    v0_h0_abs = h0_abs[:, 0].reshape((pVal, 1))
    v1_h0_abs = h0_abs[:, 1].reshape((pVal, 1))
    W = (np.sum(np.square(v0_h0_abs * np.fabs(W3)), axis=1).reshape((pVal, 1))-np.sum(np.square(v1_h0_abs * np.fabs(W3)), axis=1).reshape((pVal, 1)))
    W = W.flatten(); print(W.shape)
    t = np.concatenate(([0], -np.sort(-np.fabs(W))))
    ratio = np.zeros(pVal + 1)
    for j in range(pVal + 1):
        ratio[j] = 1.0 * len(np.where(W <= -t[j])[0]) / np.max((1, len(np.where(W >= t[j])[0])))

    T = np.inf
    arr = np.where(ratio <= q_thres)[0]
    if len(arr) > 0:
        id = np.min(arr)
        T = t[id]

    qualifiedIndices = np.where(np.fabs(W) >= T)[0]
    return qualifiedIndices

# Function to calculate selected features (feature importance)
#def calc_selectedfeat_pytorch(model, X, X_knockoff, pVal, threshold=0.1):
#    model.eval()
#    with torch.no_grad():
#        # Forward pass through pairwise layer and MLP
#        coupled_features = model.pairwise_layer(X, X_knockoff)
#        feature_importances = coupled_features.cpu().numpy()
#
#        # Select features based on importance scores and threshold
#        selected_features = np.where(np.abs(feature_importances) > threshold)[0]
#        print(f"Selected features: {selected_features}")
#        return selected_features

# Training function
def train_model(model, X, X_knockoff, y, epochs, pVal, outputDir, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Assuming regression task

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X, X_knockoff)

        # Loss calculation
        loss = criterion(y_pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print loss and save model weights every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        # Call on_epoch_end
        on_epoch_end(epoch, model, pVal, outputDir)

    # Call feature selection at the end of training
    selected_features = calc_selectedfeat(model, pVal, 0.05)
    return selected_features

# Example usage:
p = 100  # Number of features
X = torch.randn(320, p)  # Original features
X_knockoff = torch.randn(320, p)  # Knockoff features
y = torch.randn(320, 1)  # Target labels
outputDir = './output'  # Directory to save results
epochs = 300  # Total number of epochs

model = DeepPINK(p=p)
print (model)
# Print the entire state_dict, which contains all weights and biases
for name, param in model.state_dict().items():
    print(f"Layer: {name} | Size: {param.size()}")
    print(param)  # This will print the actual tensor values (weights)

selected_features = train_model(model, X, X_knockoff, y, epochs, p, outputDir)
