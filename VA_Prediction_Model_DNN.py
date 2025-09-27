import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
from sklearn.linear_model import LinearRegression

# DNN model
class DNNModel(nn.Module):
    def __init__(self, input_size):
        super(DNNModel, self).__init__()
        self.hidden1 = nn.Linear(input_size, 128)  # First hidden layer
        self.hidden2 = nn.Linear(128, 64)  # Second hidden layer
        self.output = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))  # ReLU activation
        x = torch.relu(self.hidden2(x))  # ReLU activation
        x = self.output(x)  # Output layer
        return x


# Data loading and processing
def load_excel_all_data(excel_file_path):
    # Read Excel file
    df = pd.read_excel(excel_file_path)
    # Assume the column 'Sex': 0 = male, 1 = female
    male_all_data = df[df['Sex'] == 0]  # 0 for male
    female_all_data = df[df['Sex'] == 1]  # 1 for female
    return male_all_data, female_all_data


def getfeature(data):
    data_Numpy = data.values  # Convert DataFrame to NumPy array
    Y = data_Numpy[1:, 1]  # Target variable
    X = data_Numpy[1:, 2:-1]  # Feature variables
    return X, Y


def apply_pca(features, n_components=None):
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features_standardized)
    return pca_features, pca


# Train DNN model
def train_dnn_model(X_train, y_train, X_test, y_test, input_size, epochs=100, batch_size=64, lr=0.001, device='cpu'):
    # Convert data into PyTorch tensors and move to CPU/GPU
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                               torch.tensor(y_train, dtype=torch.float32).to(device))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                              torch.tensor(y_test, dtype=torch.float32).to(device))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize DNN model
    model = DNNModel(input_size).to(device)
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    # Train model
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs.squeeze(), targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Test model
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            y_pred.append(outputs.cpu().numpy())  # Move back to CPU

    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred

# Create a directory to save the images
RESULTS_DIR = r''
os.makedirs(RESULTS_DIR, exist_ok=True)

# General plotting function
def plot_results(true_age, pred_age, model_name, color='#1f77b4'):
    # Ensure pred_age is a 1D array
    pred_age = pred_age.squeeze()

    plt.rcParams.update({'font.size': 12, 'savefig.dpi': 300})
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate evaluation metrics
    mae = mean_absolute_error(true_age, pred_age)
    rmse = np.sqrt(mean_squared_error(true_age, pred_age))
    r2 = r2_score(true_age, pred_age)
    r = np.corrcoef(true_age, pred_age)[0, 1]

    # Fit regression line
    fit_model = LinearRegression()
    fit_model.fit(true_age.reshape(-1, 1), pred_age)
    fit_line = fit_model.predict(np.array([20, 80]).reshape(-1, 1))

    # Compute std error of residuals
    errors = pred_age - fit_model.predict(true_age.reshape(-1, 1))
    std_error = np.std(errors)

    # Draw error band (95% CI = ±1.96*std)
    x = np.array([20, 80])
    y_fit = fit_model.predict(x.reshape(-1, 1))
    ax.fill_between(x,
                    y_fit - 1.96 * std_error,
                    y_fit + 1.96 * std_error,
                    color='lightgray', alpha=0.3,
                    label='95% confidence band')

    # Scatter plot
    ax.scatter(true_age, pred_age, s=30, alpha=0.7, c=color)

    # Regression line
    ax.plot([20, 80], fit_line, 'r-', linewidth=1.5,
            label=f'Fit line (Slope={fit_model.coef_[0]:.2f})')

    # Ideal line
    ax.plot([20, 80], [20, 80], 'k--', linewidth=1, label='Ideal Prediction')

    # Axis settings
    ax.set_xlim(20, 80)
    ax.set_ylim(20, 80)
    ax.set_xlabel('Chronological Age (Y)')
    ax.set_ylabel(f'Vascular Age ({model_name})')
    ax.set_title(f'{model_name} Model\n'
                 f'MAE={mae:.2f} years, RMSE={rmse:.2f} years, R²={r2:.2f}, R={r:.2f}')
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Legend
    ax.legend(loc='upper left')

    # Save figure
    filename = f"{RESULTS_DIR}/{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"[{model_name}] MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, R: {r:.2f}")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'R': r}

# Plot error distribution
def plot_error_distribution(true_age, pred_age, model_name, color='#1f77b4'):
    true_age = np.array(true_age).squeeze()
    pred_age = np.array(pred_age).squeeze()
    plt.rcParams.update({'font.size': 12, 'savefig.dpi': 300})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Compute errors
    errors = pred_age - true_age

    # Figure 1: error vs age
    ax1.scatter(true_age, errors, s=30, alpha=0.7, c=color)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xlabel('Chronological Age (Y)')
    ax1.set_ylabel('Vascular Age (Y)')
    ax1.set_title(f'{model_name} - Error vs Age')
    ax1.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Trend line
    z = np.polyfit(true_age, errors, 1)
    p = np.poly1d(z)
    ax1.plot(true_age, p(true_age), "r--",
             label=f'Trend (Slope={z[0]:.3f})')
    ax1.legend()

    # Figure 2: error histogram
    ax2.hist(errors, bins=15, color=color, alpha=0.7,
             edgecolor='black', density=True)
    ax2.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Prediction Error (Y)')
    ax2.set_ylabel('Density')
    ax2.set_title(f'{model_name} - Error Distribution')
    ax2.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Normal distribution curve
    mu, std = np.mean(errors), np.std(errors)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (std * np.sqrt(2 * np.pi)) *
         np.exp(-(x - mu) ** 2 / (2 * std ** 2)))
    ax2.plot(x, p, 'k', linewidth=2,
             label=f'Normal fit ($\mu$={mu:.2f}, $\sigma$={std:.2f})')
    ax2.legend()

    # Layout
    plt.tight_layout()

    # Save figure
    filename = f"{RESULTS_DIR}/{model_name.replace(' ', '_')}_error_dist.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

    return {
        'Mean Error': np.mean(errors),
        'Std Error': np.std(errors),
        'MAE': mean_absolute_error(true_age, pred_age),
        'RMSE': np.sqrt(mean_squared_error(true_age, pred_age))
    }


# Main function
def main():
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    excel_file_path = r""
    male_all_data, female_all_data = load_excel_all_data(excel_file_path)

    # Extract male data
    X_male, Y_male = getfeature(male_all_data)
    X_female, Y_female = getfeature(female_all_data)

    # Apply PCA
    X_male_pca, pca_male = apply_pca(X_male, n_components=0.95)
    X_female_pca, pca_female = apply_pca(X_female, n_components=0.95)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_male_pca, Y_male, test_size=0.3, random_state=42)

    # Train DNN
    y_pred_dnn = train_dnn_model(X_train, y_train, X_test, y_test, X_male_pca.shape[1], device=device)

    # Plot results
    plot_results(y_test, y_pred_dnn, 'DNN-Male', color='#1f77b4')
    error_stats = plot_error_distribution(y_test, y_pred_dnn, 'DNN-Male', color='#1f77b4')

    # Repeat for female data
    X_train, X_test, y_train, y_test = train_test_split(X_female_pca, Y_female, test_size=0.3, random_state=42)
    y_pred_dnn_female = train_dnn_model(X_train, y_train, X_test, y_test, X_female_pca.shape[1], device=device)
    plot_results(y_test, y_pred_dnn_female, 'DNN-Female', color='#1f77b4')
    error_stats = plot_error_distribution(y_test, y_pred_dnn_female, 'DNN-Female', color='#1f77b4')


# Run main
if __name__ == "__main__":
    main()
