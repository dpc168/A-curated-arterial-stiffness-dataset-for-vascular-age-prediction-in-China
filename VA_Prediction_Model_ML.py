import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('TkAgg')
from scipy.integrate import simpson
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso



def load_excel_all_data(excel_file_path):
    # 读取Excel文件
    df = pd.read_excel(excel_file_path)
    # 假设文件中的'性别'列为0代表男性，1代表女性
    male_all_data = df[df['Sex'] == 0]  # 0表示男性
    female_all_data = df[df['Sex'] == 1]  # 1表示女性

    return male_all_data, female_all_data


def getfeature(data):
    data_Numpy = data.values #将data转换为Numpy数组
    Y = data_Numpy[1:,1]
    X = data_Numpy[1:,2:-1]

    return X, Y

# PCA dimensionality reduction function
def apply_pca(features, n_components=None):
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)

    pca = PCA(n_components=n_components)  # Apply PCA with specified number of components
    pca_features = pca.fit_transform(features_standardized)

    # Output PCA results
    print("\nPCA analysis results:")
    print("The proportion of variance explained by each principal component:", pca.explained_variance_ratio_)
    print("Cumulative explained variance proportion:", np.cumsum(pca.explained_variance_ratio_))
    print(f"The number of principal components to retain: {pca.n_components_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)[-1]:.4f}")

    return pca_features, pca

# Create a directory to save the images
RESULTS_DIR = r''
os.makedirs(RESULTS_DIR, exist_ok=True)
# Define general drawing functions
def plot_results(true_age, pred_age, model_name, color='#1f77b4'):
    plt.rcParams.update({'font.size': 12, 'savefig.dpi': 300})
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate evaluation metrics
    mae = mean_absolute_error(true_age, pred_age)
    rmse = np.sqrt(mean_squared_error(true_age, pred_age))
    r2 = r2_score(true_age, pred_age)
    r = np.corrcoef(true_age, pred_age)[0, 1]

    # Calculate the fitted line
    fit_model = LinearRegression()
    fit_model.fit(true_age.reshape(-1, 1), pred_age)
    fit_line = fit_model.predict(np.array([20, 80]).reshape(-1, 1))

    # Calculate the standard deviation of the error
    errors = pred_age - fit_model.predict(true_age.reshape(-1, 1))
    std_error = np.std(errors)

    # Draw error bands based on the fitted line (±1.96*std represents a 95% confidence interval)
    x = np.array([20, 80])
    y_fit = fit_model.predict(x.reshape(-1, 1))
    ax.fill_between(x,
                    y_fit - 1.96 * std_error,
                    y_fit + 1.96 * std_error,
                    color='lightgray', alpha=0.3,
                    label='95% confidence band')

    # Draw scatter points
    ax.scatter(true_age, pred_age, s=30, alpha=0.7, c=color)

    # draw the fitted line
    ax.plot([20, 80], fit_line, 'r-', linewidth=1.5,
            label=f'Fit line (Slope={fit_model.coef_[0]:.2f})')

    # Draw the ideal prediction line
    ax.plot([20, 80], [20, 80], 'k--', linewidth=1, label='Ideal Prediction')

    # Set the axis
    ax.set_xlim(20, 80)
    ax.set_ylim(20, 80)
    ax.set_xlabel('Chronological Age (Y)')
    ax.set_ylabel(f'Vascular Age ({model_name})')
    ax.set_title(f'{model_name} Model\n'
                 f'MAE={mae:.2f} years, RMSE={rmse:.2f} years, R²={r2:.2f}, R={r:.2f}')
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Add a legend
    ax.legend(loc='upper left')

    # Save the image
    filename = f"{RESULTS_DIR}/{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'R': r}


# Define the error distribution function
def plot_error_distribution(true_age, pred_age, model_name, color='#1f77b4'):
    plt.rcParams.update({'font.size': 12, 'savefig.dpi': 300})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Calculate the error
    errors = pred_age - true_age

    # Figure 1: Scatter plot of error as a function of age
    ax1.scatter(true_age, errors, s=30, alpha=0.7, c=color)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xlabel('Chronological Age (Y)')
    ax1.set_ylabel('Vascular Age (Y)')
    ax1.set_title(f'{model_name} - Error vs Age')
    ax1.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Add error trend line
    z = np.polyfit(true_age, errors, 1)
    p = np.poly1d(z)
    ax1.plot(true_age, p(true_age), "r--",
             label=f'Trend (Slope={z[0]:.3f})')
    ax1.legend()

    # Figure 2: Error histogram
    ax2.hist(errors, bins=15, color=color, alpha=0.7,
             edgecolor='black', density=True)
    ax2.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Prediction Error (Y)')
    ax2.set_ylabel('Density')
    ax2.set_title(f'{model_name} - Error Distribution')
    ax2.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Add a normal distribution curve
    mu, std = np.mean(errors), np.std(errors)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (std * np.sqrt(2 * np.pi)) *
         np.exp(-(x - mu) ** 2 / (2 * std ** 2)))
    ax2.plot(x, p, 'k', linewidth=2,
             label=f'Normal fit ($\mu$={mu:.2f}, $\sigma$={std:.2f})')
    ax2.legend()

    # Adjust the layout
    plt.tight_layout()

    # Save the image
    filename = f"{RESULTS_DIR}/{model_name.replace(' ', '_')}_error_dist.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

    return {
        'Mean Error': np.mean(errors),
        'Std Error': np.std(errors),
        'MAE': mean_absolute_error(true_age, pred_age),
        'RMSE': np.sqrt(mean_squared_error(true_age, pred_age))
    }

# Corrected KDM model
def KDM_Age(X, Y, X_test, Y_test):
    KDM_age = np.zeros((1, len(Y_test)))
    table_kdm = np.zeros((3, X.shape[1]))

    for i in range(X.shape[1]):
        xx = X[:, i].reshape(-1, 1)
        Y1 = Y.reshape(-1, 1)
        model = LinearRegression()
        model.fit(Y1, xx)
        residuals = model.predict(Y1) - xx
        indices = [index for index, value in enumerate(residuals) if
                   -2.3 * np.std(residuals) < value < 2.3 * np.std(residuals)]

        xx_new = xx[indices]
        yy_new = Y1[indices]
        model.fit(yy_new, xx_new)
        xx_new_pre = model.predict(yy_new)

        # Fix array to scalar conversion problem
        table_kdm[0, i] = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
        table_kdm[1, i] = model.coef_[0, 0] if model.coef_.ndim > 1 else model.coef_[0]
        table_kdm[2, i] = np.sqrt(mean_squared_error(xx_new, xx_new_pre))

    model1 = LinearRegression()
    model1.fit(X, Y)
    Y_pred = model1.predict(X)
    Sba = np.sqrt(mean_squared_error(Y, Y_pred))
    #print(f'KDMModel original features RMSE: {Sba:.4f}')

    for i in range(len(Y_test)):
        xxx = X_test[i, :]
        fenzi = np.dot((xxx - table_kdm[0, :]), (table_kdm[1, :] / (table_kdm[2, :] ** 2))) + Y_test[i] / (Sba ** 2)
        fenmu = np.dot(table_kdm[1, :] / table_kdm[2, :], table_kdm[1, :] / table_kdm[2, :]) + 1 / (Sba ** 2)
        KDM_age[0, i] = fenzi / fenmu

    return KDM_age

#  Main Function
def main():
    # Excel file path
    excel_file_path = r""  #

    male_all_data, female_all_data = load_excel_all_data(excel_file_path)
    # Original features

    X_male, Y_male = getfeature(male_all_data)
    # Male 训练与测试
    X_female, Y_female = getfeature(female_all_data)
    # Female 训练与测试

    # Apply PCA for both male and female data (let's keep 95% variance)
    X_male_pca, pca_male = apply_pca(X_male, n_components=0.95)
    X_female_pca, pca_female = apply_pca(X_female, n_components=0.95)

    # ==================== KDM  Model ====================
    print("\n========== KDM Model ==========")

    # KDM Model male
    KDM_male = KDM_Age(X_male_pca, Y_male, X_male_pca, Y_male)
    rmse_pca = np.sqrt(mean_squared_error(Y_male, KDM_male[0]))
    print(f'KDM  Model male_PCA RMSE: {rmse_pca:.4f}')
    plot_results(Y_male, KDM_male[0], 'KDM-Male', color='#1f77b4')
    error_stats = plot_error_distribution(Y_male, KDM_male[0], 'KDM-Male', color='#1f77b4')

    # KDM Model female
    KDM_female = KDM_Age(X_female_pca, Y_female, X_female_pca, Y_female)
    rmse_pca = np.sqrt(mean_squared_error(Y_female, KDM_female[0]))
    print(f'KDM  Model female_PCA RMSE: {rmse_pca:.4f}')
    plot_results(Y_female, KDM_female[0], 'KDM-Female', color='#1f77b4')
    error_stats = plot_error_distribution(Y_female, KDM_female[0], 'KDM-Female', color='#1f77b4')
    print("\n========== KDM Model End ==========")

    # Random Forest male (PCA)
    Y = Y_male
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_male_pca, Y, test_size=0.3, random_state=42)

    rf_model_pca = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=25)
    rf_model_pca.fit(X_train_pca, y_train_pca)
    y_pred_pca = rf_model_pca.predict(X_test_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_pca))
    print(f'Random Forest  Model male_PCA RMSE: {rmse_pca:.4f}')
    plot_results(y_test_pca, y_pred_pca, 'RF-Male', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_pca, y_pred_pca, 'RF-Male', color='#1f77b4')


    # Random Forest female (PCA)
    Y = Y_female
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_female_pca, Y, test_size=0.3, random_state=42)

    rf_model_pca = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=25)
    rf_model_pca.fit(X_train_pca, y_train_pca)
    y_pred_pca = rf_model_pca.predict(X_test_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_pca))
    print(f'Random Forest  Model female_PCA RMSE: {rmse_pca:.4f}')
    plot_results(y_test_pca, y_pred_pca, 'RF-Female', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_pca, y_pred_pca, 'RF-Female', color='#1f77b4')

    # ==================== Multiple Linear Regression Model ====================
    print("\n========== Multiple Linear Regression Model ==========")


    # Multiple Linear Regression Model Male (PCA)
    Y = Y_male
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_male_pca, Y, test_size=0.3, random_state=42)
    linear_reg_model_pca = LinearRegression()
    linear_reg_model_pca.fit(X_train_pca, y_train_pca)
    y_pred_mlr_pca = linear_reg_model_pca.predict(X_test_pca)
    rmse_mlr_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_mlr_pca))
    print(f'Multiple Linear Regression Model male_PCA RMSE:: {rmse_mlr_pca:.4f}')
    plot_results(y_test_pca, y_pred_mlr_pca, 'MLR-Male', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_pca, y_pred_mlr_pca, 'MLR-Male', color='#1f77b4')

    # Multiple Linear Regression Model Female (PCA)
    Y = Y_female
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_female_pca, Y, test_size=0.3, random_state=42)
    linear_reg_model_pca = LinearRegression()
    linear_reg_model_pca.fit(X_train_pca, y_train_pca)
    y_pred_mlr_pca = linear_reg_model_pca.predict(X_test_pca)
    rmse_mlr_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_mlr_pca))
    print(f'  Multiple Linear Regression Model female_PCA RMSE: {rmse_mlr_pca:.4f}')
    plot_results(y_test_pca, y_pred_mlr_pca, 'MLR-Female', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_pca, y_pred_mlr_pca, 'MLR-Female', color='#1f77b4')

    # ==================== LASSO Regression Model ====================
    print("\n========== LASSO Regression Model ==========")

    # LASSO Male (PCA)
    Y = Y_male
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_male_pca, Y, test_size=0.3, random_state=42)

    # Use grid search to find the best alpha parameter
    lasso = Lasso()
    parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    lasso_model_pca = GridSearchCV(lasso, parameters, cv=5)
    lasso_model_pca.fit(X_train_pca, y_train_pca)

    print(f"Best alpha for male: {lasso_model_pca.best_params_['alpha']}")

    y_pred_pca = lasso_model_pca.predict(X_test_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_pca))
    print(f'LASSO Model male_PCA RMSE: {rmse_pca:.4f}')
    plot_results(y_test_pca, y_pred_pca, 'LASSO-Male', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_pca, y_pred_pca, 'LASSO-Male', color='#1f77b4')

    # LASSO Female (PCA)
    Y = Y_female
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_female_pca, Y, test_size=0.3, random_state=42)

    lasso_model_pca = GridSearchCV(lasso, parameters, cv=5)
    lasso_model_pca.fit(X_train_pca, y_train_pca)

    print(f"Best alpha for female: {lasso_model_pca.best_params_['alpha']}")

    y_pred_pca = lasso_model_pca.predict(X_test_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_pca))
    print(f'LASSO Model female_PCA RMSE: {rmse_pca:.4f}')
    plot_results(y_test_pca, y_pred_pca, 'LASSO-Female', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_pca, y_pred_pca, 'LASSO-Female', color='#1f77b4')

    print("\n========== SVR Model ==========")


    # SVR Male (PCA)
    Y = Y_male
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_male_pca, Y, test_size=0.3, random_state=42)


    svr_model_pca = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model_pca.fit(X_train_pca, y_train_pca)
    y_pred_pca = svr_model_pca.predict(X_test_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_pca))
    print(f'SVR Model male_PCA RMSE: {rmse_pca:.4f}')
    plot_results(y_test_pca, y_pred_pca, 'SVR-Male', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_pca, y_pred_pca, 'SVR-Male', color='#1f77b4')

    # SVR Female (PCA )
    Y = Y_female
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_female_pca, Y, test_size=0.3, random_state=42)

    svr_model_pca.fit(X_train_pca, y_train_pca)
    y_pred_pca = svr_model_pca.predict(X_test_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_pca))
    print(f'SVR Model female_PCA RMSE: {rmse_pca:.4f}')
    plot_results(y_test_pca, y_pred_pca, 'SVR-Female', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_pca, y_pred_pca, 'SVR-Female', color='#1f77b4')

    #XGBoost
    print("\n========== XGBoost Model ==========")
    Y = Y_male
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_male_pca, Y, test_size=0.3, random_state=42)

    xgb_model_pca = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model_pca.fit(X_train_pca, y_train_pca)
    y_pred_pca = xgb_model_pca.predict(X_test_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_pca))
    print(f'XGBoost Model male_PCA RMSE: {rmse_pca:.4f}')
    plot_results(y_test_pca, y_pred_pca, 'XGB-Male', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_pca, y_pred_pca, 'XGB-Male', color='#1f77b4')

    # XGBoost Female (PCA)
    Y = Y_female
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_female_pca, Y, test_size=0.3, random_state=42)

    xgb_model_pca.fit(X_train_pca, y_train_pca)
    y_pred_pca = xgb_model_pca.predict(X_test_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_pca))
    print(f'XGBoost Model female_PCA RMSE: {rmse_pca:.4f}')
    plot_results(y_test_pca, y_pred_pca, 'XGB-Female', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_pca, y_pred_pca, 'XGB-Female', color='#1f77b4')

# run demo
if __name__ == "__main__":
    main()