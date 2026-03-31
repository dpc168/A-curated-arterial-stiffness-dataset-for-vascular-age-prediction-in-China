import os
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as stats
import scikit_posthocs as sp
import shap

from itertools import combinations
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# =========================
# 0. Random Seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# 1. 数据读取与预处理
# Sex coding: 0 = Female, 1 = Male
# =========================
def load_excel_all_data(excel_file_path):
    df = pd.read_excel(excel_file_path)
    female_all_data = df[df['Sex'] == 0].copy()
    male_all_data = df[df['Sex'] == 1].copy()
    return male_all_data, female_all_data


def getfeature(data):
    Y = pd.to_numeric(data.iloc[:, 1], errors='coerce')
    X = data.iloc[:, 2:].copy()
    X = X.drop(columns=["Label", "Sex"], errors="ignore")
    X = X.loc[:, ~X.columns.astype(str).str.contains(r"^Unnamed", case=False, na=False)]
    X = X.apply(pd.to_numeric, errors='coerce')
    return X, Y


def clean_features_and_target(X, Y, sex_name="Unknown"):
    X = pd.DataFrame(X).copy()
    Y = pd.Series(Y).copy()

    valid_y = Y.notna()
    X = X.loc[valid_y].reset_index(drop=True)
    Y = Y.loc[valid_y].reset_index(drop=True)

    X = X.apply(pd.to_numeric, errors="coerce")
    Y = pd.to_numeric(Y, errors="coerce")

    X = X.dropna(axis=1, how="all")

    nunique = X.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if len(const_cols) > 0:
        print(f"[{sex_name}] Dropping constant columns: {const_cols}")
        X = X.drop(columns=const_cols, errors="ignore")

    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna(axis=1, how="any")

    valid_rows = ~X.isna().any(axis=1)
    X = X.loc[valid_rows].reset_index(drop=True)
    Y = Y.loc[valid_rows].reset_index(drop=True)

    print(f"[{sex_name}] Cleaned X shape: {X.shape}, Y shape: {Y.shape}")
    return X.to_numpy(dtype=float), Y.to_numpy(dtype=float), list(X.columns)


# =========================
# 2. SHAP Feature Selection
# =========================
def select_features_by_shap(
    X_train, y_train, X_val=None, X_test=None,
    feature_names=None, sex_name="Unknown", model_name="Selector",
    results_dir="results", shap_threshold=1.5,
    min_features=5, max_features=None, random_state=42
):
    os.makedirs(results_dir, exist_ok=True)

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    selector_model = XGBRegressor(
        random_state=random_state,
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    selector_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(selector_model)
    shap_values = np.asarray(explainer.shap_values(X_train))
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze()

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    selected_idx = [i for i in sorted_idx if mean_abs_shap[i] > shap_threshold]
    if len(selected_idx) < min_features:
        selected_idx = sorted_idx[:min_features].tolist()
    if max_features is not None:
        selected_idx = selected_idx[:max_features]

    selected_features = [feature_names[i] for i in selected_idx]

    summary_df = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values("MeanAbsSHAP", ascending=False)
    summary_df.to_csv(
        os.path.join(results_dir, f"SHAP_Feature_Importance_{model_name}-{sex_name}.csv"),
        index=False, encoding="utf-8-sig"
    )

    selected_df = pd.DataFrame({
        "Rank": np.arange(1, len(selected_idx) + 1),
        "Feature": selected_features,
        "MeanAbsSHAP": mean_abs_shap[selected_idx]
    })
    selected_df.to_csv(
        os.path.join(results_dir, f"SHAP_Selected_Features_{model_name}-{sex_name}.csv"),
        index=False, encoding="utf-8-sig"
    )

    plt.figure(figsize=(10, 8))
    top_plot_k = min(20, len(sorted_idx))
    plot_idx = sorted_idx[:top_plot_k][::-1]
    plt.barh([feature_names[i] for i in plot_idx], mean_abs_shap[plot_idx])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top SHAP Features - {model_name}-{sex_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"SHAP_TopFeatures_{model_name}-{sex_name}.png"),
        dpi=600, bbox_inches="tight"
    )
    plt.close()

    print(f"[{sex_name}][{model_name}] Selected {len(selected_features)} features: {selected_features}")

    X_train_sel = X_train[:, selected_idx]
    X_val_sel = X_val[:, selected_idx] if X_val is not None else None
    X_test_sel = X_test[:, selected_idx] if X_test is not None else None

    return X_train_sel, X_val_sel, X_test_sel, selected_idx, selected_features, selector_model


# =========================
# 3. Basic Function
# =========================
def safe_corrcoef(y_true, y_pred):
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        return np.corrcoef(y_true, y_pred)[0, 1]
    return np.nan


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    r = safe_corrcoef(y_true, y_pred)

    lr = LinearRegression().fit(y_true.reshape(-1, 1), y_pred)
    slope = float(lr.coef_[0])
    intercept = float(lr.intercept_)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "R": r,
        "Bias": float(np.mean(error)),
        "Error_SD": float(np.std(error)),
        "Slope": slope,
        "Intercept": intercept
    }


def build_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def plot_results(true_age, pred_age, model_name, results_dir, color='#1f77b4'):
    true_age = np.asarray(true_age).squeeze()
    pred_age = np.asarray(pred_age).squeeze()

    plt.rcParams.update({'font.size': 12, 'savefig.dpi': 600})
    fig, ax = plt.subplots(figsize=(8, 8))

    mae = mean_absolute_error(true_age, pred_age)
    rmse = np.sqrt(mean_squared_error(true_age, pred_age))
    r2 = r2_score(true_age, pred_age)
    r = safe_corrcoef(true_age, pred_age)

    fit_model = LinearRegression()
    fit_model.fit(true_age.reshape(-1, 1), pred_age)

    xmin = min(true_age.min(), pred_age.min())
    xmax = max(true_age.max(), pred_age.max())
    margin = max(1.0, 0.05 * (xmax - xmin))
    x_line = np.array([xmin, xmax])
    fit_line = fit_model.predict(x_line.reshape(-1, 1))

    errors = pred_age - fit_model.predict(true_age.reshape(-1, 1))
    std_error = np.std(errors)

    y_fit = fit_model.predict(x_line.reshape(-1, 1))
    ax.fill_between(
        x_line,
        y_fit - 1.96 * std_error,
        y_fit + 1.96 * std_error,
        color='lightgray',
        alpha=0.3,
        label='95% confidence band'
    )

    ax.scatter(true_age, pred_age, s=30, alpha=0.7, c=color)
    ax.plot(x_line, fit_line, 'r-', linewidth=1.5, label=f'Fit line (Slope={fit_model.coef_[0]:.2f})')
    ax.plot(x_line, x_line, 'k--', linewidth=1, label='Ideal Prediction')

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(xmin - margin, xmax + margin)
    ax.set_xlabel('Chronological Age (Y)')
    ax.set_ylabel(f'Vascular Age ({model_name})')
    ax.set_title(f'{model_name}\nMAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}, R={r:.2f}')
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left')

    filename = os.path.join(results_dir, f"{model_name.replace(' ', '_')}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=600)
    plt.close()

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'R': r}


def plot_error_distribution(true_age, pred_age, model_name, results_dir, color='#1f77b4'):
    true_age = np.asarray(true_age).squeeze()
    pred_age = np.asarray(pred_age).squeeze()

    plt.rcParams.update({'font.size': 12, 'savefig.dpi': 600})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    errors = pred_age - true_age

    ax1.scatter(true_age, errors, s=30, alpha=0.7, c=color)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xlabel('Chronological Age (Y)')
    ax1.set_ylabel('Prediction Error (Y)')
    ax1.set_title(f'{model_name} - Error vs Age')
    ax1.grid(color='lightgray', linestyle='--', linewidth=0.5)

    if len(true_age) >= 2:
        z = np.polyfit(true_age, errors, 1)
        p = np.poly1d(z)
        sort_idx = np.argsort(true_age)
        ax1.plot(true_age[sort_idx], p(true_age[sort_idx]), "r--", label=f'Trend (Slope={z[0]:.3f})')
        ax1.legend()

    ax2.hist(errors, bins=15, color=color, alpha=0.7, edgecolor='black', density=True)
    ax2.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Prediction Error (Y)')
    ax2.set_ylabel('Density')
    ax2.set_title(f'{model_name} - Error Distribution')
    ax2.grid(color='lightgray', linestyle='--', linewidth=0.5)

    mu, std = np.mean(errors), np.std(errors)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    if std > 0:
        p_norm = (1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * std ** 2)))
        ax2.plot(x, p_norm, 'k', linewidth=2, label=f'Normal fit ($\\mu$={mu:.2f}, $\\sigma$={std:.2f})')
    ax2.legend()

    plt.tight_layout()
    filename = os.path.join(results_dir, f"{model_name.replace(' ', '_')}_error_dist.png")
    plt.savefig(filename, bbox_inches='tight', dpi=600)
    plt.close()

    return {
        'Mean Error': np.mean(errors),
        'Std Error': np.std(errors),
        'MAE': mean_absolute_error(true_age, pred_age),
        'RMSE': np.sqrt(mean_squared_error(true_age, pred_age))
    }


def plot_bland_altman(y_true, y_pred, model_name, results_dir):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mean_val = (y_true + y_pred) / 2
    diff = y_pred - y_true

    bias = np.mean(diff)
    sd = np.std(diff)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    plt.figure(figsize=(8, 6))
    plt.scatter(mean_val, diff, alpha=0.6, s=20)
    plt.axhline(bias, linestyle="--", linewidth=1)
    plt.axhline(loa_upper, linestyle="--", linewidth=1)
    plt.axhline(loa_lower, linestyle="--", linewidth=1)
    plt.xlabel("Mean of True and Predicted Age")
    plt.ylabel("Prediction Error (Pred - True)")
    plt.title(f"Bland-Altman: {model_name}\nBias={bias:.2f}, LoA=({loa_lower:.2f}, {loa_upper:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_bland_altman.png"), dpi=600, bbox_inches="tight")
    plt.close()


def plot_mean_ranks(mean_ranks, metric_name, sex_name, results_dir):
    mean_ranks = mean_ranks.sort_values()

    plt.figure(figsize=(8, 4))
    y = np.arange(len(mean_ranks))
    plt.scatter(mean_ranks.values, y, s=50)
    for i, (model, rank) in enumerate(mean_ranks.items()):
        plt.text(rank + 0.02, i, model, va="center")

    plt.yticks([])
    plt.xlabel("Average Rank")
    plt.title(f"Average Rank of Models ({metric_name}, {sex_name})")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_png = os.path.join(results_dir, f"MeanRankPlot_{metric_name}_{sex_name}.png")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()


def plot_metric_boxplot(metrics_df, sex_name, metric_name, results_dir):
    sub = metrics_df[metrics_df["Sex"] == sex_name].copy()
    models = sorted(sub["Model"].unique())
    data = [sub[sub["Model"] == m][metric_name].values for m in models]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=models)
    plt.xticks(rotation=45)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} across repeated splits ({sex_name})")
    plt.tight_layout()

    out_png = os.path.join(results_dir, f"Boxplot_{metric_name}_{sex_name}.png")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()


# =========================
# 4. KDM
# =========================
def KDM_Age(X, Y, X_test, Y_test, feature_names=None, verbose=False):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    X_test = np.asarray(X_test, dtype=float)
    Y_test = np.asarray(Y_test, dtype=float).reshape(-1)

    KDM_age = np.zeros((1, len(Y_test)))
    table_kdm = np.zeros((3, X.shape[1]))

    for i in range(X.shape[1]):
        feat_name = feature_names[i] if feature_names is not None else f"Feature_{i}"
        xx = X[:, i].reshape(-1, 1)
        Y1 = Y.reshape(-1, 1)

        model = LinearRegression()
        model.fit(Y1, xx)

        residuals = (model.predict(Y1) - xx).reshape(-1)
        res_std = np.std(residuals)

        if (not np.isfinite(res_std)) or (res_std < 1e-12):
            xx_new = xx
            yy_new = Y1
            kept_n = len(yy_new)
        else:
            mask = np.abs(residuals) <= 2.3 * res_std
            kept_n = int(np.sum(mask))
            if kept_n < 3:
                xx_new = xx
                yy_new = Y1
                kept_n = len(yy_new)
            else:
                xx_new = xx[mask]
                yy_new = Y1[mask]

        model.fit(yy_new, xx_new)
        xx_new_pre = model.predict(yy_new)

        intercept = model.intercept_[0] if np.ndim(model.intercept_) > 0 else model.intercept_
        coef = model.coef_[0, 0] if model.coef_.ndim > 1 else model.coef_[0]
        rmse_feat = np.sqrt(mean_squared_error(xx_new, xx_new_pre))
        rmse_feat = max(rmse_feat, 1e-8)

        table_kdm[0, i] = intercept
        table_kdm[1, i] = coef
        table_kdm[2, i] = rmse_feat

        if verbose:
            print(f"[KDM] {feat_name}: kept_samples={kept_n}, rmse_feat={rmse_feat:.8f}")

    model1 = LinearRegression()
    model1.fit(X, Y)
    Y_pred = model1.predict(X)
    Sba = np.sqrt(mean_squared_error(Y, Y_pred))
    Sba = max(Sba, 1e-8)

    for i in range(len(Y_test)):
        xxx = X_test[i, :]
        fenzi = np.dot((xxx - table_kdm[0, :]), (table_kdm[1, :] / (table_kdm[2, :] ** 2))) + Y_test[i] / (Sba ** 2)
        fenmu = np.dot(table_kdm[1, :] / table_kdm[2, :], table_kdm[1, :] / table_kdm[2, :]) + 1 / (Sba ** 2)
        fenmu = max(fenmu, 1e-8)
        KDM_age[0, i] = fenzi / fenmu

    return KDM_age


# =========================
# 5. SHAP explainability helpers
# =========================
def build_shap_importance_df(shap_values, feature_names, sex_name, model_name, repeat_id):
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze()
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    df = pd.DataFrame({
        "Sex": sex_name,
        "Repeat": repeat_id,
        "Model": model_name,
        "Feature": np.array(feature_names),
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

    df["Rank"] = np.arange(1, len(df) + 1)
    return df


def save_shap_values_to_txt(shap_values, feature_names, file_path, sample_prefix="Sample", top_n_summary=20):
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("========== SHAP Global Importance ==========" + "\n")
        for rank, idx in enumerate(sorted_idx[:min(top_n_summary, len(sorted_idx))], start=1):
            f.write(f"Rank {rank}: {feature_names[idx]} | mean(|SHAP|) = {mean_abs_shap[idx]:.8f}\n")

        f.write("\n\n========== SHAP Values For Each Sample ==========" + "\n")
        for i in range(shap_values.shape[0]):
            f.write(f"\n--- {sample_prefix}_{i+1} ---\n")
            for j, feat in enumerate(feature_names):
                f.write(f"{feat}: {shap_values[i, j]:.8f}\n")


def save_local_shap_plots(
    shap_values, X_explain, feature_names, results_dir, model_name, sex_name,
    base_values=None, top_n=3
):
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze()
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    X_explain = np.asarray(X_explain, dtype=float)

    sample_scores = np.mean(np.abs(shap_values), axis=1)
    candidate_idx = np.argsort(sample_scores)[::-1][:top_n]

    if base_values is None:
        base_values = np.zeros(shap_values.shape[0], dtype=float)
    else:
        base_values = np.asarray(base_values)

        if base_values.ndim == 0:
            base_values = np.repeat(float(base_values), shap_values.shape[0])
        elif base_values.ndim > 1:
            base_values = np.squeeze(base_values)

        if len(base_values) != shap_values.shape[0]:
            base_values = np.repeat(float(np.mean(base_values)), shap_values.shape[0])

    for rank_i, idx in enumerate(candidate_idx, start=1):
        try:
            exp = shap.Explanation(
                values=np.asarray(shap_values[idx], dtype=float),
                base_values=float(base_values[idx]),
                data=np.asarray(X_explain[idx], dtype=float),
                feature_names=feature_names
            )

            shap.plots.waterfall(exp, max_display=15, show=False)
            plt.savefig(
                os.path.join(results_dir, f"SHAP_{model_name}-{sex_name}_local_{rank_i}.png"),
                dpi=600, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            plt.close()
            print(f"[{sex_name}][{model_name}] local SHAP plot failed for sample {idx}: {e}")


def save_shap_plots_for_pipeline(
    fitted_pipe, X_train, X_test, sex_name, model_name, results_dir,
    feature_names, repeat_id, max_background=200, max_explain=300, random_state=42
):
    rng = np.random.default_rng(random_state)

    scaler = fitted_pipe.named_steps["scaler"]
    model = fitted_pipe.named_steps["model"]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_bg = min(max_background, X_train_scaled.shape[0])
    n_ex = min(max_explain, X_test_scaled.shape[0])

    bg_idx = rng.choice(X_train_scaled.shape[0], size=n_bg, replace=False)
    ex_idx = rng.choice(X_test_scaled.shape[0], size=n_ex, replace=False)

    X_bg = X_train_scaled[bg_idx]
    X_explain = X_test_scaled[ex_idx]

    try:
        if isinstance(model, (RandomForestRegressor, XGBRegressor)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
            try:
                base_values = explainer.expected_value
            except Exception:
                base_values = 0.0

        elif isinstance(model, (LinearRegression, Lasso)):
            explainer = shap.LinearExplainer(model, X_bg)
            shap_values = explainer.shap_values(X_explain)
            try:
                base_values = explainer.expected_value
            except Exception:
                base_values = 0.0

        else:
            explainer = shap.KernelExplainer(model.predict, X_bg)
            shap_values = explainer.shap_values(X_explain, nsamples=100)
            try:
                base_values = explainer.expected_value
            except Exception:
                base_values = 0.0
    except Exception as e:
        print(f"[{sex_name}][{model_name}] SHAP calculation failed: {e}")
        return None

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze()

    try:
        plt.figure()
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False, max_display=20)
        plt.savefig(os.path.join(results_dir, f"SHAP_{model_name}-{sex_name}_summary.png"),
                    bbox_inches="tight", dpi=600)
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type="bar",
                          show=False, max_display=20)
        plt.savefig(os.path.join(results_dir, f"SHAP_{model_name}-{sex_name}_bar.png"),
                    bbox_inches="tight", dpi=600)
        plt.close()

        save_local_shap_plots(
            shap_values=shap_values,
            X_explain=X_explain,
            feature_names=feature_names,
            results_dir=results_dir,
            model_name=model_name,
            sex_name=sex_name,
            top_n=3
        )

        txt_path = os.path.join(results_dir, f"SHAP_{model_name}-{sex_name}_values.txt")
        save_shap_values_to_txt(shap_values, feature_names, txt_path)

        imp_df = build_shap_importance_df(
            shap_values=shap_values,
            feature_names=feature_names,
            sex_name=sex_name,
            model_name=model_name,
            repeat_id=repeat_id
        )
        imp_df.to_csv(
            os.path.join(results_dir, f"SHAP_{model_name}-{sex_name}_importance.csv"),
            index=False, encoding="utf-8-sig"
        )
        return imp_df
    except Exception as e:
        print(f"[{sex_name}][{model_name}] SHAP save failed: {e}")
        return None


# =========================
# 6. DNN
# =========================
class DNNModel(nn.Module):
    def __init__(self, input_size):
        super(DNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def save_shap_plots_for_dnn(
    model, scaler, X_train, X_test,
    sex_name, model_name, results_dir, feature_names, repeat_id,
    device=None, max_background=200, max_explain=300, random_state=42
):
    rng = np.random.default_rng(random_state)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_bg = min(max_background, X_train_scaled.shape[0])
    n_ex = min(max_explain, X_test_scaled.shape[0])

    bg_idx = rng.choice(X_train_scaled.shape[0], size=n_bg, replace=False)
    ex_idx = rng.choice(X_test_scaled.shape[0], size=n_ex, replace=False)

    X_bg = X_train_scaled[bg_idx]
    X_explain = X_test_scaled[ex_idx]

    X_bg_tensor = torch.tensor(X_bg, dtype=torch.float32).to(device)
    X_explain_tensor = torch.tensor(X_explain, dtype=torch.float32).to(device)

    model.eval()
    shap_values = None
    explainer_name = None

    base_values = 0.0

    try:
        explainer = shap.DeepExplainer(model, X_bg_tensor)
        shap_values = explainer.shap_values(X_explain_tensor)
        explainer_name = "DeepExplainer"
        try:
            base_values = explainer.expected_value
        except Exception:
            base_values = 0.0
    except Exception:
        try:
            def predict_fn(x):
                x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                with torch.no_grad():
                    pred = model(x_tensor).cpu().numpy().squeeze()
                return pred

            explainer = shap.KernelExplainer(predict_fn, X_bg)
            shap_values = explainer.shap_values(X_explain, nsamples=100)
            explainer_name = "KernelExplainer"
            try:
                base_values = explainer.expected_value
            except Exception:
                base_values = 0.0
        except Exception as e2:
            print(f"[{sex_name}][{model_name}] DNN SHAP failed: {e2}")
            return None
        try:
            def predict_fn(x):
                x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                with torch.no_grad():
                    pred = model(x_tensor).cpu().numpy().squeeze()
                return pred

            explainer = shap.KernelExplainer(predict_fn, X_bg)
            shap_values = explainer.shap_values(X_explain, nsamples=200)
            explainer_name = "KernelExplainer"
        except Exception as e2:
            print(f"[{sex_name}][{model_name}] DNN SHAP failed: {e2}")
            return None

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze()
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    try:
        plt.figure()
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False, max_display=20)
        plt.savefig(os.path.join(results_dir, f"SHAP_{model_name}-{sex_name}_summary.png"),
                    bbox_inches="tight", dpi=600)
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type="bar",
                          show=False, max_display=20)
        plt.savefig(os.path.join(results_dir, f"SHAP_{model_name}-{sex_name}_bar.png"),
                    bbox_inches="tight", dpi=600)
        plt.close()

        save_local_shap_plots(
            shap_values=shap_values,
            X_explain=X_explain,
            feature_names=feature_names,
            results_dir=results_dir,
            model_name=model_name,
            sex_name=sex_name,
            base_values=base_values,
            top_n=3
        )

        txt_path = os.path.join(results_dir, f"SHAP_{model_name}-{sex_name}_values.txt")
        save_shap_values_to_txt(shap_values, feature_names, txt_path)

        with open(os.path.join(results_dir, f"SHAP_{model_name}-{sex_name}_explainer.txt"), "w", encoding="utf-8") as f:
            f.write(f"Explainer used: {explainer_name}\n")

        imp_df = build_shap_importance_df(
            shap_values=shap_values,
            feature_names=feature_names,
            sex_name=sex_name,
            model_name=model_name,
            repeat_id=repeat_id
        )
        imp_df.to_csv(
            os.path.join(results_dir, f"SHAP_{model_name}-{sex_name}_importance.csv"),
            index=False, encoding="utf-8-sig"
        )
        return imp_df
    except Exception as e:
        print(f"[{sex_name}][{model_name}] DNN SHAP save failed: {e}")
        return None


def train_dnn_with_shap_features(
    X_train, y_train, X_val, y_val, X_test, y_test,
    sex_name, model_name, results_dir, feature_names,
    repeat_id, epochs=300, batch_size=32, lr=1e-3,
    patience=30, random_state=42
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = DNNModel(input_size=X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_rmse = np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor.to(device)).cpu().numpy().squeeze()

        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor.to(device)).cpu().numpy().squeeze()

    dnn_shap_df = save_shap_plots_for_dnn(
        model=model,
        scaler=scaler,
        X_train=X_train,
        X_test=X_test,
        sex_name=sex_name,
        model_name=model_name,
        results_dir=results_dir,
        feature_names=feature_names,
        repeat_id=repeat_id,
        device=device,
        random_state=random_state
    )

    return model, scaler, test_pred, dnn_shap_df


# =========================
# 7. Bootstrap / permutation / rank-based
# =========================
def bootstrap_ci(values, n_boot=5000, ci=95, random_state=42):
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    n = len(values)

    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means.append(np.mean(values[idx]))

    boot_means = np.asarray(boot_means)
    mean_v = np.mean(values)
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return mean_v, lower, upper


def build_publication_summary_table_bootstrap(metrics_df, sex_name, results_dir, n_boot=5000):
    rows = []
    sub_df = metrics_df[metrics_df["Sex"] == sex_name].copy()

    for model in sub_df["Model"].unique():
        mdf = sub_df[sub_df["Model"] == model]
        row = {"Sex": sex_name, "Model": model}
        for metric in ["MAE", "RMSE", "R2", "R", "Bias", "Error_SD", "Slope", "Intercept"]:
            mean_v, lo, hi = bootstrap_ci(mdf[metric].values, n_boot=n_boot)
            row[f"{metric}_mean"] = mean_v
            row[f"{metric}_CI95_low"] = lo
            row[f"{metric}_CI95_high"] = hi
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("MAE_mean", ascending=True)
    out_csv = os.path.join(results_dir, f"Publication_Summary_Bootstrap_{sex_name}.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[{sex_name}] Bootstrap summary saved: {out_csv}")
    return out_df


def bootstrap_model_difference(metrics_df, sex_name, metric_name, model_a, model_b,
                               results_dir, n_boot=5000, random_state=42):
    rng = np.random.default_rng(random_state)

    df = metrics_df[metrics_df["Sex"] == sex_name].copy()
    pivot_df = df.pivot_table(index="Repeat", columns="Model", values=metric_name, aggfunc="mean").dropna(axis=0)

    a = pivot_df[model_a].values
    b = pivot_df[model_b].values
    diff = a - b
    n = len(diff)

    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means.append(np.mean(diff[idx]))

    mean_diff = np.mean(diff)
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)

    out_df = pd.DataFrame([{
        "Sex": sex_name,
        "Metric": metric_name,
        "Model_A": model_a,
        "Model_B": model_b,
        "Observed_Mean_Diff(A-B)": mean_diff,
        "Bootstrap_CI95_low": ci_low,
        "Bootstrap_CI95_high": ci_high
    }])

    out_csv = os.path.join(results_dir, f"BootstrapDiff_{metric_name}_{sex_name}_{model_a}_vs_{model_b}.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[{sex_name}] Bootstrap model difference saved: {out_csv}")
    return out_df


def paired_permutation_test(x, y, n_perm=10000, random_state=42):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    observed = np.mean(diff)

    perm_stats = []
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_stats.append(np.mean(diff * signs))

    perm_stats = np.asarray(perm_stats)
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
    return observed, p_value


def run_pairwise_permutation_tests(metrics_df, sex_name, metric_name, results_dir,
                                   n_perm=10000, p_adjust="holm"):
    df = metrics_df[metrics_df["Sex"] == sex_name].copy()
    pivot_df = df.pivot_table(index="Repeat", columns="Model", values=metric_name, aggfunc="mean").dropna(axis=0)

    model_names = list(pivot_df.columns)
    rows = []
    raw_pvals = []

    for m1, m2 in combinations(model_names, 2):
        x1 = pivot_df[m1].values
        x2 = pivot_df[m2].values
        obs_diff, p = paired_permutation_test(x1, x2, n_perm=n_perm)

        rows.append({
            "Sex": sex_name,
            "Metric": metric_name,
            "Model_1": m1,
            "Model_2": m2,
            "Model_1_mean": np.mean(x1),
            "Model_2_mean": np.mean(x2),
            "Observed_Mean_Diff(Model1-Model2)": obs_diff,
            "p_value_raw": p
        })
        raw_pvals.append(p)

    reject, pvals_adj, _, _ = multipletests(raw_pvals, method=p_adjust)

    for i in range(len(rows)):
        rows[i]["p_value_adj"] = pvals_adj[i]
        rows[i]["Significant"] = "Yes" if reject[i] else "No"

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, f"Pairwise_Permutation_{metric_name}_{sex_name}.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[{sex_name}] Pairwise permutation test saved: {out_csv}")
    return out_df


def run_friedman_nemenyi_test(metrics_df, sex_name, metric_name, results_dir):
    df = metrics_df[metrics_df["Sex"] == sex_name].copy()
    pivot_df = df.pivot_table(index="Repeat", columns="Model", values=metric_name, aggfunc="mean").dropna(axis=0)

    model_names = list(pivot_df.columns)
    values = [pivot_df[m].values for m in model_names]
    stat, p = stats.friedmanchisquare(*values)

    friedman_txt = os.path.join(results_dir, f"Friedman_{metric_name}_{sex_name}.txt")
    with open(friedman_txt, "w", encoding="utf-8") as f:
        f.write(f"Friedman test for {metric_name} ({sex_name})\n")
        f.write(f"Statistic = {stat:.6f}\n")
        f.write(f"P-value = {p:.10f}\n")

    nemenyi_df = sp.posthoc_nemenyi_friedman(pivot_df)
    nemenyi_csv = os.path.join(results_dir, f"Nemenyi_{metric_name}_{sex_name}.csv")
    nemenyi_df.to_csv(nemenyi_csv, encoding="utf-8-sig")

    rank_df = pivot_df.rank(axis=1, method="average", ascending=True)
    mean_ranks = rank_df.mean(axis=0).sort_values()

    rank_csv = os.path.join(results_dir, f"MeanRanks_{metric_name}_{sex_name}.csv")
    mean_ranks.to_csv(rank_csv, encoding="utf-8-sig", header=["MeanRank"])

    print(f"[{sex_name}] Friedman saved: {friedman_txt}")
    print(f"[{sex_name}] Nemenyi saved: {nemenyi_csv}")
    print(f"[{sex_name}] Mean ranks saved: {rank_csv}")

    return {
        "friedman_stat": stat,
        "friedman_p": p,
        "nemenyi": nemenyi_df,
        "mean_ranks": mean_ranks
    }


# =========================
# 8. Single model trainer
# =========================
def fit_model_and_predict(X_train, y_train, X_test, model, param_grid=None, random_state=42):
    cv5 = KFold(n_splits=5, shuffle=True, random_state=random_state)
    pipe = build_pipeline(model)

    if param_grid and len(param_grid) > 0:
        search = GridSearchCV(
            pipe, param_grid, cv=cv5,
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        best_params = search.best_params_
    else:
        fitted = pipe.fit(X_train, y_train)
        best_params = {}

    pred = fitted.predict(X_test)
    return fitted, pred, best_params


# =========================
# 9. Cross-model interpretive analysis
# =========================
def summarize_feature_rankings_across_models(all_shap_df, sex_name, results_dir, top_k=15):
    df = all_shap_df[all_shap_df["Sex"] == sex_name].copy()
    if df.empty:
        return None, None

    mean_imp = (
        df.groupby(["Model", "Feature"], as_index=False)["MeanAbsSHAP"]
        .mean()
    )

    mean_imp["RankWithinModel"] = (
        mean_imp.groupby("Model")["MeanAbsSHAP"]
        .rank(method="dense", ascending=False)
    )

    pivot_rank = mean_imp.pivot(index="Feature", columns="Model", values="RankWithinModel")
    pivot_imp = mean_imp.pivot(index="Feature", columns="Model", values="MeanAbsSHAP")

    pivot_rank["AppearanceCount"] = pivot_rank.notna().sum(axis=1)
    rank_cols = [c for c in pivot_rank.columns if c != "AppearanceCount"]
    pivot_rank["MeanRank"] = pivot_rank[rank_cols].mean(axis=1, skipna=True)

    rank_out = pivot_rank.sort_values(["AppearanceCount", "MeanRank"], ascending=[False, True])
    imp_out = pivot_imp.copy()

    rank_out.to_csv(
        os.path.join(results_dir, f"CrossModel_FeatureRank_{sex_name}.csv"),
        encoding="utf-8-sig"
    )
    imp_out.to_csv(
        os.path.join(results_dir, f"CrossModel_FeatureImportance_{sex_name}.csv"),
        encoding="utf-8-sig"
    )

    try:
        top_features = rank_out.head(top_k).index.tolist()
        heat_df = rank_out.loc[top_features, rank_cols]

        plt.figure(figsize=(10, max(4, len(top_features) * 0.5)))
        plt.imshow(heat_df.values, aspect="auto")
        plt.xticks(range(len(heat_df.columns)), heat_df.columns, rotation=45, ha="right")
        plt.yticks(range(len(heat_df.index)), heat_df.index)
        plt.colorbar(label="Rank")
        plt.title(f"Cross-model feature ranking ({sex_name})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, f"CrossModel_FeatureRankHeatmap_{sex_name}.png"),
            dpi=600, bbox_inches="tight"
        )
        plt.close()
    except Exception as e:
        print(f"[{sex_name}] Heatmap failed: {e}")

    try:
        top_features = rank_out.head(top_k).index.tolist()
        freq = rank_out.loc[top_features, "AppearanceCount"]
        plt.figure(figsize=(10, max(4, len(top_features) * 0.45)))
        plt.barh(top_features[::-1], freq.values[::-1])
        plt.xlabel("Number of models in which the feature appeared")
        plt.title(f"Cross-model feature appearance frequency ({sex_name})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, f"CrossModel_FeatureAppearance_{sex_name}.png"),
            dpi=600, bbox_inches="tight"
        )
        plt.close()
    except Exception as e:
        print(f"[{sex_name}] Appearance plot failed: {e}")

    return rank_out, imp_out


# =========================
# 10. Main Workflow
# =========================
def run_publication_pipeline_for_sex(
    X, Y, sex_name, results_dir, feature_names,
    shap_threshold=1.5, min_features=5, max_features=None,
    n_repeats=20, test_size=0.15, base_seed=42
):
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = []
    all_predictions = []
    all_selected_features = []
    all_model_shap_importance = []

    best_mae = np.inf
    best_record = None

    for repeat in range(n_repeats):
        seed = base_seed + repeat
        print(f"\n[{sex_name}] ===== Repeat {repeat+1}/{n_repeats} | seed={seed} =====")

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=seed
        )

        val_ratio_within_trainval = 0.15 / 0.85
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio_within_trainval,
            random_state=seed
        )

        print(f"[{sex_name}] Split sizes | Train: {len(y_tr)}, Val: {len(y_val)}, Test: {len(y_test)}")

        X_tr_sel, X_val_sel, X_test_sel, selected_idx, selected_features, _ = select_features_by_shap(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            X_test=X_test,
            feature_names=feature_names,
            sex_name=sex_name,
            model_name=f"Repeat{repeat+1}_SHAPSelector",
            results_dir=results_dir,
            shap_threshold=shap_threshold,
            min_features=min_features,
            max_features=max_features,
            random_state=seed
        )

        for feat in selected_features:
            all_selected_features.append({"Sex": sex_name, "Repeat": repeat + 1, "Feature": feat})

        model_preds = {}

        # KDM
        kdm_pred = KDM_Age(
            X=X_tr_sel,
            Y=y_tr,
            X_test=X_test_sel,
            Y_test=y_test,
            feature_names=selected_features,
            verbose=False
        )[0]
        model_preds["KDM"] = kdm_pred

        # MLR
        mlr_fitted, mlr_pred, _ = fit_model_and_predict(
            X_tr_sel, y_tr, X_test_sel,
            LinearRegression(), {}, random_state=seed
        )
        model_preds["MLR"] = mlr_pred
        mlr_shap_df = save_shap_plots_for_pipeline(
            fitted_pipe=mlr_fitted,
            X_train=X_tr_sel,
            X_test=X_test_sel,
            sex_name=sex_name,
            model_name=f"MLR_Repeat{repeat+1}",
            results_dir=results_dir,
            feature_names=selected_features,
            repeat_id=repeat + 1,
            random_state=seed
        )
        if mlr_shap_df is not None:
            all_model_shap_importance.append(mlr_shap_df)

        # LASSO
        lasso_fitted, lasso_pred, _ = fit_model_and_predict(
            X_tr_sel, y_tr, X_test_sel,
            Lasso(max_iter=20000, random_state=seed),
            {"model__alpha": [0.001, 0.01, 0.1, 1, 10]},
            random_state=seed
        )
        model_preds["LASSO"] = lasso_pred
        lasso_shap_df = save_shap_plots_for_pipeline(
            fitted_pipe=lasso_fitted,
            X_train=X_tr_sel,
            X_test=X_test_sel,
            sex_name=sex_name,
            model_name=f"LASSO_Repeat{repeat+1}",
            results_dir=results_dir,
            feature_names=selected_features,
            repeat_id=repeat + 1,
            random_state=seed
        )
        if lasso_shap_df is not None:
            all_model_shap_importance.append(lasso_shap_df)

        # SVR
        svr_fitted, svr_pred, _ = fit_model_and_predict(
            X_tr_sel, y_tr, X_test_sel,
            SVR(kernel="rbf"),
            {"model__C": [0.5, 1, 5], "model__epsilon": [0.05, 0.1, 0.2], "model__gamma": ["scale", "auto"]},
            random_state=seed
        )
        model_preds["SVR"] = svr_pred
        svr_shap_df = save_shap_plots_for_pipeline(
            fitted_pipe=svr_fitted,
            X_train=X_tr_sel,
            X_test=X_test_sel,
            sex_name=sex_name,
            model_name=f"SVR_Repeat{repeat+1}",
            results_dir=results_dir,
            feature_names=selected_features,
            repeat_id=repeat + 1,
            random_state=seed
        )
        if svr_shap_df is not None:
            all_model_shap_importance.append(svr_shap_df)

        # RF
        rf_fitted, rf_pred, _ = fit_model_and_predict(
            X_tr_sel, y_tr, X_test_sel,
            RandomForestRegressor(random_state=seed),
            {"model__n_estimators": [200, 500], "model__max_depth": [10, 25, None]},
            random_state=seed
        )
        model_preds["RF"] = rf_pred
        rf_shap_df = save_shap_plots_for_pipeline(
            fitted_pipe=rf_fitted,
            X_train=X_tr_sel,
            X_test=X_test_sel,
            sex_name=sex_name,
            model_name=f"RF_Repeat{repeat+1}",
            results_dir=results_dir,
            feature_names=selected_features,
            repeat_id=repeat + 1,
            random_state=seed
        )
        if rf_shap_df is not None:
            all_model_shap_importance.append(rf_shap_df)

        # XGB
        xgb_fitted, xgb_pred, _ = fit_model_and_predict(
            X_tr_sel, y_tr, X_test_sel,
            XGBRegressor(random_state=seed, objective="reg:squarederror", n_estimators=500),
            {
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            },
            random_state=seed
        )
        model_preds["XGB"] = xgb_pred
        xgb_shap_df = save_shap_plots_for_pipeline(
            fitted_pipe=xgb_fitted,
            X_train=X_tr_sel,
            X_test=X_test_sel,
            sex_name=sex_name,
            model_name=f"XGB_Repeat{repeat+1}",
            results_dir=results_dir,
            feature_names=selected_features,
            repeat_id=repeat + 1,
            random_state=seed
        )
        if xgb_shap_df is not None:
            all_model_shap_importance.append(xgb_shap_df)

        # DNN
        _, _, dnn_pred, dnn_shap_df = train_dnn_with_shap_features(
            X_train=X_tr_sel, y_train=y_tr,
            X_val=X_val_sel, y_val=y_val,
            X_test=X_test_sel, y_test=y_test,
            sex_name=sex_name,
            model_name=f"DNN_Repeat{repeat+1}",
            results_dir=results_dir,
            feature_names=selected_features,
            repeat_id=repeat + 1,
            epochs=300, batch_size=32, lr=1e-3, patience=30,
            random_state=seed
        )
        model_preds["DNN"] = dnn_pred
        if dnn_shap_df is not None:
            all_model_shap_importance.append(dnn_shap_df)

        # 记录指标
        for model_name, pred in model_preds.items():
            pred = np.asarray(pred).reshape(-1)
            metrics = compute_metrics(y_test, pred)
            row = {"Sex": sex_name, "Repeat": repeat + 1, "Model": model_name}
            row.update(metrics)
            all_metrics.append(row)

            pred_df = pd.DataFrame({
                "Sex": sex_name,
                "Repeat": repeat + 1,
                "Model": model_name,
                "y_true": y_test,
                "y_pred": pred
            })
            all_predictions.append(pred_df)

            if repeat == n_repeats - 1:
                repeat_plot_dir = os.path.join(results_dir, sex_name, f"Repeat_{repeat+1}", model_name)
                os.makedirs(repeat_plot_dir, exist_ok=True)
                plot_base_name = f"{sex_name}_Repeat{repeat+1}_{model_name}"
                plot_results(y_test, pred, plot_base_name, repeat_plot_dir)
                plot_error_distribution(y_test, pred, plot_base_name, repeat_plot_dir)
                plot_bland_altman(y_test, pred, plot_base_name, repeat_plot_dir)

            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                best_record = {
                    "model_name": model_name,
                    "repeat": repeat + 1,
                    "y_true": y_test.copy(),
                    "y_pred": np.asarray(pred).copy(),
                    "sex_name": sex_name
                }

    metrics_df = pd.DataFrame(all_metrics)
    pred_df = pd.concat(all_predictions, axis=0, ignore_index=True)
    feat_df = pd.DataFrame(all_selected_features)

    metrics_df.to_csv(os.path.join(results_dir, f"Repeated_Metrics_{sex_name}.csv"),
                      index=False, encoding="utf-8-sig")
    pred_df.to_csv(os.path.join(results_dir, f"Repeated_Predictions_{sex_name}.csv"),
                   index=False, encoding="utf-8-sig")

    if len(feat_df) > 0:
        feat_freq = feat_df.groupby("Feature").size().reset_index(name="SelectionCount")
        feat_freq["SelectionFrequency"] = feat_freq["SelectionCount"] / n_repeats
        feat_freq = feat_freq.sort_values("SelectionCount", ascending=False)
        feat_freq.to_csv(os.path.join(results_dir, f"SelectedFeatureFrequency_{sex_name}.csv"),
                         index=False, encoding="utf-8-sig")

    if len(all_model_shap_importance) > 0:
        all_shap_df = pd.concat(all_model_shap_importance, axis=0, ignore_index=True)
        all_shap_df.to_csv(
            os.path.join(results_dir, f"AllModel_SHAP_Importance_{sex_name}.csv"),
            index=False, encoding="utf-8-sig"
        )
        summarize_feature_rankings_across_models(all_shap_df, sex_name, results_dir, top_k=15)

    summary_df = build_publication_summary_table_bootstrap(metrics_df, sex_name, results_dir, n_boot=5000)

    run_pairwise_permutation_tests(metrics_df, sex_name, "MAE", results_dir, n_perm=10000, p_adjust="holm")
    run_pairwise_permutation_tests(metrics_df, sex_name, "RMSE", results_dir, n_perm=10000, p_adjust="holm")

    rank_result_mae = run_friedman_nemenyi_test(metrics_df, sex_name, "MAE", results_dir)
    rank_result_rmse = run_friedman_nemenyi_test(metrics_df, sex_name, "RMSE", results_dir)
    plot_mean_ranks(rank_result_mae["mean_ranks"], "MAE", sex_name, results_dir)
    plot_mean_ranks(rank_result_rmse["mean_ranks"], "RMSE", sex_name, results_dir)

    plot_metric_boxplot(metrics_df, sex_name, "MAE", results_dir)
    plot_metric_boxplot(metrics_df, sex_name, "RMSE", results_dir)

    if best_record is not None:
        best_name = f"BestModel_{best_record['model_name']}_{sex_name}_Repeat{best_record['repeat']}"
        plot_results(best_record["y_true"], best_record["y_pred"], best_name, results_dir)
        plot_error_distribution(best_record["y_true"], best_record["y_pred"], best_name, results_dir)
        plot_bland_altman(best_record["y_true"], best_record["y_pred"], best_name, results_dir)

    if len(summary_df) >= 2:
        best_model = summary_df.iloc[0]["Model"]
        second_model = summary_df.iloc[1]["Model"]
        bootstrap_model_difference(metrics_df, sex_name, "MAE", best_model, second_model, results_dir)
        bootstrap_model_difference(metrics_df, sex_name, "RMSE", best_model, second_model, results_dir)

    return metrics_df, pred_df, summary_df


# =========================
# 11. Main Function
# =========================
def main():
    set_seed(42)

    excel_file_path = r""
    RESULTS_DIR = r""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.isfile(excel_file_path):
        raise FileNotFoundError(f"Excel file not found: {excel_file_path}")

    male_all_data, female_all_data = load_excel_all_data(excel_file_path)

    X_male, Y_male = getfeature(male_all_data)
    X_female, Y_female = getfeature(female_all_data)

    X_male, Y_male, male_feature_names = clean_features_and_target(X_male, Y_male, sex_name="Male")
    X_female, Y_female, female_feature_names = clean_features_and_target(X_female, Y_female, sex_name="Female")

    print("\n========== Publication-grade modeling: Male ==========")
    run_publication_pipeline_for_sex(
        X_male, Y_male,
        sex_name="Male",
        results_dir=RESULTS_DIR,
        feature_names=male_feature_names,
        shap_threshold=1.5,
        min_features=5,
        max_features=None,
        n_repeats=20,
        test_size=0.15,
        base_seed=42
    )

    print("\n========== Publication-grade modeling: Female ==========")
    run_publication_pipeline_for_sex(
        X_female, Y_female,
        sex_name="Female",
        results_dir=RESULTS_DIR,
        feature_names=female_feature_names,
        shap_threshold=1.5,
        min_features=5,
        max_features=None,
        n_repeats=20,
        test_size=0.15,
        base_seed=1042
    )

    print("\n========== Done ==========")


if __name__ == "__main__":
    main()
