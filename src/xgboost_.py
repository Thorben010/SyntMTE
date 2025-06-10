import os
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from pymatgen.core import Composition
from tqdm import tqdm
from pymatgen.core.periodic_table import Element
from xgboost import XGBRegressor
import xgboost as xgb


# List of elements for feature vector
ALL_ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
ELEMENT_MAP = {el: i for i, el in enumerate(ALL_ELEMENTS)}
N_ELEMENTS = len(ALL_ELEMENTS)


def featurize_composition(formula_str: str) -> np.ndarray:
    """Converts a chemical formula string into a fixed-size feature vector."""
    feature_vector = np.zeros(N_ELEMENTS)
    try:
        comp = Composition(formula_str)
        total_atoms = comp.num_atoms
        if total_atoms > 0:
            for element, amount in comp.items():
                if element.symbol in ELEMENT_MAP:
                    feature_vector[ELEMENT_MAP[element.symbol]] = amount / total_atoms
                else:
                    print(
                        f"Warning: Element {element.symbol} not in known element list."
                    )
    except Exception as e:
        # Handle cases where Composition parsing fails (e.g., invalid format)
        print(f"Warning: Could not parse formula '{formula_str}': {e}")
        # Return a zero vector or handle as appropriate
        pass
    return feature_vector


def safe_literal_eval(val):
    import ast

    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    elif pd.isna(val):
        return []
    return val


def load_and_featurize_data(file_path: str, use_precursor: bool = False):
    """Loads data from CSV, featurizes compositions (target and optionally precursor), returns X, y, mask."""
    print(
        f"Loading and featurizing data from {file_path} "
        f"(use_precursor={use_precursor})..."
    )
    df = pd.read_csv(file_path, na_values=["nan"])

    # Target formula parsing - target_formula is already a string, not a list!
    # So we don't need to parse it with safe_literal_eval
    df["formula_str"] = df["target_formula"]

    # Precursor formula parsing (if requested)
    if use_precursor:
        if "precursor_formulas" not in df.columns:
            raise KeyError(
                "Column 'precursor_formulas' not found in data but "
                "--use-precursor was set."
            )
        df["precursor_formula_list"] = df["precursor_formulas"].apply(safe_literal_eval)
        df["precursor_str"] = df["precursor_formula_list"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        )

    # Drop rows where formula couldn't be extracted
    drop_cols = ["formula_str"] + (["precursor_str"] if use_precursor else [])
    df = df.dropna(subset=drop_cols).reset_index(drop=True)

    # Featurize compositions
    print("Featurizing compositions...")
    features = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        f_target = featurize_composition(row["formula_str"])
        if use_precursor:
            f_prec = featurize_composition(row["precursor_str"])
            # average target and precursor features
            feat_vec = 0.5 * (f_target + f_prec)
        else:
            feat_vec = f_target
        features.append(feat_vec)

    X = np.array(features)
    
    # Ensure X is always 2D, even when empty
    if len(features) == 0:
        X = np.empty((0, N_ELEMENTS))
    
    print(f"  [Debug] Feature matrix shape: {X.shape}")

    # Extract targets and mask
    target_cols = [
        "Sintering Temperature",
        "Sintering Time",
        "Calcination Temperature",
        "Calcination Time",
    ]
    y = df[target_cols].values
    mask = df[target_cols].notna().astype(int).values

    print(f"Loaded {len(df)} samples.")
    return X, y, mask, target_cols


def train_evaluate_xgboost(
    train_path,
    val_path,
    test_path,
    log_dir,
    random_state_seed=None,
    n_iter_search=10,
    cv_search=3,
    use_precursor: bool = False,
):
    """
    Trains and evaluates separate XGBoost models for each target variable
    using hyperparameter optimization (RandomizedSearchCV) and early stopping.

    Args:
        train_path (str): Path to the training CSV file.
        val_path (str): Path to the validation CSV file.
        test_path (str): Path to the test CSV file.
        log_dir (str): Directory to save the trained models and results.
        random_state_seed (int, optional): Seed for random operations. Defaults to None.
        n_iter_search (int): Number of parameter settings that are sampled for RandomizedSearchCV.
        cv_search (int): Number of cross-validation folds for RandomizedSearchCV.
    """
    X_train, y_train, mask_train, target_cols = load_and_featurize_data(
        train_path, use_precursor=use_precursor
    )
    X_val, y_val, mask_val, _ = load_and_featurize_data(
        val_path, use_precursor=use_precursor
    )
    X_test, y_test, mask_test, _ = load_and_featurize_data(
        test_path, use_precursor=use_precursor
    )

    # Ensure consistent feature dimensions
    # (Handling potential inconsistencies is important, but omitted here for brevity)
    
    # Check if any dataset is empty
    if len(X_train) == 0:
        print("Error: Training dataset is empty. Cannot proceed.")
        return {}, {}
    
    if len(X_val) == 0:
        print("Warning: Validation dataset is empty. This may affect training.")
    
    if len(X_test) == 0:
        print("Warning: Test dataset is empty. Cannot evaluate models.")
    
    # Check feature dimensions only for non-empty datasets
    datasets_to_check = []
    if len(X_train) > 0:
        datasets_to_check.append(("train", X_train))
    if len(X_val) > 0:
        datasets_to_check.append(("val", X_val))
    if len(X_test) > 0:
        datasets_to_check.append(("test", X_test))
    
    for name, dataset in datasets_to_check:
        if dataset.shape[1] != N_ELEMENTS:
            print(f"Warning: {name} dataset feature dimensions mismatch. "
                  f"Expected {N_ELEMENTS}, got {dataset.shape[1]}")

    models = {}
    # Initialize y_pred based on test set shape, handle empty test set
    if len(X_test) > 0:
        y_pred = np.full(y_test.shape, np.nan)  # Initialize predictions with NaN
    else:
        y_pred = np.full((0, len(target_cols)), np.nan)  # Empty array with correct shape
    metrics = {}
    run_best_hyperparams = {}  # To store best HPs for this run's targets
    target_names_map = {
        "Sintering Temperature": "sint_temp",
        "Sintering Time": "sint_time",
        "Calcination Temperature": "calc_temp",
        "Calcination Time": "calc_time",
    }
    target_names = [target_names_map[col] for col in target_cols]

    print("Training and evaluating models for each target...")
    for i, col_name in enumerate(target_cols):
        target_short_name = target_names_map[col_name]
        print(f"--- Training model for: {col_name} ({target_short_name}) ---")

        # Filter data based on mask for the current target
        train_indices = mask_train[:, i] == 1
        # Handle empty validation set gracefully
        val_indices = mask_val[:, i] == 1 if len(X_val) > 0 else np.array([], dtype=bool)
        test_indices = mask_test[:, i] == 1 if len(X_test) > 0 else np.array([], dtype=bool)

        X_train_masked = X_train[train_indices]
        y_train_masked = y_train[train_indices, i]
        
        # Handle validation data
        if len(X_val) > 0:
            X_val_masked = X_val[val_indices]
            y_val_masked = y_val[val_indices, i]
        else:
            X_val_masked = np.empty((0, N_ELEMENTS))
            y_val_masked = np.array([])

        # Check if we have enough training data (validation is optional)
        if len(X_train_masked) == 0:
            print(
                f"Skipping {col_name}: No training data available after masking."
            )
            metrics[f"{target_short_name}_mae"] = np.nan
            metrics[f"{target_short_name}_r2"] = np.nan
            metrics[f"{target_short_name}_rmse"] = np.nan
            models[target_short_name] = None  # Store None if model couldn't be trained
            run_best_hyperparams[target_short_name] = None  # Store None for HPs as well
            continue

        # Warn about validation data
        if len(X_val_masked) == 0:
            print(f"  Warning: No validation data for {col_name}, training without early stopping.")

        print(
            f"Training samples for HPO: {len(X_train_masked)}, "
            f"Validation samples for early stopping: {len(X_val_masked)}"
        )

        # Define the parameter grid for RandomizedSearchCV
        param_dist = {
            "n_estimators": [100, 200, 300, 500, 700, 1000],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6, 7, 8],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "gamma": [0, 0.1, 0.2, 0.3],
            "reg_alpha": [0, 0.001, 0.01, 0.1],
            "reg_lambda": [1, 1.1, 1.2, 1.3],  # XGBoost default is 1 for L2
        }

        # Initialize XGBRegressor for HPO (early stopping is handled within 
        # RandomizedSearchCV's fit if estimator supports it, or in the final fit)
        xgb_reg_hpo = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            random_state=random_state_seed,
            n_jobs=-1,
            # Early stopping for HPO needs to be handled carefully.
            # If RandomizedSearchCV's fit uses eval_set and early_stopping_rounds, it's fine.
            # Otherwise, we use a large number of estimators and rely on early 
            # stopping in the *final* model training.
            # For simplicity, let's use early stopping in the final training step post-HPO.
        )

        print(
            f"  Starting RandomizedSearchCV for {col_name} "
            f"(n_iter={n_iter_search}, cv={cv_search})..."
        )
        # Note: For RandomizedSearchCV with XGBoost, if you want early stopping 
        # *during the search for each fold*,
        # you'd typically pass fit_params to search.fit().
        # Example: fit_params={'early_stopping_rounds': 10, 
        #                      'eval_set': [[X_val_masked, y_val_masked]]}
        # However, this applies the *same* validation set to all folds of the HPO CV, 
        # which is not ideal.
        # A more robust HPO would involve nested CV or using the validation set purely 
        # for the *final* model's early stopping.
        # Here, we'll perform HPO on (X_train_masked, y_train_masked) with CV, 
        # then train a final model
        # using (X_train_masked, y_train_masked) and early stopping against 
        # (X_val_masked, y_val_masked).

        search = RandomizedSearchCV(
            estimator=xgb_reg_hpo,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring="neg_mean_absolute_error",
            cv=cv_search,
            random_state=random_state_seed,
            verbose=1,  # Set to higher for more output
            n_jobs=-1,  # Use all available cores for HPO
        )

        search.fit(
            X_train_masked, y_train_masked
        )  # HPO is done on the training data partition
        print(
            f"  RandomizedSearchCV complete for {col_name}. "
            f"Best params: {search.best_params_}"
        )

        # Train the final model for the current target with best HPO params and early stopping
        best_params = search.best_params_
        run_best_hyperparams[target_short_name] = (
            best_params  # Store the best params for this target
        )

        xgb_regressor_final = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            **best_params,  # Unpack best parameters found by HPO
            random_state=random_state_seed,
            n_jobs=-1,
        )

        # Fit with or without early stopping depending on validation data availability
        if len(X_val_masked) > 0:
            xgb_regressor_final.set_params(early_stopping_rounds=50)
            xgb_regressor_final.fit(
                X_train_masked,
                y_train_masked,
                eval_set=[
                    (X_val_masked, y_val_masked)
                ],  # Use the dedicated validation set for early stopping
                verbose=False,  # Set to True or a number to see training progress
            )
            print(
                f"  [Debug] Final model training complete for {col_name}. "
                f"Best iteration: {xgb_regressor_final.best_iteration}"
            )
        else:
            # Train without early stopping if no validation data
            xgb_regressor_final.fit(X_train_masked, y_train_masked)
            print(
                f"  [Debug] Final model training complete for {col_name} "
                f"(without early stopping)."
            )

        models[target_short_name] = xgb_regressor_final  # Store the trained model

        # --- Evaluation ---
        # Only evaluate if test data is available
        if len(X_test) > 0:
            # Predict on the *full* test set
            y_pred_full = xgb_regressor_final.predict(X_test)
            # Fill the corresponding column in our overall prediction matrix
            y_pred[:, i] = y_pred_full

            # Evaluate only on the test samples where the true value exists
            if len(test_indices) > 0:
                y_test_masked = y_test[test_indices, i]
                y_pred_masked = y_pred[
                    test_indices, i
                ]  # Select predictions using the test mask

                if len(y_test_masked) > 0:
                    mae = mean_absolute_error(y_test_masked, y_pred_masked)
                    r2 = r2_score(y_test_masked, y_pred_masked)
                    mse = mean_squared_error(y_test_masked, y_pred_masked)
                    rmse = np.sqrt(mse)
                    metrics[f"{target_short_name}_mae"] = mae
                    metrics[f"{target_short_name}_r2"] = r2
                    metrics[f"{target_short_name}_rmse"] = rmse
                    print(
                        f"  Test Set Metrics (Masked) - MAE: {mae:.4f}, R2: {r2:.4f}, "
                        f"RMSE: {rmse:.4f} (n={len(y_test_masked)})"
                    )
                else:
                    metrics[f"{target_short_name}_mae"] = np.nan
                    metrics[f"{target_short_name}_r2"] = np.nan
                    metrics[f"{target_short_name}_rmse"] = np.nan
                    print(f"  {col_name} - No valid test samples found.")
            else:
                metrics[f"{target_short_name}_mae"] = np.nan
                metrics[f"{target_short_name}_r2"] = np.nan
                metrics[f"{target_short_name}_rmse"] = np.nan
                print(f"  {col_name} - No test samples found after masking.")
        else:
            metrics[f"{target_short_name}_mae"] = np.nan
            metrics[f"{target_short_name}_r2"] = np.nan
            metrics[f"{target_short_name}_rmse"] = np.nan
            print(f"  {col_name} - No test data available for evaluation.")

    # --- Save Models and Metrics ---
    os.makedirs(log_dir, exist_ok=True)

    # Save individual models
    for name, model in models.items():
        if model:  # Only save if model was trained
            model_path = os.path.join(log_dir, f"xgboost_model_{name}.joblib")
            joblib.dump(model, model_path)
            print(f"Model for {name} saved to {model_path}")
        else:
            print(f"Model for {name} was not trained (skipped).")

    # Save metrics
    metrics_path = os.path.join(log_dir, "xgboost_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Test Set Metrics (Masked):\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
    print(f"Metrics saved to {metrics_path}")

    return metrics, run_best_hyperparams  # Return metrics and the collected best HPs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate an XGBoost model for material synthesis "
        "conditions."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="/home/thor/code/synth_con_pred/data/conditions/"
        "random_split/train.csv",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default="/home/thor/code/synth_con_pred/data/conditions/"
        "random_split/val.csv",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="/home/thor/code/synth_con_pred/data/conditions/"
        "random_split/test.csv",
    )
    parser.add_argument("--log-dir-base", type=str, default="logs/xgboost_runs")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--n-iter-search", type=int, default=20)
    parser.add_argument("--cv-search", type=int, default=3)
    parser.add_argument(
        "--use-precursor",
        default=True,
        action="store_true",
        help="Include precursor composition features (averaged with target).",
    )

    args = parser.parse_args()

    all_run_metrics = []
    all_run_hyperparams = []  # New list to store HPs from all runs

    for i in range(args.num_runs):
        run_ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(args.log_dir_base, f"run_{i+1}_{run_ts}")
        print(f"\n--- Starting Run {i+1}/{args.num_runs} (Seed: {i}) ---")
        metrics, run_hyperparams = train_evaluate_xgboost(
            args.train_path,
            args.val_path,
            args.test_path,
            log_dir,
            random_state_seed=i,
            n_iter_search=args.n_iter_search,
            cv_search=args.cv_search,
            use_precursor=args.use_precursor,
        )
        all_run_metrics.append(metrics)
        all_run_hyperparams.append(run_hyperparams)

    # --- Aggregate and print Test Set Metrics Statistics ---
    if all_run_metrics:
        print("\n--- Test Set Metrics Statistics Across Runs ---")
        # Get all unique metric keys from the first run (assuming all runs produce the same metric keys)
        if not all_run_metrics[0]:
            print("No metrics found in the first run to determine metric keys.")
        else:
            metric_keys = list(all_run_metrics[0].keys())
            aggregated_stats = {}

            for key in metric_keys:
                metric_values_for_key = []
                for run_metrics in all_run_metrics:
                    if key in run_metrics and pd.notna(
                        run_metrics[key]
                    ):  # Check for NaN
                        metric_values_for_key.append(run_metrics[key])

                if metric_values_for_key:
                    mean_val = np.mean(metric_values_for_key)
                    std_val = np.std(metric_values_for_key)
                    aggregated_stats[key] = {
                        "mean": mean_val,
                        "std": std_val,
                        "values": metric_values_for_key,
                    }
                    print(
                        f"  {key}: Mean = {mean_val:.4f}, Std = {std_val:.4f} "
                        f"(Values: {[f'{v:.4f}' for v in metric_values_for_key]})"
                    )
                else:
                    print(f"  {key}: No valid (non-NaN) values found across runs.")

        # Save all individual run metrics to a single file (as before)
        aggregated_metrics_path = os.path.join(
            args.log_dir_base, "aggregated_run_metrics.txt"
        )  # Renamed for clarity
        with open(aggregated_metrics_path, "w") as f:
            for i, run_metrics in enumerate(all_run_metrics):
                f.write(f"Run {i+1} Metrics (Seed: {i}):\n")
                if run_metrics:
                    for key, value in run_metrics.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write("  No metrics recorded for this run.\n")
                f.write("\n")
        print(f"\nAll run metrics saved to {aggregated_metrics_path}")
    else:
        print("\nNo run metrics collected to aggregate.")

    print("\nAll runs complete.")
