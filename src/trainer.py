import os
import sys
import json
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot
from tqdm import tqdm, trange
import optuna
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import matplotlib.pyplot as plt
from termcolor import colored

from models.composition_mlp import CompositionMLP
from pymatgen.core import Composition
from dataset import MaterialDataset
from utils import custom_collate_fn


class Trainer:
    """Trainer class for model training and evaluation.

    This class handles the training loop, evaluation, and prediction for the
    CompositionMLP model. It includes functionality for cyclic learning rates,
    early stopping, and metric logging.
    """

    def __init__(self, trainer_config, model_config, info, use_target_only=False):
        """Initialize the Trainer.

        Args:
            trainer_config (dict): Configuration dictionary containing training
                parameters like model, datasets, loss function, optimizer, etc.
            model_config (dict): Configuration dictionary containing model
                parameters like embedder type, number of layers, etc.
            info (list): List containing information about the training run.
            use_target_only (bool): If True, only target material representation
                is used for regression. Defaults to False.
        """
        self.model = trainer_config["model"]
        self.train_dataset = trainer_config["train_dataset"]
        self.val_dataset = trainer_config["val_dataset"]
        self.test_dataset = trainer_config["test_dataset"]
        self.loss_fn = trainer_config["loss_fn"]
        self.optimizer = trainer_config["optimizer"]
        self.batch_size = trainer_config["batch_size"]
        self.device = trainer_config["device"]
        self.log_dir = (
            trainer_config["log_dir"] + f"{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
        )
        self.use_target_only = use_target_only

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            drop_last=True,
        )
        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=1024,
                shuffle=False,
                collate_fn=custom_collate_fn,
            )
            if self.val_dataset
            else None
        )
        self.test_loader = (
            DataLoader(
                self.test_dataset,
                batch_size=1024,
                shuffle=False,
                collate_fn=custom_collate_fn,
            )
            if self.test_dataset
            else None
        )

        os.makedirs(self.log_dir, exist_ok=True)

        # Save model config
        model_config_path = os.path.join(self.log_dir, "model_config.json")
        # Convert non-serializable items like 'device' to strings
        serializable_model_config = {
            k: str(v) if isinstance(v, torch.device) else v
            for k, v in model_config.items()
        }
        with open(model_config_path, "w") as f:
            json.dump(serializable_model_config, f, indent=4)

        with open(f"{self.log_dir}/log.txt", "w") as f:
            f.write(f"INFO: {info}\n")

        # Initialize cyclic learning rate scheduler if enabled
        if trainer_config.get("use_cyclic_lr", False):
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=trainer_config.get("base_lr", 5e-5),
                max_lr=trainer_config.get("max_lr", 1e-2),
                step_size_up=len(self.train_loader) * 2,  # one epoch up
                mode=trainer_config.get("cyclic_mode", "triangular"),
            )
        else:
            self.scheduler = None

    def train(self, epochs, eval_interval=1, get_best=False):
        """Train the model for the specified number of epochs.

        Args:
            epochs (int): Number of epochs to train.
            eval_interval (int): How often to evaluate on validation set.
            get_best (bool): Whether to return best model's metrics.

        Returns:
            float: Validation loss of the best model.
        """
        self.model.to(self.device)
        self.model.train()
        self.patience_counter = 0
        self.best_val = np.inf

        physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        if getattr(self.device, "type", None) == "cuda" and physical_gpu is not None:
            print(f"Training on GPU {physical_gpu} (CUDA device {self.device})")
        else:
            print(f"Training on {self.device}")

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_samples = 0

            for it, (
                mat,
                precursor_formulas,
                sint_temp_tensor,
                sint_time_tensor,
                calc_temp_tensor,
                calc_time_tensor,
                mask,
            ) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                # Skip batch if size is less than 2 to avoid BatchNorm errors
                if len(mat) < 2:
                    print(
                        f"Warning: Skipping batch of size {len(mat)} in epoch {epoch}"
                    )
                    continue
                self.optimizer.zero_grad()

                # Move inputs to device
                sint_temp_tensor = sint_temp_tensor.to(self.device)
                sint_time_tensor = sint_time_tensor.to(self.device)
                calc_temp_tensor = calc_temp_tensor.to(self.device)
                calc_time_tensor = calc_time_tensor.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                if self.use_target_only:
                    logits = self.model(mat, [])
                else:
                    logits = self.model(mat, precursor_formulas)
                pred_sint_temp, pred_sint_time, pred_calc_temp, pred_calc_time = (
                    logits[:, 0],
                    logits[:, 1],
                    logits[:, 2],
                    logits[:, 3],
                )

                sint_temp_tensor = torch.nan_to_num(sint_temp_tensor, nan=0.0)
                sint_time_tensor = torch.nan_to_num(sint_time_tensor, nan=0.0)
                calc_temp_tensor = torch.nan_to_num(calc_temp_tensor, nan=0.0)
                calc_time_tensor = torch.nan_to_num(calc_time_tensor, nan=0.0)

                # Apply mask to the predictions and targets
                pred_sint_temp = pred_sint_temp * mask[:, 0]
                pred_sint_time = pred_sint_time * mask[:, 1]
                pred_calc_temp = pred_calc_temp * mask[:, 2]
                pred_calc_time = pred_calc_time * mask[:, 3]

                # Compute losses
                loss_sint_temp = self.loss_fn(
                    pred_sint_temp, sint_temp_tensor * mask[:, 0]
                )
                loss_sint_time = self.loss_fn(
                    pred_sint_time, sint_time_tensor * mask[:, 1]
                )
                loss_calc_temp = self.loss_fn(
                    pred_calc_temp, calc_temp_tensor * mask[:, 2]
                )
                loss_calc_time = self.loss_fn(
                    pred_calc_time, calc_time_tensor * mask[:, 3]
                )

                # Total loss
                loss = loss_calc_temp + loss_calc_time

                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                total_loss += loss.item()
                total_samples += len(mat)

            # Log training metrics
            self.log_metrics(
                f"Epoch {epoch} Training", total_loss / len(self.train_loader)
            )

            if epoch % eval_interval == 0:
                # Evaluate on validation set and log all metrics
                val_metrics = self.eval(self.val_loader)
                self.log_metrics(
                    f"Epoch {epoch} Validation",
                    val_metrics["loss"],
                    **{k: v for k, v in val_metrics.items() if k != "loss"},
                )
                if val_metrics["loss"] < self.best_val:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.log_dir, "best_model.pth"),
                    )
                    self.best_val = val_metrics["loss"]
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= 5:
                    print("Early stopping")
                    break

            # Set the model back to training mode after evaluation
            self.model.train()

        self.model.load_state_dict(
            torch.load(os.path.join(self.log_dir, "best_model.pth"))
        )
        train_score = self.eval(self.train_loader)
        self.log_metrics(
            f"Epoch {epoch} Train Final",
            train_score["loss"],
            **{k: v for k, v in train_score.items() if k != "loss"},
        )
        val_score = self.eval(self.val_loader)
        self.log_metrics(
            f"Epoch {epoch} Val Final",
            val_score["loss"],
            **{k: v for k, v in val_score.items() if k != "loss"},
        )
        test_score = self.eval(self.test_loader, plot_parity=True)
        self.log_metrics(
            f"Epoch {epoch} Test Final",
            test_score["loss"],
            **{k: v for k, v in test_score.items() if k != "loss"},
        )

        print("Final Test score:", test_score)

        return val_score["loss"]

    def eval(self, dataset, save_eval_dict=False, plot_parity=False):
        """Evaluate the model on a dataset.

        Args:
            dataset (DataLoader): Dataset to evaluate on.
            save_eval_dict (bool): Whether to save evaluation results.
            plot_parity (bool): Whether to plot parity plots.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = {
            "Sintering Temperature": [],
            "Sintering Time": [],
            "Calcination Temperature": [],
            "Calcination Time": [],
        }
        all_targets = {
            "mat": [],
            "Sintering Temperature": [],
            "Sintering Time": [],
            "Calcination Temperature": [],
            "Calcination Time": [],
        }

        with torch.no_grad():
            for it, (
                mat,
                precursor_formulas,
                sint_temp_tensor,
                sint_time_tensor,
                calc_temp_tensor,
                calc_time_tensor,
                mask,
            ) in enumerate(tqdm(dataset, desc="Evaluating")):

                sint_temp_tensor = sint_temp_tensor.to(self.device)
                sint_time_tensor = sint_time_tensor.to(self.device)
                calc_temp_tensor = calc_temp_tensor.to(self.device)
                calc_time_tensor = calc_time_tensor.to(self.device)
                mask = mask.to(self.device)
                # Forward pass
                if self.use_target_only:
                    logits = self.model(mat, [])
                else:
                    logits = self.model(mat, precursor_formulas)
                pred_sint_temp, pred_sint_time, pred_calc_temp, pred_calc_time = (
                    logits[:, 0],
                    logits[:, 1],
                    logits[:, 2],
                    logits[:, 3],
                )

                sint_temp_tensor = torch.nan_to_num(sint_temp_tensor, nan=0.0)
                sint_time_tensor = torch.nan_to_num(sint_time_tensor, nan=0.0)
                calc_temp_tensor = torch.nan_to_num(calc_temp_tensor, nan=0.0)
                calc_time_tensor = torch.nan_to_num(calc_time_tensor, nan=0.0)

                # Apply mask to the predictions and targets
                pred_sint_temp = pred_sint_temp * mask[:, 0]
                pred_sint_time = pred_sint_time * mask[:, 1]
                pred_calc_temp = pred_calc_temp * mask[:, 2]
                pred_calc_time = pred_calc_time * mask[:, 3]

                # Compute losses
                loss_sint_temp = self.loss_fn(
                    pred_sint_temp * mask[:, 0], sint_temp_tensor * mask[:, 0]
                )
                loss_sint_time = self.loss_fn(
                    pred_sint_time, sint_time_tensor * mask[:, 1]
                )
                loss_calc_temp = self.loss_fn(
                    pred_calc_temp, calc_temp_tensor * mask[:, 2]
                )
                loss_calc_time = self.loss_fn(
                    pred_calc_time, calc_time_tensor * mask[:, 3]
                )

                # Total loss
                loss = loss_calc_temp + loss_calc_time

                total_loss += loss.item()
                total_samples += len(mat)

                # Store predictions and targets
                for key, pred, target in [
                    ("Sintering Temperature", pred_sint_temp, sint_temp_tensor),
                    ("Sintering Time", pred_sint_time, sint_time_tensor),
                    ("Calcination Temperature", pred_calc_temp, calc_temp_tensor),
                    ("Calcination Time", pred_calc_time, calc_time_tensor),
                ]:
                    all_preds[key].extend(pred.cpu().numpy().tolist())
                    all_targets[key].extend(target.cpu().numpy().tolist())
                    all_targets["mat"].extend(mat)

                # Plot parity plots
                # self._plot_parity_plots(all_targets, all_preds)

        avg_loss = total_loss / len(dataset)

        # Calculate metrics
        metrics = {}
        for key in all_preds.keys():
            metrics.update(
                {
                    f"mse_{key}": mean_squared_error(all_targets[key], all_preds[key])
                    * self.train_dataset.scaling_params[key]["std"] ** 2,
                    f"mae_{key}": mean_absolute_error(all_targets[key], all_preds[key])
                    * self.train_dataset.scaling_params[key]["std"],
                    f"r2_{key}": r2_score(all_targets[key], all_preds[key]),
                }
            )

        metrics["loss"] = avg_loss

        if plot_parity:
            self._plot_parity_plots(all_targets, all_preds)

        return metrics

    def _plot_parity_plots(self, all_targets, all_preds):
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # Plot Sintering Temperature
        axes[0].scatter(
            all_targets["Sintering Temperature"],
            all_preds["Sintering Temperature"],
            alpha=0.5,
            color="blue",
            label="Predictions",
            s=2,
        )
        for i, txt in enumerate(all_targets["mat"]):
            if i < len(all_targets["Sintering Temperature"]) and i < len(
                all_preds["Sintering Temperature"]
            ):
                axes[0].annotate(
                    txt,
                    (
                        all_targets["Sintering Temperature"][i],
                        all_preds["Sintering Temperature"][i],
                    ),
                    fontsize=3,
                    alpha=0.7,
                )
        ideal_x_sint_temp = [
            min(all_targets["Sintering Temperature"]),
            max(all_targets["Sintering Temperature"]),
        ]
        if (
            ideal_x_sint_temp[0] != ideal_x_sint_temp[1]
        ):  # Avoid plotting a point if min and max are the same
            axes[0].plot(ideal_x_sint_temp, ideal_x_sint_temp, "r--", label="Ideal Fit")
        axes[0].set_title("Sintering Temperature Parity Plot")
        axes[0].set_xlabel("True Values")
        axes[0].set_ylabel("Predicted Values")
        axes[0].legend()

        # Plot Sintering Time
        axes[1].scatter(
            all_targets["Sintering Time"],
            all_preds["Sintering Time"],
            alpha=0.5,
            color="green",
            label="Predictions",
            s=2,
        )
        for i, txt in enumerate(all_targets["mat"]):
            if i < len(all_targets["Sintering Time"]) and i < len(
                all_preds["Sintering Time"]
            ):
                axes[1].annotate(
                    txt,
                    (all_targets["Sintering Time"][i], all_preds["Sintering Time"][i]),
                    fontsize=3,
                    alpha=0.7,
                )
        ideal_x_sint_time = [
            min(all_targets["Sintering Time"]),
            max(all_targets["Sintering Time"]),
        ]
        if ideal_x_sint_time[0] != ideal_x_sint_time[1]:
            axes[1].plot(ideal_x_sint_time, ideal_x_sint_time, "r--", label="Ideal Fit")
        axes[1].set_title("Sintering Time Parity Plot")
        axes[1].set_xlabel("True Values")
        axes[1].set_ylabel("Predicted Values")
        axes[1].legend()

        # Plot Calcination Temperature
        axes[2].scatter(
            all_targets["Calcination Temperature"],
            all_preds["Calcination Temperature"],
            alpha=0.5,
            color="orange",
            label="Predictions",
            s=2,
        )
        for i, txt in enumerate(all_targets["mat"]):
            if i < len(all_targets["Calcination Temperature"]) and i < len(
                all_preds["Calcination Temperature"]
            ):
                axes[2].annotate(
                    txt,
                    (
                        all_targets["Calcination Temperature"][i],
                        all_preds["Calcination Temperature"][i],
                    ),
                    fontsize=3,
                    alpha=0.7,
                )
        ideal_x_calc_temp = [
            min(all_targets["Calcination Temperature"]),
            max(all_targets["Calcination Temperature"]),
        ]
        if ideal_x_calc_temp[0] != ideal_x_calc_temp[1]:
            axes[2].plot(ideal_x_calc_temp, ideal_x_calc_temp, "r--", label="Ideal Fit")
        axes[2].set_title("Calcination Temperature Parity Plot")
        axes[2].set_xlabel("True Values")
        axes[2].set_ylabel("Predicted Values")
        axes[2].legend()

        # Plot Calcination Time
        axes[3].scatter(
            all_targets["Calcination Time"],
            all_preds["Calcination Time"],
            alpha=0.5,
            color="purple",
            label="Predictions",
            s=2,
        )
        for i, txt in enumerate(all_targets["mat"]):
            if i < len(all_targets["Calcination Time"]) and i < len(
                all_preds["Calcination Time"]
            ):
                axes[3].annotate(
                    txt,
                    (
                        all_targets["Calcination Time"][i],
                        all_preds["Calcination Time"][i],
                    ),
                    fontsize=3,
                    alpha=0.7,
                )
        ideal_x_calc_time = [
            min(all_targets["Calcination Time"]),
            max(all_targets["Calcination Time"]),
        ]
        if ideal_x_calc_time[0] != ideal_x_calc_time[1]:
            axes[3].plot(ideal_x_calc_time, ideal_x_calc_time, "r--", label="Ideal Fit")
        axes[3].set_title("Calcination Time Parity Plot")
        axes[3].set_xlabel("True Values")
        axes[3].set_ylabel("Predicted Values")
        axes[3].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "parity_plots_with_labels.png"), dpi=600)
        plt.close(fig)  # Close the figure to free memory

        # Create a new figure for plots without labels
        fig_no_labels, axes_no_labels = plt.subplots(1, 4, figsize=(24, 6))

        # Plot Sintering Temperature (no labels)
        axes_no_labels[0].scatter(
            all_targets["Sintering Temperature"],
            all_preds["Sintering Temperature"],
            alpha=0.5,
            color="blue",
            label="Predictions",
            s=2,
        )
        if ideal_x_sint_temp[0] != ideal_x_sint_temp[1]:
            axes_no_labels[0].plot(
                ideal_x_sint_temp, ideal_x_sint_temp, "r--", label="Ideal Fit"
            )
        axes_no_labels[0].set_title("Sintering Temperature Parity Plot")
        axes_no_labels[0].set_xlabel("True Values")
        axes_no_labels[0].set_ylabel("Predicted Values")
        axes_no_labels[0].legend()

        # Plot Sintering Time (no labels)
        axes_no_labels[1].scatter(
            all_targets["Sintering Time"],
            all_preds["Sintering Time"],
            alpha=0.5,
            color="green",
            label="Predictions",
            s=2,
        )
        if ideal_x_sint_time[0] != ideal_x_sint_time[1]:
            axes_no_labels[1].plot(
                ideal_x_sint_time, ideal_x_sint_time, "r--", label="Ideal Fit"
            )
        axes_no_labels[1].set_title("Sintering Time Parity Plot")
        axes_no_labels[1].set_xlabel("True Values")
        axes_no_labels[1].set_ylabel("Predicted Values")
        axes_no_labels[1].legend()

        # Plot Calcination Temperature (no labels)
        axes_no_labels[2].scatter(
            all_targets["Calcination Temperature"],
            all_preds["Calcination Temperature"],
            alpha=0.5,
            color="orange",
            label="Predictions",
            s=2,
        )
        if ideal_x_calc_temp[0] != ideal_x_calc_temp[1]:
            axes_no_labels[2].plot(
                ideal_x_calc_temp, ideal_x_calc_temp, "r--", label="Ideal Fit"
            )
        axes_no_labels[2].set_title("Calcination Temperature Parity Plot")
        axes_no_labels[2].set_xlabel("True Values")
        axes_no_labels[2].set_ylabel("Predicted Values")
        axes_no_labels[2].legend()

        # Plot Calcination Time (no labels)
        axes_no_labels[3].scatter(
            all_targets["Calcination Time"],
            all_preds["Calcination Time"],
            alpha=0.5,
            color="purple",
            label="Predictions",
            s=2,
        )
        if ideal_x_calc_time[0] != ideal_x_calc_time[1]:
            axes_no_labels[3].plot(
                ideal_x_calc_time, ideal_x_calc_time, "r--", label="Ideal Fit"
            )
        axes_no_labels[3].set_title("Calcination Time Parity Plot")
        axes_no_labels[3].set_xlabel("True Values")
        axes_no_labels[3].set_ylabel("Predicted Values")
        axes_no_labels[3].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "parity_plots_no_labels.png"), dpi=600)
        plt.close(fig_no_labels)  # Close the second figure

    def log_metrics(self, phase, loss, **metrics):
        """Log training metrics.

        Args:
            phase (str): Phase of training (e.g., 'Training', 'Validation').
            loss (float): Loss value.
            **metrics: Additional metrics to log.
        """
        # Create the base log message
        base_message = f"{phase} - Loss: {loss:.4f}"
        file_message = base_message
        # Create a colored message for terminal output
        colored_message = (
            colored(f"{phase}", "cyan", attrs=["bold"])
            + " - "
            + colored(f"Loss: {loss:.4f}", "yellow")
        )

        # Append additional metrics to the messages
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                metric_str = f"{metric_name}: {metric_value:.4f}"
                file_message += f" | {metric_str}"
                colored_message += " | " + colored(metric_str, "magenta")

        # Print the colored message to terminal
        print(colored_message)

        # Log the plain text message to a file
        with open(os.path.join(self.log_dir, "log.txt"), "a") as f:
            f.write(f"{file_message}\n")

    def predict(self, predict_dataloader):
        """Generate predictions for a dataset.

        Args:
            predict_dataloader (DataLoader): DataLoader containing data to predict.
        """
        self.model.eval()
        all_preds = []
        all_compositions = []

        with torch.no_grad():
            # Unpack all 7 items yielded by the dataloader (due to custom_collate_fn)
            for mat, precursor_formulas, _, _, _, _, _ in tqdm(
                predict_dataloader, desc="Predicting"
            ):
                # Pass the correct arguments (target formulas and precursor formulas) to the model
                if self.use_target_only:
                    logits = self.model(mat, [])
                else:
                    logits = self.model(mat, precursor_formulas)
                all_preds.extend(logits.cpu().numpy())
                all_compositions.extend(mat)

        # Convert predictions to DataFrame
        # Un-normalize predictions before saving
        # Assuming self.train_dataset.scaling_params holds the necessary info
        unscaled_preds = []
        keys_in_order = [
            "Sintering Temperature",
            "Sintering Time",
            "Calcination Temperature",
            "Calcination Time",
        ]
        for pred_row in all_preds:
            unscaled_row = {}
            for i, key in enumerate(keys_in_order):
                scaler = self.train_dataset.scaling_params[key]
                scaled_val = pred_row[i]
                # Inverse transform: (scaled_val * std) + mean
                original_val = scaled_val * scaler["std"] + scaler["mean"]
                # Reverse log if needed
                if scaler.get("use_log", False):
                    original_val = np.exp(original_val)
                unscaled_row[key] = original_val
            unscaled_preds.append(unscaled_row)

        # df for predictions.csv
        df_preds_only = pd.DataFrame(unscaled_preds)
        df_preds_only.rename(
            columns={
                "Sintering Temperature": "pred_sint_temp",
                "Sintering Time": "pred_sint_time",
                "Calcination Temperature": "pred_calc_temp",
                "Calcination Time": "pred_calc_time",
            },
            inplace=True,
        )

        df_preds_only["composition"] = all_compositions
        df_preds_only = df_preds_only[
            [
                "composition",
                "pred_sint_temp",
                "pred_sint_time",
                "pred_calc_temp",
                "pred_calc_time",
            ]
        ]

        """# Save the predictions.csv (only compositions and their predictions)
        predictions_csv_path = os.path.join(self.log_dir, 'predictions.csv')
        df_preds_only.to_csv(predictions_csv_path, index=False)
        print(f"Predictions saved to {predictions_csv_path}")"""

        # If the original dataset DataFrame exists, add predictions to it and save
        if hasattr(predict_dataloader, "dataset") and hasattr(
            predict_dataloader.dataset, "df"
        ):
            # Make a copy to avoid modifying the original DataFrame in memory
            df_with_predictions = predict_dataloader.dataset.df.copy()

            # Ensure the number of predictions matches the number of rows in the original DataFrame
            if len(df_preds_only) == len(df_with_predictions):
                # Add new prediction columns
                for col_name in [
                    "pred_sint_temp",
                    "pred_sint_time",
                    "pred_calc_temp",
                    "pred_calc_time",
                ]:
                    # Ensure the column from df_preds_only is correctly aligned if original df has a different index
                    df_with_predictions[col_name] = df_preds_only[col_name].values

                # Save the DataFrame with added prediction columns
                output_path = os.path.join(
                    self.log_dir, "predict_dataset_with_preds.csv"
                )
                df_with_predictions.to_csv(output_path, index=False)
                print(f"Predict dataset with added predictions saved to {output_path}")
            else:
                print(
                    f"Warning: Mismatch in prediction count ({len(df_preds_only)}) and "
                    f"original dataset row count ({len(df_with_predictions)}). "
                    f"Skipping merge and save of predict_dataset_with_preds.csv."
                )
