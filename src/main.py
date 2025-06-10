"""
Main script for training, tuning, and predicting with the CompositionMLP model.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import GPUtil
import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset import MaterialDataset
from models.composition_mlp import CompositionMLP
from trainer import Trainer
from utils import custom_collate_fn

# get list of GPUs, pick the one with the smallest memoryUsed
gpu = min(GPUtil.getGPUs(), key=lambda g: g.memoryUsed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu.id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using GPU {gpu.id}, memoryUsed={gpu.memoryUsed}MB")


class AleatoricLoss(nn.Module):
    """Aleatoric loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, mu, log_var, target):
        """Calculate the aleatoric loss."""
        # Calculate the element-wise loss
        loss = 0.5 * torch.exp(-log_var) * (target - mu) ** 2 + 0.5 * log_var
        # Return the average loss over the batch
        return torch.mean(loss)


def get_model_config(
    embedder_type, num_layers=2, dropout=0.0, aggregation_mode="attention"
):
    """Get model configuration based on embedder type.

    Args:
        embedder_type (str): Type of embedder to use
        num_layers (int): Number of layers in the model
        dropout (float): Dropout rate
        aggregation_mode (str): Aggregation mode to use ('attention', 'mean', etc.)

    Returns:
        dict: Model configuration dictionary
    """
    input_dim = (
        512
        if embedder_type in ["MTEncoder", "CrabNet"]
        else 118 if embedder_type == "composition" else None
    )
    return {
        "num_layers": num_layers,
        "dropout": dropout,
        "hidden_dim": 512,
        "embedder_type": embedder_type,
        "input_dim": input_dim,
        "device": "cuda:0",
        "aggregation_mode": aggregation_mode,
    }


def get_trainer_config(  # pylint: disable=too-many-arguments
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=3**2,
    learning_rate=0.00001,
):
    """Get trainer configuration.

    Args:
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer

    Returns:
        dict: Trainer configuration dictionary
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "loss_fn": nn.L1Loss(),
        "model": model,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "epochs": 200,
        "device": model.device,
        "use_cyclic_lr": False,
        "base_lr": learning_rate,
        "max_lr": learning_rate * 1000,
        "cyclic_mode": "triangular",
        "log_dir": "/home/thor/code/synth_con_pred/logs/",
    }


def objective(  # pylint: disable=too-many-arguments,too-many-locals
    trial, embedder_type, train_dataset, val_dataset, test_dataset, use_target_only, args
):
    """Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object
        embedder_type (str): Type of embedder to use
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        use_target_only (bool): Whether to use only target material
            representation for regression
        args: Command line arguments

    Returns:
        float: Validation F1 score
    """
    num_layers = trial.suggest_int("num_layers", 1, 5)
    # dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 10)
    batch_size = batch_size**2
    # aggregation_mode = trial.suggest_categorical('aggregation_mode', ['attention', 'mean', 'max', 'sum', 'mean_max', 'conv', 'lstm'])
    aggregation_mode = "mean"

    model_config = get_model_config(
        embedder_type, num_layers, aggregation_mode=aggregation_mode
    )
    model = CompositionMLP(model_config)

    # --- Dummy forward pass to initialize lazy layers ---
    dummy_target = ["NaFe"] * 8
    dummy_precursors = [["Na", "Fe"]] * 8  # Batch of 8
    model.to(model.device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_target, dummy_precursors)
    # --- End dummy forward pass ---

    trainer_config = get_trainer_config(
        model, train_dataset, val_dataset, test_dataset, batch_size, learning_rate
    )
    trainer = Trainer(
        trainer_config,
        model_config,
        [embedder_type, args.dataset],
        use_target_only=use_target_only,
    )
    val_f1 = trainer.train(
        epochs=trainer_config["epochs"], eval_interval=1, get_best=False
    )
    return val_f1


def load_checkpoint(model, checkpoint_path):
    """Load a model checkpoint."""
    try:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        # Adjust this key based on how the checkpoint is saved
        model_state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(model_state_dict)
        print("Checkpoint loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading checkpoint: {e}")
        return None


def predict_dataset(predict_dataloader, scaling_params, trainer):
    """Predict on a dataset."""
    pred_dataset = MaterialDataset(
        file_path=args.predict_dataset, scaling_params=scaling_params, inference=True
    )
    predict_dataloader = DataLoader(
        pred_dataset, batch_size=4000, shuffle=False, collate_fn=custom_collate_fn
    )
    trainer.predict(predict_dataloader)


def main(args):  # pylint: disable=too-many-locals,redefined-outer-name
    """Main function to run training or hyperparameter tuning.

    Args:
        args: Command line arguments
    """
    # Construct absolute path for the dataset directory
    dataset_dir = args.dataset
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.getcwd(), dataset_dir)

    # Check for existence of normalization file and provide a clear error
    normalization_path = os.path.join(dataset_dir, "normalization.json")
    if not os.path.exists(normalization_path):
        print(
            f"Error: 'normalization.json' not found at the specified path: {normalization_path}"
        )
        print(
            "Please ensure the --dataset argument points to a valid directory "
            "containing 'normalization.json'."
        )
        sys.exit(1)

    with open(normalization_path, "r", encoding="utf-8") as f:
        scaling_params = json.load(f)

    # Use the unified dataset_dir for all dataset files
    train_dataset = MaterialDataset(
        file_path=os.path.join(dataset_dir, "train.csv"), scaling_params=scaling_params
    )
    val_dataset = MaterialDataset(
        file_path=os.path.join(dataset_dir, "val.csv"), scaling_params=scaling_params
    )
    test_dataset = MaterialDataset(
        file_path=os.path.join(dataset_dir, "test.csv"), scaling_params=scaling_params
    )

    if args.mode == "tune":
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(),
            study_name=f'{args.embedder_type}_{datetime.now().strftime("%Y%m%d%H%M%S")}',
            storage=f"sqlite:///{os.getcwd()}/optuna.db",
        )
        study.optimize(
            lambda trial: objective(
                trial,
                args.embedder_type,
                train_dataset,
                val_dataset,
                test_dataset,
                args.use_target_only,
                args,
            ),
            n_trials=200,
        )
        print("Best hyperparameters: ", study.best_params)

    elif args.mode in ("train", "predict"):
        model_config = get_model_config(
            args.embedder_type, aggregation_mode=args.aggregation_mode
        )
        model = CompositionMLP(model_config)

        # --- Dummy forward pass to initialize lazy layers ---
        print("Performing dummy forward pass to initialize layers...")

        dummy_target = ["NaFe"] * 8
        dummy_precursors = [["Na", "Fe"]] * 8  # Batch of 8

        # Ensure model is on the correct device and in eval mode
        model.to(model.device)
        model.eval()

        with torch.no_grad():  # No need to track gradients
            # The actual output value doesn't matter here
            _ = model(dummy_target, dummy_precursors)
        print("Dummy forward pass successful.")
        # --- End dummy forward pass ---

        trainer_config = get_trainer_config(
            model,
            train_dataset,
            val_dataset,
            test_dataset,
            learning_rate=(
                args.learning_rate if args.mode == "train" else 0.00001
            ),  # Use CLI LR for train mode
        )

        trainer = Trainer(
            trainer_config,
            model_config,
            [args.embedder_type, args.dataset],
            use_target_only=args.use_target_only,
        )

        if args.checkpoint_path:
            model = load_checkpoint(model, args.checkpoint_path)

        if args.mode == "predict":
            predict_dataset(trainer, scaling_params, trainer)

        else:
            _metrics = trainer.train(
                epochs=trainer_config["epochs"], eval_interval=1, get_best=False
            )
            predict_dataset(trainer, scaling_params, trainer)

    else:
        print("Invalid mode. Please use 'train', 'tune', or 'predict'.")
        sys.exit(1)


if __name__ == "__main__":
    # Set random seeds based on current time
    current_time = int(time.time())
    random.seed(current_time)
    np.random.seed(current_time)
    torch.manual_seed(current_time)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_time)

    parser = argparse.ArgumentParser(
        description="Train or tune the CompositionMLP model."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "tune", "predict"],
        default="train",
        help="Mode to run: 'train' for standard training, 'tune' for "
        "hyperparameter optimization, 'predict' for prediction.",
    )
    parser.add_argument(
        "--embedder_type",
        choices=["CrabNet", "composition", "MTEncoder", "clr"],
        default="MTEncoder",
        help="Type of embedder to use in the model.",
    )
    parser.add_argument(
        "--aggregation_mode",
        choices=[
            "attention",
            "mean",
            "max",
            "sum",
            "mean_max",
            "conv",
            "lstm",
            "precursor_target_concat",
            "concat",
        ],
        default="mean",
        help="Type of aggregation to use for target and precursors.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/conditions/random_split",
        help="Dataset path.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to load model checkpoint from.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0000439204,
        help="Learning rate for the optimizer (only used in 'train' mode).",
    )
    parser.add_argument(
        "--use_target_only",
        action="store_true",
        default=False,
        help="Use only target material representation for regression.",
    )
    parser.add_argument(
        "--predict_dataset",
        type=str,
        default="data/conditions/random_split/test.csv",
        help="Path to the dataset to predict.",
    )

    # Parse arguments
    args = parser.parse_args()
    main(args)
