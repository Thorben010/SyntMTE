"""Utility functions for the synthesis condition prediction model."""
import os
import json
import torch
import numpy as np


all_elements = [
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


def look_json_from_path(path):
    """Looks for a JSON file in the given path and returns its content."""
    json_files = [f for f in os.listdir(path) if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory {path}")
    with open(os.path.join(path, json_files[0]), "r", encoding="utf-8") as f:
        return json.load(f)


def get_config(config):
    """Gets the model and trainer configuration."""
    return {
        "model": {
            "decoder": "cpd",
            "d_model": config["model"]["d_model"],
            "N": config["model"]["N"],
            "encoder_ff": config["model"]["encoder_ff"],
            "heads": config["model"]["heads"],
            "residual_nn_dim": config["model"]["residual_nn_dim"],
            "branched_ffnn": config["model"]["branched_ffnn"],
            "dropout": 0,
            "special_tok_zero_fracs": False,
            "numeral_embeddings": config["model"]["numeral_embeddings"],
        },
        "trainer": {
            "swa_start": 1,
            "n_elements": "infer",
            "masking": False,
            "fraction_to_mask": 0,
            "cpd_token": config["trainer"]["cpd_token"],
            "eos_token": config["trainer"]["eos_token"],
            "base_lr": 5e-8,
            "mlm_loss_weight": None,
            "delay_scheduler": 0,
        },
        "max_epochs": 200,
        "sampling_prob": None,
        "task_types": None,
        "save_every": 100000,
        "task_list": None,
        "task_dim": None,
        "eval": True,
        "wandb": False,
        "batchsize": None,
    }


def custom_collate_fn(batch):
    """
    Custom collate function to handle batching when precursor_formulas
    are lists of string items.

    Args:
        batch (list): List of samples as returned by __getitem__.

    Returns:
        Tuple containing:
            - list of target formulas (str)
            - list of precursor_formulas (list of str)
            - stacked tensor for sintering temperature
            - stacked tensor for sintering time
            - stacked tensor for calcination temperature
            - stacked tensor for calcination time
            - stacked tensor for mask
    """
    (
        mats,
        precursor_formulas,
        sint_temp_tensors,
        sint_time_tensors,
        calc_temp_tensors,
        calc_time_tensors,
        masks,
    ) = zip(*batch)

    sint_temp_batch = torch.stack(sint_temp_tensors)
    sint_time_batch = torch.stack(sint_time_tensors)
    calc_temp_batch = torch.stack(calc_temp_tensors)
    calc_time_batch = torch.stack(calc_time_tensors)
    mask_batch = torch.stack(masks)

    return (
        list(mats),
        list(precursor_formulas),
        sint_temp_batch,
        sint_time_batch,
        calc_temp_batch,
        calc_time_batch,
        mask_batch,
    )


class LeanScaler:
    """A lean scaler for standard scaling and log transformation."""

    def __init__(self, mean, std, use_log=False):
        """
        Initialize the scaler.

        Parameters:
            mean (float): The mean value used for scaling.
            std (float): The standard deviation used for scaling.
            use_log (bool): If True, perform log transformation before scaling.
        """
        self.mean = mean
        self.std = std
        self.use_log = use_log

    def transform(self, x):
        """
        Apply scaling to the input data.

        Parameters:
            x (float or np.ndarray): Input data to transform.

        Returns:
            Transformed data.
        """
        # If use_log is True, apply logarithmic transformation first.
        if self.use_log:
            x = np.log(x)
        # Then apply standard scaling.
        return (x - self.mean) / self.std

    def inverse_transform(self, x_scaled):
        """
        Reverse the scaling operation.

        Parameters:
            x_scaled (float or np.ndarray): Scaled data to reverse transform.

        Returns:
            Data in the original scale.
        """
        # Reverse standard scaling.
        x = x_scaled * self.std + self.mean
        # If use_log is True, reverse the log transformation.
        if self.use_log:
            x = np.exp(x)
        return x
