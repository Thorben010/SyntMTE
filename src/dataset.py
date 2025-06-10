import pandas as pd
import pickle
import ast
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import product
import random


# Define a safe version of literal_eval that handles non-string inputs like np.nan
def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # Return empty list or None if string is malformed or not a valid literal
            return []  # Or potentially None, depending on desired handling
    # Handle non-string inputs (like np.nan) by returning an empty list
    elif pd.isna(val):
        return []  # Or None
    # Optionally handle other types if necessary
    return val  # Or return [], depends on expected types


class MaterialDataset(Dataset):
    """
    Dataset for materials.

    Args:
        file_path (str): Path to the CSV file.
        scaling_params (dict): Dictionary with scaling parameters for each field.
            Expected keys:
                - 'temperature(K)'
                - 'seebeck_coefficient(μV/K)'
                - 'electrical_conductivity(S/m)'
                - 'thermal_conductivity(W/mK)'
            Each value should be a dict with:
                - 'mean': (float) mean value for scaling
                - 'std': (float) standard deviation for scaling
                - 'use_log': (bool) whether to apply log transform before scaling
    """

    def __init__(self, file_path: str, scaling_params: dict, inference: bool = False):
        # Load CSV file
        print(f"Loading data from {file_path}")
        self.df = pd.read_csv(file_path, na_values=["nan"])
        self.length = len(self.df)
        print(f"Dataset length: {self.length}")
        self.scaling_params = scaling_params
        self.inference = inference
        # Convert precursor formulas and target formula from string representation using safe_literal_eval
        self.df["precursor_formulas"] = self.df["precursor_formulas"].apply(
            safe_literal_eval
        )
        # self.df['target_formula'] = self.df['target_formula'].apply(safe_literal_eval)

    def __len__(self):
        return self.length

    def _scale_value(self, value, params):
        """
        Apply (optional log) then standard scaling to a single value.
        """
        # Ensure value is a float
        x = float(value)
        # Apply log transformation if requested.
        if params.get("use_log", False):
            # Avoid taking log of non-positive numbers by adding a small epsilon.
            if x <= 0:
                x = 1e-10
            x = np.log(x)
        # Standard scaling
        return (x - params["mean"]) / params["std"]

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        # Convert sample to dictionary for safety
        sample_dict = sample.to_dict()
        # Convert "Formula" to string directly (taking first element)
        mat = sample_dict["target_formula"]
        precursor_formulas = sample_dict.get("precursor_formulas", [])

        if self.inference:
            return (
                mat,
                precursor_formulas,
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
            )

        # Apply transformations using the provided scaling parameters.
        sint_temp = self._scale_value(
            sample_dict.get("Sintering Temperature", 0.0),
            self.scaling_params["Sintering Temperature"],
        )
        sint_time = self._scale_value(
            sample_dict.get("Sintering Time", 0.0),
            self.scaling_params["Sintering Time"],
        )
        calc_temp = self._scale_value(
            sample_dict.get("Calcination Temperature", 0.0),
            self.scaling_params["Calcination Temperature"],
        )
        calc_time = self._scale_value(
            sample_dict.get("Calcination Time", 0.0),
            self.scaling_params["Calcination Time"],
        )

        # Convert scaled values to torch tensors
        sint_temp_tensor = torch.tensor(sint_temp, dtype=torch.float32)
        sint_time_tensor = torch.tensor(sint_time, dtype=torch.float32)
        calc_temp_tensor = torch.tensor(calc_temp, dtype=torch.float32)
        calc_time_tensor = torch.tensor(calc_time, dtype=torch.float32)

        mask = torch.tensor(
            [
                1 if pd.notna(sample_dict.get("Sintering Temperature")) else 0,
                1 if pd.notna(sample_dict.get("Sintering Time")) else 0,
                1 if pd.notna(sample_dict.get("Calcination Temperature")) else 0,
                1 if pd.notna(sample_dict.get("Calcination Time")) else 0,
            ],
            dtype=torch.float32,
        )

        return (
            mat,
            precursor_formulas,
            sint_temp_tensor,
            sint_time_tensor,
            calc_temp_tensor,
            calc_time_tensor,
            mask,
        )


"""# Example usage:
if __name__ == "__main__":
    # Define scaling parameters for each field.
    scaling_params = {
        'Sintering Temperature': {'mean': 1406.53, 'std': 13449.01, 'use_log': False},
        'Sintering Time': {'mean': 13.37, 'std': 19.20, 'use_log': False},
        'Calcination Temperature': {'mean': 1248.44, 'std': 11743.82, 'use_log': False},
        'Calcination Time': {'mean': 12.57, 'std': 25.05, 'use_log': False}
    }
    
    dataset = MaterialDataset(file_path='/home/thor/code/synth_con_pred/data/conditions/datasets/train.csv',
                                scaling_params=scaling_params)
    
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)
    
    for batch in dataloader:
        mats, precursor_formulas, sint_temp, sint_time, calc_temp, calc_time, mask = batch
        print("Target formulas:", mats)
        print("Precursor formulas:", precursor_formulas)
        print("Sintering Temperature Tensor:", sint_temp)
        # Process one batch only for demonstration
        break
"""
