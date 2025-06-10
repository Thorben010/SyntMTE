"""
This module defines the CompositionMLP model for synthesis condition prediction.
"""

import torch
from torch import nn
import numpy as np
from pymatgen.core.composition import Composition
from skbio.stats.composition import multiplicative_replacement, clr
from torch.utils.data import Dataset

#sys.path.append('/home/thor/code/synth_con_pred')
from src.crabnet import CrabNet
from src.mtencoder import MTEncoder
from src.wrap_crabnet import CrabNetModel
from src.wrap_mtencoder import MTEncoder_model
from utils import all_elements, get_config, look_json_from_path


class LinearLayer(nn.Module):
    """A linear layer with optional activation, dropout, and layer normalization."""

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self, input_dim, output_dim, activation=True, dropout=0.0, layer_norm=True
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Add linear transformation
        self.layers.append(nn.Linear(input_dim, output_dim))

        # Add activation if specified
        if activation:
            self.layers.append(nn.ReLU())

        # Add dropout if specified
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))

        # Add layer normalization if specified
        if layer_norm:
            self.layers.append(nn.LayerNorm(output_dim))

    def forward(self, x):
        """Forward pass for the linear layer."""
        for layer in self.layers:
            x = layer(x)
        return x


class CompositionMLP(nn.Module):  # pylint: disable=too-many-instance-attributes
    """A multi-layer perceptron model for compositions."""

    def __init__(self, model_config: dict = None):
        super().__init__()
        self.input_dim = model_config["input_dim"]
        self.hidden_dim = model_config["hidden_dim"]
        self.num_layers = model_config["num_layers"]
        self.device = model_config["device"]
        self.aggregation_mode = model_config["aggregation_mode"]
        # self.precursor_dict = train_dataset.precursor_dict

        self.embedder_type = model_config["embedder_type"]
        # Safer way to check for mtencoder_path
        if self.embedder_type == "MTEncoder":
            mte_path = "/home/thor/code/synth_con_pred/model_weights"
            loaded_config = look_json_from_path(mte_path)
            config = get_config(loaded_config)
            self.mtencoder = MTEncoder(config)
            self.mtencoder.device = self.device
            self.mtencoder_wrapper = MTEncoder_model(self.mtencoder, mte_path)
        if self.embedder_type == "CrabNet":
            mte_path = "/home/thor/code/synth_con_pred/model_weights"
            loaded_config = look_json_from_path(mte_path)
            config = get_config(loaded_config)
            self.mtencoder = CrabNet(config)
            self.mtencoder.device = self.device
            self.mtencoder_wrapper = CrabNetModel(self.mtencoder, mte_path)

        # Initialize layers potentially needed for different aggregation modes
        # These might be re-initialized or defined lazily in forward if emb_dim is needed
        self.conv1d = None
        self.lstm = None
        self.pool = None  # For global pooling after conv/lstm
        self.attn_linear = None

        self.layers = nn.ModuleList()
        # MLP layers will be defined dynamically in forward after aggregation,
        # as the input dimension depends on the aggregation mode.
        self.mlp_hidden_dim = model_config["hidden_dim"]
        self.mlp_num_layers = model_config["num_layers"]
        self.mlp_dropout = model_config["dropout"]
        self.final_linear = None  # Will be defined in forward

        # The attention layer will be lazily initialized in forward once emb_dim is known.
        # Alternatively, if model_config provides 'embedding_dim', you can initialize it here:
        # self.attn_linear = nn.Linear(model_config['embedding_dim'], 1)

    def forward(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self, target_formulas, precursor_formulas, target_only=False
    ):
        """Forward pass for the CompositionMLP model."""
        # Batch-encode target formulas
        if self.embedder_type == "MTEncoder":
            torch_target_formulas = self.mtencoder_wrapper.encode_materials(
                target_formulas
            )  # (B, emb_dim)
        elif self.embedder_type == "composition":
            torch_target_formulas = self._embedder(target_formulas)
        elif self.embedder_type == "CrabNet":
            torch_target_formulas = self.mtencoder_wrapper.encode_materials(
                target_formulas
            )
        elif self.embedder_type == "clr":
            torch_target_formulas = self.clr_embedder(target_formulas)
        else:
            raise ValueError(f"Unknown embedder type: {self.embedder_type}")

        if not target_only:
            batch_size, emb_dim = torch_target_formulas.shape

            # Ensure precursor_formulas has the same batch dimension as target_formulas
            if (
                isinstance(precursor_formulas, list)
                and not precursor_formulas
                and batch_size > 0
            ):
                # If precursor_formulas is an empty list (e.g., []), and there are targets (B > 0),
                # interpret as no precursors for any target.
                precursor_formulas = [[] for _ in range(batch_size)]
            elif (
                not isinstance(precursor_formulas, list)
                or len(precursor_formulas) != batch_size
            ):
                # precursor_formulas must be a list and its length must match the batch size B.
                raise ValueError(
                    f"Mismatch in batch structure or type: "
                    f"target_formulas has {batch_size} samples. "
                    f"precursor_formulas should be a list of {batch_size} "
                    f"lists of precursor strings. "
                    f"Received type: {type(precursor_formulas)}, "
                    f"value: {precursor_formulas}"
                )

            max_seq_length = 10  # total sequence length including target embedding

            # Flatten all precursor formulas from the batch
            precursor_counts = [len(pf_list) for pf_list in precursor_formulas]
            all_precursors = [pf for pf_list in precursor_formulas for pf in pf_list]

            # Batch encode all precursor formulas at once
            if len(all_precursors) > 0:
                if self.embedder_type == "MTEncoder":
                    all_precursors_emb = self.mtencoder_wrapper.encode_materials(
                        all_precursors
                    )  # (N, emb_dim)
                elif self.embedder_type == "composition":
                    all_precursors_emb = self._embedder(all_precursors)
                elif self.embedder_type == "CrabNet":
                    all_precursors_emb = self.mtencoder_wrapper.encode_materials(
                        all_precursors
                    )
                elif self.embedder_type == "clr":
                    all_precursors_emb = self.clr_embedder(all_precursors)
                else:
                    raise ValueError(f"Unknown embedder type: {self.embedder_type}")
            else:
                all_precursors_emb = torch.empty(0, emb_dim, device=self.device)

            # Split the encoded precursor tensor back into per-sample lists
            precursor_embeddings_split = []
            index = 0
            for count in precursor_counts:
                if count > 0:
                    precursor_embeddings_split.append(
                        all_precursors_emb[index : index + count]
                    )
                    index += count
                else:
                    precursor_embeddings_split.append(
                        torch.empty(0, emb_dim, device=self.device)
                    )

            # Build the sequence for each sample:
            # Prepend the target embedding to the precursor embeddings,
            # then pad (or truncate) the sequence to max_seq_length.
            sequence_tensor = torch.zeros(
                batch_size, max_seq_length, emb_dim, device=self.device
            )
            for i in range(batch_size):
                seq_list = [torch_target_formulas[i]]
                if precursor_embeddings_split[i].size(0) > 0:
                    seq_list.extend(torch.unbind(precursor_embeddings_split[i], dim=0))
                seq_len = len(seq_list)
                if seq_len > max_seq_length:
                    seq_list = seq_list[:max_seq_length]
                    seq_len = max_seq_length
                seq_tensor = torch.stack(seq_list, dim=0)  # (seq_len, emb_dim)
                sequence_tensor[i, :seq_len, :] = seq_tensor

            # --- Aggregation Pooling ---
            aggregation_output_dim = emb_dim  # Default
            if self.aggregation_mode == "attention":
                # Instead of mean pooling, use an attention mechanism over the sequence dimension.
                if self.attn_linear is None:
                    self.attn_linear = nn.Linear(emb_dim, 1).to(self.device)
                attn_scores = self.attn_linear(
                    sequence_tensor
                )  # (B, max_seq_length, 1)
                attn_weights = torch.softmax(
                    attn_scores, dim=1
                )  # (B, max_seq_length, 1)
                final_representation = (sequence_tensor * attn_weights).sum(
                    dim=1
                )  # (B, emb_dim)
            elif self.aggregation_mode == "mean":
                # Simple mean pooling (ignoring padding for simplicity here, might need masking for correctness)
                final_representation = sequence_tensor.mean(dim=1)  # (B, emb_dim)
            elif self.aggregation_mode == "max":
                # Max pooling over the sequence dimension
                # Use .values to get the max values, ignore indices
                final_representation = torch.max(
                    sequence_tensor, dim=1
                ).values  # (B, emb_dim)
            elif self.aggregation_mode == "sum":
                # Sum pooling over the sequence dimension
                final_representation = torch.sum(sequence_tensor, dim=1)  # (B, emb_dim)
            elif self.aggregation_mode == "mean_max":
                mean_pooled = sequence_tensor.mean(dim=1)
                max_pooled = torch.max(sequence_tensor, dim=1).values
                final_representation = torch.cat(
                    (mean_pooled, max_pooled), dim=1
                )  # (B, emb_dim * 2)
                aggregation_output_dim = emb_dim * 2
            elif self.aggregation_mode == "precursor_target_concat":
                precursor_embs = sequence_tensor[
                    :, 1:, :
                ]  # (B, max_seq_length-1, emb_dim)
                mean_precursors = torch.mean(precursor_embs, dim=1)  # (B, emb_dim)
                final_representation = torch.cat(
                    (torch_target_formulas, mean_precursors), dim=1
                )  # (B, emb_dim * 2)
                aggregation_output_dim = emb_dim * 2
            elif self.aggregation_mode == "concat":
                # Flatten the entire sequence tensor (which includes the target at index 0).
                # sequence_tensor shape: (B, max_seq_length, emb_dim)
                batch_size = sequence_tensor.shape[0]  # Get batch size
                final_representation = sequence_tensor.reshape(
                    batch_size, -1
                )  # Reshape to (B, max_seq_length * emb_dim)
                aggregation_output_dim = emb_dim * max_seq_length
                # Note: The original torch.cat operation was removed as it was incompatible
                # and didn't align with the aggregation_output_dim calculation.

            elif self.aggregation_mode == "conv":
                if self.conv1d is None:
                    # Example: Simple conv layer. Kernel size 3, same padding.
                    # Output channels = emb_dim (can be changed)
                    self.conv1d = nn.Conv1d(
                        emb_dim, emb_dim, kernel_size=3, padding=1
                    ).to(self.device)
                    self.pool = nn.AdaptiveMaxPool1d(1)  # Global Max Pooling
                # Conv1d expects (B, C, L), so permute dimensions
                conv_in = sequence_tensor.permute(
                    0, 2, 1
                )  # (B, emb_dim, max_seq_length)
                conv_out = self.conv1d(conv_in)  # (B, emb_dim, max_seq_length)
                pooled_out = self.pool(conv_out).squeeze(-1)  # (B, emb_dim)
                final_representation = pooled_out
                aggregation_output_dim = (
                    emb_dim  # Stays the same if out_channels = emb_dim
                )
            elif self.aggregation_mode == "lstm":
                if self.lstm is None:
                    # Example: Simple LSTM layer.
                    # Hidden size = emb_dim (can be changed)
                    self.lstm = nn.LSTM(emb_dim, emb_dim, batch_first=True).to(
                        self.device
                    )
                # LSTM input: (B, seq_len, input_size)
                _, (h_n, _) = self.lstm(sequence_tensor)
                # Use the last hidden state of the last layer
                final_representation = h_n[-1]  # (B, emb_dim)
                aggregation_output_dim = (
                    emb_dim  # Stays the same if hidden_size = emb_dim
                )
            else:
                raise ValueError(f"Unknown aggregation mode: {self.aggregation_mode}")
        else:
            final_representation = torch_target_formulas
            aggregation_output_dim = emb_dim

        # --- MLP Head ---
        # Define MLP layers dynamically based on aggregation output dim
        if not self.layers:  # Initialize MLP layers only once
            current_dim = aggregation_output_dim
            self.layers.append(
                nn.BatchNorm1d(current_dim)
            )  # .to(self.device) # BatchNorm should be on the same device
            self.layers.append(nn.Dropout(p=self.mlp_dropout))  # .to(self.device)
            for _ in range(self.mlp_num_layers - 1):
                self.layers.append(
                    LinearLayer(current_dim, self.mlp_hidden_dim)
                )  # .to(self.device)
                self.layers.append(
                    nn.BatchNorm1d(self.mlp_hidden_dim)
                )  # .to(self.device)
                self.layers.append(nn.ReLU())  # .to(self.device)
                current_dim = self.mlp_hidden_dim
            self.final_linear = nn.Linear(current_dim, 4)  # .to(self.device)
            # Move all dynamically created layers to the correct device
            self.layers = self.layers.to(self.device)
            self.final_linear = self.final_linear.to(self.device)

        # Pass the final representation through the MLP layers for final prediction
        x = final_representation  # .to(self.device) # Already on device
        for layer in self.layers[:-1]:  # Apply layers up to the last ReLU/BatchNorm
            x = layer(x)
        logits = self.final_linear(x)  # Apply final linear layer

        return logits

    def _embedder(self, formulas):
        # Input is a list of formulas (strings)
        # Uses 'all_elements' imported from 'utils' (due to 'from utils import *')
        compositional_emb_dim = len(all_elements)

        if not formulas:  # Handle empty list of formulas
            return torch.empty(
                (0, compositional_emb_dim), dtype=torch.float32, device=self.device
            )

        mt_embeddings = []
        for formula in formulas:
            comp = Composition(formula)
            element_fractions = comp.get_el_amt_dict()
            total = sum(element_fractions.values())
            # Ensure total is not zero to avoid division by zero for safety
            normalized_element_fractions = {
                el: amt / total if total > 0 else 0.0
                for el, amt in element_fractions.items()
            }
            vector = [normalized_element_fractions.get(el, 0.0) for el in all_elements]
            mt_embeddings.append(vector)
        return torch.tensor(mt_embeddings, dtype=torch.float32, device=self.device)

    def clr_embedder(self, formulas):
        """CLR embedder for formulas."""
        # Always assume len(all_elements) elements (e.g. 118)
        compositional_emb_dim = len(all_elements)

        if not formulas:  # Handle empty list of formulas
            return torch.empty(
                (0, compositional_emb_dim), dtype=torch.float32, device=self.device
            )

        # Build a matrix of normalized element fractions
        embeddings = []
        for formula in formulas:
            comp = Composition(formula)
            element_fractions = comp.get_el_amt_dict()
            total = sum(element_fractions.values())
            normalized = {
                el: (amt / total) if total > 0 else 0.0
                for el, amt in element_fractions.items()
            }
            # Construct a vector in the order of all_elements
            row = [normalized.get(el, 0.0) for el in all_elements]
            embeddings.append(row)

        # Convert to numpy array and replace zeros for compositional clr
        embeddings_matrix = np.array(
            embeddings, dtype=float
        )  # shape: (n_samples, compositional_emb_dim)
        replaced_matrix = multiplicative_replacement(
            embeddings_matrix
        )  # still (n_samples, compositional_emb_dim)

        # Apply centered log-ratio transform → (n_samples, compositional_emb_dim)
        clr_matrix = clr(replaced_matrix)

        return torch.tensor(clr_matrix, dtype=torch.float32, device=self.device)

    def predict(self, anchor, candidate):
        """This function is for a singular anchor and candidate list"""
        if self.embedder_type == "composition":
            torch_anchor = self._embedder([anchor]).to(self.device)
        elif self.embedder_type == "clr":
            torch_anchor = self.clr_embedder([anchor]).to(self.device)
        else:
            torch_anchor = self.mtencoder_wrapper.encode_materials([anchor]).to(
                self.device
            )

        torch_anchor = torch_anchor.to(self.device)
        x = torch_anchor
        for layer in self.layers:
            x = layer(x)
        logits = self.final_linear(x)
        probability = torch.sigmoid(logits).reshape(-1)
        sorted_indices = torch.argsort(probability, descending=True)
        sorted_candidates = [candidate[i] for i in sorted_indices]
        return probability, sorted_candidates
