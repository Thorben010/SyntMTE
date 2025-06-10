"""Wrapper module for CrabNet model functionality."""
import json
import os

import torch
from pymatgen.core import Composition


class CrabNetModel:
    """A wrapper class for CrabNet model with encoding and prediction capabilities."""

    def __init__(self, mtencoder, config_path):
        """
        Initialize the CrabNet model wrapper.
        
        Args:
            mtencoder: The encoder model instance
            config_path: Path to configuration (currently unused)
        """
        self.model = mtencoder
        # self.look_pth_from_path(config_path)
        self.load_lookup()

    def look_pth_from_path(self, path):
        """
        Load a .pth file from the given path.
        
        Args:
            path: Directory path containing .pth files
            
        Returns:
            Result of loading the network from the found .pth file
        """
        pth_files = [f for f in os.listdir(path) if f.endswith(".pth")]
        if not pth_files:
            raise FileNotFoundError(f"No .pth files found in directory {path}")
        return self.load_network(os.path.join(path, pth_files[0]))

    def load_lookup(self):
        """Load the periodic table element symbols and create element-to-id mapping."""
        self.all_symbols = [
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
        self.element_to_id = {
            symbol: index + 1 for index, symbol in enumerate(self.all_symbols)
        }

    def load_json_from_path(self, path):
        """
        Load JSON data from the specified file path.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_network(self, path):
        """
        Load a neural network model from the specified path.
        
        Args:
            path: Path to the model file
        """
        network = torch.load(path, weights_only=True)
        filtered_state_dict = {
            k: v for k, v in network["weights"].items() if "output_nn" not in k
        }
        self.model.load_state_dict(filtered_state_dict)
        print(f"Model loaded from {path}")
        loaded_layers = self.count_loaded_subkeys(
            self.model.state_dict(), filtered_state_dict
        )
        print(f"Number of layers loaded: {loaded_layers}")

    def count_loaded_subkeys(self, model_state_dict, loaded_state_dict):
        """
        Count the number of successfully loaded model layers.
        
        Args:
            model_state_dict: Current model state dictionary
            loaded_state_dict: Loaded state dictionary from file
            
        Returns:
            Number of successfully loaded layers
        """
        return sum(
            1
            for key in model_state_dict
            if (
                modified_key := (
                    "module." + key if "module." + key in loaded_state_dict else key
                )
            )
            in loaded_state_dict
            and torch.equal(
                model_state_dict[key].to(self.model.device),
                loaded_state_dict[modified_key].to(self.model.device),
            )
        )

    def forward_test(self):
        """
        Perform a forward pass test with random input data.
        
        Returns:
            Model output with embeddings
        """
        model_input = {
            "src_masked": torch.randint(1, 10, (512, 10)).long().to(self.model.device),
            "frac": torch.rand((512, 10)).to(self.model.device),
        }
        return self.model(model_input, embeddings=True)

    def forward(self, encoded_materials):
        """
        Perform forward pass with encoded materials.
        
        Args:
            encoded_materials: List of encoded material tensors
            
        Returns:
            Model output with embeddings
        """
        model_input = {
            "src_masked": torch.stack([encoded[0] for encoded in encoded_materials]),
            "frac": torch.stack([encoded[1] for encoded in encoded_materials]),
        }
        return self.model(model_input, embeddings=True)

    def encode_materials(self, materials_list):
        """
        Encode a list of material formulas for model input.
        
        Args:
            materials_list: List of chemical formula strings
            
        Returns:
            Model output for the encoded materials
        """
        # encoded_materials = [self.encode_formula(formula) for formula in materials_list]
        encoded_materials = []
        for formula in materials_list:
            encoded_material = self.encode_formula(formula)
            encoded_materials.append(encoded_material)

        max_len = max(len(encoded[0]) for encoded in encoded_materials)

        for i, encoded_material in enumerate(encoded_materials):
            pad_len = max_len - len(encoded_material[0])
            if pad_len > 0:
                encoded_materials[i] = (
                    torch.cat(
                        [
                            encoded_material[0],
                            torch.zeros(pad_len, dtype=torch.long).to(
                                self.model.device
                            ),
                        ]
                    ),
                    torch.cat(
                        [
                            encoded_material[1],
                            torch.zeros(pad_len).to(self.model.device),
                        ]
                    ),
                )
        return self.forward(encoded_materials)

    def encode_formula(self, formula):
        """
        Encode a chemical formula into element IDs and fractions.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Tuple of (element_ids_tensor, fractions_tensor)
        """
        fractions = []
        element_ids = []
        composition = Composition(formula)
        total_fraction = sum(composition.get_el_amt_dict().values())

        for element, fraction in composition.get_el_amt_dict().items():
            normalized_fraction = fraction / total_fraction
            fractions.append(normalized_fraction)
            element_ids.append(self.element_to_id[element])
        # fractions.insert(0, 0)
        # element_ids.insert(0, 119)

        return (
            torch.tensor(element_ids).long().to(self.model.device),
            torch.tensor(fractions).to(self.model.device),
        )


""" # Initialize the model
mtencoder = MTEncoder_model(f'{os.getcwd()}/data/mtencoder')
mtencoder.model.eval()
test_material = ["FeNa2O4"]
output1 = mtencoder.encode_materials(test_material)
output2 = mtencoder.encode_materials(test_material)

print("Output 1:", output1)
print("Output 2:", output2)
print("Difference:", torch.abs(output1 - output2).sum())
 """
