import numpy as np
import torch
from torch import nn

"""
This code is built upon CrabNet: https://github.com/anthony-wang/CrabNet
The original code is modified to fit the needs of the SyntMTE project.
"""

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32  # test


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    """

    def __init__(
        self, input_dim, output_dim, hidden_layer_dims, branched_ffnn, dropout
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super().__init__()
        dims = [input_dim] + hidden_layer_dims

        self.branched_ffnn = branched_ffnn

        if dropout:
            print("Dropout is used!")
            self.dropout = nn.ModuleList(
                [nn.Dropout(dropout) for i in range(len(dims) - 1)]
            )
        else:
            self.dropout = None

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.res_fcs = nn.ModuleList(
            [
                (
                    nn.Linear(dims[i], dims[i + 1], bias=False)
                    if (dims[i] != dims[i + 1])
                    else nn.Identity()
                )
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        if self.branched_ffnn:
            for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
                fea = act(fc(fea)) + res_fc(fea)
        else:
            if self.dropout:
                for fc, act, drop in zip(self.fcs, self.acts, self.dropout):
                    fea = act(fc(drop(fea)))
            else:
                for fc, act in zip(self.fcs, self.acts):
                    fea = act(fc(fea))

        return self.fc_out(fea)

    def __repr__(self):
        """Return a string representation of the model."""
        return f"{self.__class__.__name__}"


class Embedder(nn.Module):
    """Embedder class for element embeddings."""
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        """elem_dir = 'data/element_properties'
        # # Choose what element information the model receives
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding
        # mat2vec = f'{elem_dir}/onehot.csv'  # onehot encoding (atomic number)
        # mat2vec = f'{elem_dir}/random_200.csv'  # random vec for elements

        cbfv = pd.read_csv(mat2vec, index_col=0).values"""

        cbfv = np.random.rand(118, 200)
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))

        cpd = torch.randn((3, feat_size))
        cpd = cpd / np.sqrt(feat_size)
        cat_array = np.concatenate([zeros, cbfv, cpd])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array).to(
            self.compute_device, dtype=data_type_torch
        )
        self.cfv = nn.Embedding(cbfv.shape[0] + 1, self.d_model)

    def forward(self, src):
        """Forward pass for the Embedder."""
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb


class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(
            0, self.resolution - 1, self.resolution, requires_grad=False
        ).view(self.resolution, 1)
        fraction = (
            torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False)
            .view(1, self.d_model)
            .repeat(self.resolution, 1)
        )

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Forward pass for the FractionalEncoder."""
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x)) ** 2
            # clamp x[x > 1] = 1
            x = torch.clamp(x, max=1)
            # x = 1 - x  # for sinusoidal encoding at x=0
        # clamp x[x < 1/self.resolution] = 1/self.resolution
        x = torch.clamp(x, min=1 / self.resolution)
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out


class FractionalEncoderNumeral(nn.Module):
    """
    Ablation fractional encoding.
    """

    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.compute_device = compute_device
        self.project = nn.Linear(1, d_model).to(self.compute_device)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        """Forward pass for the FractionalEncoderNumeral."""
        x = x.clone()
        x = self.act(self.project(x.unsqueeze(dim=-1)))
        return x


class Encoder(nn.Module):
    """Encoder module."""
    def __init__(
        self,
        d_model,
        N,
        heads,
        encoder_ff,
        numeral_embeddings,
        special_tok_zero_fracs=False,
        frac=False,
        attn=True,
        compute_device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.special_tok_zero_fracs = special_tok_zero_fracs
        self.numeral_embeddings = numeral_embeddings
        if self.numeral_embeddings:
            print("\n Learned numeral_embeddings!")
        self.compute_device = compute_device
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.numeral = FractionalEncoderNumeral(self.d_model)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.0]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.0]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.0]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(
                self.d_model, nhead=self.heads, dim_feedforward=encoder_ff, dropout=0.1
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.N
            )

    def forward(self, src, frac):
        """Forward pass for the Encoder."""
        x = self.embed(src) * 2**self.emb_scaler

        attention = src.clone()
        attention[attention != 0] = 1

        mask = attention
        src_mask = ~attention.bool()

        if self.numeral_embeddings:
            frac_embed = self.numeral(frac)

        else:
            pe = torch.zeros_like(x)
            ple = torch.zeros_like(x)
            pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
            ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
            pe[:, :, : self.d_model // 2] = self.pe(frac) * pe_scaler
            ple[:, :, self.d_model // 2 :] = self.ple(frac) * ple_scaler
            frac_embed = pe + ple

        if self.special_tok_zero_fracs == "both":
            frac_embed[:, 0, :] = 0
            frac_embed[:, -1, :] = 0

        if self.special_tok_zero_fracs == "cpd":
            frac_embed[:, 0, :] = 0

        if self.special_tok_zero_fracs == "eos":
            frac_embed[:, -1, :] = 0

        if self.attention:
            x_src = x + frac_embed
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = attention.unsqueeze(-1).expand(-1, -1, self.d_model)
        # hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)

        return x


class MTEncoder(nn.Module):
    """Multi-Task Encoder module."""
    def __init__(self, config, compute_device=None):
        super().__init__()

        self.config = config["model"]

        self.decoder = self.config["decoder"]

        valid_options = [
            "mask prediction",
            "multi-task",
            "meanpool",
            "cpd",
            "roost",
            "predict",
        ]
        assert (
            self.decoder in valid_options
        ), f"{self.decoder} is not a valid option. Please select from {valid_options}."

        self.d_model = self.config["d_model"]
        self.N = self.config["N"]
        self.heads = self.config["heads"]
        self.residual_nn_dim = self.config["residual_nn_dim"]
        self.dropout = self.config["dropout"]
        self.branched_ffnn = self.config["branched_ffnn"]
        self.encoder_ff = self.config["encoder_ff"]
        self.special_tok_zero_fracs = self.config["special_tok_zero_fracs"]
        self.numeral_embeddings = self.config["numeral_embeddings"]
        self.task_list = config["task_list"]
        self.task_types = config["task_types"]
        self.task_dim = config["task_dim"]

        if self.special_tok_zero_fracs:
            print("\n special_tok_zero_fracs!")

        self.compute_device = compute_device
        self.encoder = Encoder(
            d_model=self.d_model,
            N=self.N,
            heads=self.heads,
            encoder_ff=self.encoder_ff,
            special_tok_zero_fracs=self.special_tok_zero_fracs,
            compute_device=self.compute_device,
            numeral_embeddings=self.numeral_embeddings,
        )

    def get_number_classes(self):
        """Get the number of classes."""
        pass

    def forward(self, input, embeddings=False):
        """Forward pass for the MTEncoder."""
        # 1 Get an embedding for the compound
        output = self.encoder(input["src_masked"], input["frac"])
        # mask so you only average "elements"
        mask = (input["src_masked"] == 0).unsqueeze(-1).repeat(1, 1, self.d_model)
        output = output.masked_fill(mask, 0)

        if embeddings:
            return output[:, 0, :]

        res = {}
        # 2 decode
        if self.decoder == "roost":
            output = self.output_nn(output)
            mask = (input["src"] == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
            output = output.masked_fill(mask, 0)

            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, : logits.shape[-1]] = torch.sigmoid(logits)
            output_pred = output * probability
            return output_pred

        if self.decoder == "meanpool":
            output = output.mean(dim=1)
            output_pred = self.output_nn(output)
            return output_pred

        if self.decoder == "cpd":
            output = output[:, 0, :]
            output_pred = self.output_nn(output)
            return output_pred

        # todo if masking is used
        # 1. get an encoding for the compound
        # 2. identify the number of masked tokens
        # 3. push each mask encoding through the decoder layers

        if self.decoder == "predict":
            print("decoder")
            output_pred = torch.zeros(output.shape[0], len(self.output_nn), 2).to(
                input["src_masked"].device
            )

            for task_id, task_network in enumerate(self.output_nn):
                # Forward pass for the specific task network
                task_output = task_network(output[:, 0, :])

                # Check the output shape
                if task_output.shape != (output.shape[0], 2):
                    continue  # Skip this iteration and move to the next one
                # Store the output in the corresponding slice of output_pred
                output_pred[:, task_id] = task_output
            return output_pred


if __name__ == "__main__":
    dummy_config = {
        "model": {
            "d_model": 256,
            "N": 3,
            "heads": 4,
            "residual_nn_dim": 256,
            "dropout": 0.1,
            "branched_ffnn": False,
            "encoder_ff": 1024,
            "special_tok_zero_fracs": False,
            "numeral_embeddings": False,
            "decoder": "predict",  # or another valid option
        },
        "task_list": ["task1", "task2"],
        "task_types": ["regression", "regression"],
        "task_dim": [1, 1],
    }
    model = MTEncoder(config=dummy_config)
