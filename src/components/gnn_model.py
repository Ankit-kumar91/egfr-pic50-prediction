"""SMILES -> molecular graph, and a from-scratch message-passing GNN (MPNN).

Message passing follows Gilmer et al. (2017): edge features condition the
message, not just node features, which is what separates an MPNN from a
plain GCN/GAT.
"""

import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool

_ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C N O F P S Cl Br I; rest -> "other"
_HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
]
_BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

NODE_DIM = (
    len(_ATOM_LIST) + 1 + len(_HYBRIDIZATIONS) + 1 + 4 + 1 + 1
)  # see _atom_features
EDGE_DIM = len(_BOND_TYPES) + 1 + 1  # bond type one-hot + conjugated + in-ring


def _one_hot(value, choices) -> list[float]:
    return [1.0 if value == c else 0.0 for c in choices]


def _atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), _ATOM_LIST)
        + [1.0 if atom.GetAtomicNum() not in _ATOM_LIST else 0.0]
        + _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
        + [1.0 if atom.GetHybridization() not in _HYBRIDIZATIONS else 0.0]
        + _one_hot(atom.GetDegree(), [0, 1, 2, 3])
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [atom.GetTotalNumHs() / 4.0]
    )


def _bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), _BOND_TYPES)
        + [1.0 if bond.GetIsConjugated() else 0.0]
        + [1.0 if bond.IsInRing() else 0.0]
    )


def mol_to_graph(smiles: str, y: float | None = None) -> Data | None:
    """RDKit SMILES -> PyG Data(x, edge_index, edge_attr[, y]). None if unparsable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = _bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feats, feats]

    if edge_index:
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index_t = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, EDGE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data


def graphs_from_df(df, smiles_col: str = "smiles", target_col: str | None = "pIC50"):
    """DataFrame -> list[Data], dropping rows RDKit can't parse."""
    graphs = []
    for _, row in df.iterrows():
        y = float(row[target_col]) if target_col is not None else None
        g = mol_to_graph(row[smiles_col], y)
        if g is not None:
            graphs.append(g)
    return graphs


class MPNNConv(MessagePassing):
    """One message-passing step: message depends on sender node + edge features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr="mean")
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim), nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class MPNN(nn.Module):
    """Stack of MPNNConv layers -> mean pooling -> FFN regression head.

    Dropout stays active at inference (see model_trainer.predict_gnn_with_uncertainty)
    for MC-dropout uncertainty estimates.
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [MPNNConv(hidden_dim, edge_dim, hidden_dim) for _ in range(depth)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = self.embed(data.x)
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x, data.edge_index, data.edge_attr)))
        pooled = global_mean_pool(x, data.batch)
        return self.head(pooled).squeeze(-1)
