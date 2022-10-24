from typing import *
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .environment import State


class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[State, np.ndarray, float]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, pi, v = self.examples[idx]
        return state.tensor, pi, v


class DataModule(pl.LightningDataModule):
    def __init__(self, examples: Sequence[State, np.ndarray, float], batch_size: int):
        self.dataset = Dataset(examples)
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class AttentiveModels(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.block1 = TransformerBlock(embed_dim)
        self.block2 = TransformerBlock(embed_dim)
        self.block3 = TransformerBlock(embed_dim)

    def forward(self, grids):
        """NOTE: modify ``grids`` inplace.
        """
        for i, j, block in [(0, 1, self.block1), (2, 0, self.block2), (1, 2, self.block3)]:
            x = torch.cat((grids[i], grids[j].transpose(1, 2)), dim=2)  # (bs, S, 2S, c)
            x = block(x.flatten(0, 1)).unflatten(0, 1, x.shape[:2])  # (bs, S, 2S, c)

            S = grids[i].shape[2]
            assert grids[j].shape[1] == S
            assert x.shape[2] == 2 * S

            grids[i] = x[:, :, :S]
            grids[j] = x[:, :, S:].transpose(1, 2)

        return grids


class Torso(nn.Module):
    def __init__(self, input_size: int, embed_dim: int, num_attn_models: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, embed_dim)
        self.fc2 = nn.Linear(input_size, embed_dim)
        self.fc3 = nn.Linear(input_size, embed_dim)
        
        self.attn_models = nn.ModuleList([AttentiveModels() for _ in range(num_attn_models)])
        
        self.input_size = input_size
        self.embed_dim = embed_dim

    def forward(self, s: torch.Tensor):
        assert s.ndim == 4
        assert s.shape[1] == s.shape[2] == s.shape[3] == self.input_size

        grids = self.fc1(s), self.fc2(s.permute((0, 3, 1, 2))), self.fc3(s.permute((0, 2, 3, 1)))
        for attn_models in self.attn_models:
            grids = attn_models(grids)

        e = torch.stack(grids, dim=1).flatten(1, -1)  # (bs, 3 * S * S, c)
        return torch.max(e, dim=1).value  # (bs, c), global max pool instead of cross attn


class PolicyHead(nn.Module):
    def __init__(self, embed_dim: int, num_actions: int, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ValueHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 512, num_quantiles: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_quantiles)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class NeuralNet(pl.LightningDataModule):
    def __init__(self, input_size: int, embed_dim: int, num_actions: int, u_q: float = 0.75):
        self.torso = Torso(input_size, embed_dim)
        self.policy_head = PolicyHead(embed_dim, num_actions)
        self.value_head = ValueHead(embed_dim)
        
        self.u_q = u_q

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.torso(s)
        return self.policy_head(e), self.value_head(e)

    def train_step(self, batch, batch_idx=None):
        s, pi, v = batch
        logits, quantiles = self.forward(s)
        
        policy_loss = F.cross_entropy(logits, pi)

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        s = torch.tensor(state.tensor, dtype=torch.float32, device=self.device)[None]
        s = s.unsqueeze()  # add batch dim
        logits, quantiles = self.forward(s)
        logits, quantiles = logits.squeeze(), quantiles.squeeze()  # remove batch dims
        p = torch.softmax(logits, dim=0)
        v = torch.mean(quantiles[int(len(quantiles) * self.u_q):])
        return p.data.cpu().numpy(), v.item()
