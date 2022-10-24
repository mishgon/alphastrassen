from typing import *
import numpy as np

import torch
from torch import nn
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


class Attention(nn.Module):
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass


class SelfAttention(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class AttentiveModels(nn.Module):
    def __init__(self, embed_dim: int):
        self.attn1 = SelfAttention()
        self.attn2 = SelfAttention()
        self.attn3 = SelfAttention()

    def forward(self, grids):
        """NOTE: modify ``grids`` inplace.
        """
        for i, j, self_attn in [(0, 1, self.attn1), (2, 0, self.attn2), (1, 2, self.attn3)]:
            x = torch.cat((grids[i], grids[j]), dim=-2)  # (bs, S, 2S, c)
            x = self_attn(x.flatten(0, 1)).unflatten(0, 1, x.shape[:2])  # (bs * S, 2S, c)
            grids[i], grids[j] = torch.split(x, [grids[i].shape[-2], grids[j].shape[-2]], dim=-2)

        return grids


class Torso(nn.Module):
    def __init__(self, input_size: int, embed_dim: int, num_attn_models: int):
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

        return torch.stack(grids, dim=1).flatten(1, -1)  # (bs, 3 * S * S, c)


class NeuralNet(pl.LightningDataModule):
    def __init__(self, ):
        pass

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def train_step(self, batch, batch_idx=None):
        pass

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        s = torch.tensor(state.tensor, dtype=torch.float32, device=self.device)[None]
        s = s.unsqueeze()  # add batch dim
        pi, v = self.forward(s)
        pi, v = pi.squeeze(), v.squeeze()  # remove batch dims
        return pi.data.cpu().numpy(), v.item()
