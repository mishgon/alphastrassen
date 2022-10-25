from typing import *
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .environment import State


class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[Tuple[State, np.ndarray, float]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, pi, v = self.examples[idx]
        return np.float32(state.tensor), np.float32(pi), np.float32(v)


class DataModule(pl.LightningDataModule):
    def __init__(self, examples: Sequence[Tuple[State, np.ndarray, float]], batch_size: int):
        super().__init__()

        self.dataset = Dataset(examples)
        self.batch_size = batch_size
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: int = 4):
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
    def __init__(self, embed_dim: int = 256):
        super().__init__()

        self.block1 = TransformerBlock(embed_dim)
        self.block2 = TransformerBlock(embed_dim)
        self.block3 = TransformerBlock(embed_dim)

    def forward(self, grids):
        grids = [g.clone() for g in grids]
        for i, j, block in [(0, 1, self.block1), (2, 0, self.block2), (1, 2, self.block3)]:
            x = torch.cat((grids[i], grids[j].transpose(1, 2)), dim=2)  # (bs, S, 2S, c)
            x = block(x.flatten(0, 1)).unflatten(0, x.shape[:2])  # (bs, S, 2S, c)

            S = grids[i].shape[2]
            assert grids[j].shape[1] == S
            assert x.shape[2] == 2 * S

            grids[i] = x[:, :, :S]
            grids[j] = x[:, :, S:].transpose(1, 2)

        return grids


class Torso(nn.Module):
    def __init__(self, input_size: Tuple[int, int, int], embed_dim: int = 256, num_attn_models: int = 4):
        super().__init__()

        self.fc1 = nn.Linear(input_size[-1], embed_dim)
        self.fc2 = nn.Linear(input_size[-2], embed_dim)
        self.fc3 = nn.Linear(input_size[-3], embed_dim)

        self.attn_models = nn.ModuleList([AttentiveModels() for _ in range(num_attn_models)])

        self.input_size = input_size
        self.embed_dim = embed_dim

    def forward(self, s: torch.Tensor):
        assert s.ndim == 4
        assert s.shape[-3:] == self.input_size

        grids = self.fc1(s), self.fc2(s.permute((0, 3, 1, 2))), self.fc3(s.permute((0, 2, 3, 1)))
        for attn_models in self.attn_models:
            grids = attn_models(grids)

        e = torch.stack(grids, dim=1).flatten(1, -2)  # (bs, 3 * S * S, c)
        return torch.max(e, dim=1).values  # (bs, c), global max pool instead of cross attn


class PolicyHead(nn.Module):
    def __init__(self, num_actions: int, embed_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ValueHead(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 256, num_quantiles: int = 8):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_quantiles)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))  # (bs, num_quantiles)


class NeuralNet(pl.LightningModule):
    def __init__(self, input_size: Tuple[int, int, int], num_actions: int,
                 embed_dim: int = 256, u_q: float = 0.75, lr: float = 1e-2):
        super().__init__()

        self.torso = Torso(input_size, embed_dim)
        self.policy_head = PolicyHead(num_actions, embed_dim)
        self.value_head = ValueHead(embed_dim)

        self.save_hyperparameters()

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.torso(s)
        return self.policy_head(e), self.value_head(e)

    def quantile_regression_loss(self, quantiles, target):
        num_quantiles = quantiles.shape[1]
        tau = (torch.arange(num_quantiles).to(quantiles) + 0.5) / num_quantiles

        target = target.unsqueeze(1)
        weights = torch.where(quantiles > target, tau, 1 - tau)
        return torch.mean(weights * F.huber_loss(quantiles, target, reduction='none'))

    def training_step(self, batch, batch_idx=None):
        s, pi, v = batch

        logits, quantiles = self.forward(s)

        policy_loss = F.cross_entropy(logits, pi)
        self.log('policy_loss', policy_loss, on_epoch=True)

        value_loss = self.quantile_regression_loss(quantiles, v)
        self.log('value_loss', value_loss, on_epoch=True)

        return policy_loss + value_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        self.eval()

        s = torch.tensor(state.tensor, dtype=torch.float32, device=self.device)
        s = s.unsqueeze(0)  # add batch dim
        logits, quantiles = self.forward(s)
        logits, quantiles = logits.squeeze(), quantiles.squeeze()  # remove batch dims
        p = torch.softmax(logits, dim=0)
        v = torch.mean(quantiles[int(len(quantiles) * self.hparams.u_q):])
        return p.data.cpu().numpy(), v.item()
