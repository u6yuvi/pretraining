"""Projection head for mapping student features to teacher dimension."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ProjectionHead(nn.Module):
    """
    MLP projection head for feature dimension mapping.
    
    Maps student features from (B, C_in, H, W) to (B, C_out, H, W).
    
    Args:
        in_dim: Input dimension (student feature channels).
        out_dim: Output dimension (teacher embedding dimension).
        n_layers: Number of MLP layers.
        hidden_dim: Hidden dimension for intermediate layers.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int = 1,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        n_layers = max(n_layers, 1)
        
        if n_layers == 1:
            self.mlp: nn.Module = nn.Linear(in_dim, out_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
            
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Project student features.
        
        Args:
            x: (B, C_in, H, W)
            
        Returns:
            (B, C_out, H, W)
        """
        # Channel-last: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        # Channel-first: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x
