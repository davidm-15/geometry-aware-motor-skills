import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256, out_dim=1024, project=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = out_dim if project else hidden_dim

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hidden_dim, out_dim) if project else nn.Identity()

    def forward(self, x, lengths, return_last=False):
        # x: [B, C, T]
        feat = self.feature_extractor(x)  # [B, hidden_dim, T]

        if return_last:
            # Average pool over valid (non-padding) timesteps
            T = feat.size(2)
            mask = (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))  # [B, T]
            pooled = (feat * mask.unsqueeze(1)).sum(dim=2) / lengths.float().unsqueeze(1).clamp(min=1)
            return self.proj(pooled)  # [B, out_dim]

        return self.proj(feat.transpose(1, 2))  # [B, T, out_dim]
