import torch
import torch.nn as nn

from .RNN import TrajectoryEncoder
from .CAD_encoder import CADEncoder

ACTIVE_RULES = ["vel_scale", "ori_x", "proximity"]


class FusionModel(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        cad_embed_dim=256,
        num_rules=len(ACTIVE_RULES),
        num_geom_classes=3,
    ):
        super().__init__()
        self.num_rules = num_rules

        self.traj_encoder = TrajectoryEncoder(
            in_channels=10,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            project=False,
        )

        self.cad_encoder = CADEncoder(embed_dim=cad_embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + cad_embed_dim, hidden_dim),
            nn.ReLU(),
        )

        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.continuous_head = nn.Linear(16, num_rules)
        self.discrete_heads = nn.ModuleList(
            [nn.Linear(16, num_geom_classes) for _ in range(num_rules)]
        )

    def forward(self, traj, lengths, cad):
        """
        traj:    [B, C, T]  – trajectory features (C=10)
        lengths: [B]        – actual sequence lengths (before padding)
        cad:     [B, 3, N]  – CAD point cloud

        Returns:
            cont_preds:  [B, num_rules]
            geom_logits: list of num_rules tensors, each [B, num_geom_classes]
        """
        traj_feat = self.traj_encoder(traj, lengths, return_last=True)  # [B, hidden_dim]
        cad_feat, _ = self.cad_encoder(cad)                             # [B, cad_embed_dim]

        fused = self.fusion(torch.cat([traj_feat, cad_feat], dim=-1))   # [B, hidden_dim]
        shared = self.shared_fc(fused)                                   # [B, 16]

        cont_preds = self.continuous_head(shared)                        # [B, num_rules]
        geom_logits = [head(shared) for head in self.discrete_heads]     # list of [B, num_geom_classes]

        return cont_preds, geom_logits
