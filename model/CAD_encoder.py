import torch.nn as nn
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstraction


# python -m model.CAD_encoder

class CADEncoder(nn.Module):
    """
    PointNet++ SSG-style encoder.

    Input:
        x: [B, 3, N] or [B, 6, N] if normal_channel=True

    Returns:
        global_feat: [B, embed_dim]
        end_points: dict with intermediate xyz/features
    """
    def __init__(self, embed_dim=256, normal_channel=False, dropout=0.0):
        super().__init__()

        self.normal_channel = normal_channel
        in_channel = 3 if not normal_channel else 6

        # Match PointNet++ cls_ssg
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=in_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        # Replace TF classification head with compact embedding head
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.LayerNorm(512)
        self.dp1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, embed_dim)
        self.bn2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        end_points = {}

        if self.normal_channel:
            l0_xyz = x[:, :3, :]
            l0_points = x[:, 3:, :]
        else:
            l0_xyz = x
            l0_points = None

        end_points["l0_xyz"] = l0_xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        end_points["l1_xyz"] = l1_xyz
        end_points["l1_points"] = l1_points

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        end_points["l2_xyz"] = l2_xyz
        end_points["l2_points"] = l2_points

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        end_points["l3_xyz"] = l3_xyz
        end_points["l3_points"] = l3_points

        # l3_points: [B, 1024, 1] -> [B, 1024]
        global_feat = l3_points.squeeze(-1)

        global_feat = F.relu(self.bn1(self.fc1(global_feat)))
        global_feat = self.dp1(global_feat)
        global_feat = self.bn2(self.fc2(global_feat))

        return global_feat, end_points
    
