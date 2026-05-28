import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import copy
import wandb
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from model.RNN import TrajectoryEncoder


# Global settings for active rules
ACTIVE_RULES = [
    'vel_scale',
    # 'pos_y',
    'ori_x'
]

# python -m scripts.rule_regression

np.set_printoptions(suppress=True)

class TrajDataset(Dataset):
    def __init__(self, csv_file, feature_stats=None, target_stats=None, padding=False):
        self.data = pd.read_csv(csv_file)


        
        # load the points, times and quaternions
        raw_coords = self.data[['x', 'y', 'z']].values.astype(np.float32)                   # Position of the end effector x,y,z
        raw_time = self.data['time(s)'].values.astype(np.float32).reshape(-1, 1)            # timestamp
        raw_quaternions = self.data[['qx', 'qy', 'qz', 'qw']].values.astype(np.float32)     # Rotation of the end effector qx, qy, qz, qw
        segment_types = self.data['segment_type'].values.astype(str)                        # For point if it is part of straigth or corner omit for now #TODO
        window_id = self.data['window_id'].values.astype(str)                               # To identify different trajectories in the same dataset
        demo_id   = self.data['demonstration_id'].values.astype(str) if 'demonstration_id' in self.data.columns else window_id

        target_cols = []
        if 'vel_scale' in ACTIVE_RULES:
            target_cols.append(self.data['rule_vel_scale'].values.astype(np.float32).reshape(-1, 1))
        if 'pos_y' in ACTIVE_RULES:
            target_cols.append(self.data['rule_pos_y'].values.astype(np.float32).reshape(-1, 1))
        if 'ori_x' in ACTIVE_RULES:
            target_cols.append(self.data['rule_ori_x'].values.astype(np.float32).reshape(-1, 1))

        if 'vel_scale' in ACTIVE_RULES:
            rule_geom = self.data['rule_vel_scale_geom'].values.astype(str)
            rule_geom = np.where(rule_geom == "straight", 0, np.where(rule_geom == "corner", 1, 2))
            target_cols.append(rule_geom.reshape(-1, 1))
        if 'pos_y' in ACTIVE_RULES:
            rule_geom = self.data['rule_pos_y_geom'].values.astype(str)
            rule_geom = np.where(rule_geom == "straight", 0, np.where(rule_geom == "corner", 1, 2))
            target_cols.append(rule_geom.reshape(-1, 1))
        if 'ori_x' in ACTIVE_RULES:
            rule_geom = self.data['rule_ori_x_geom'].values.astype(str)
            rule_geom = np.where(rule_geom == "straight", 0, np.where(rule_geom == "corner", 1, 2))
            target_cols.append(rule_geom.reshape(-1, 1))
            
        targets_raw_vals = np.concatenate(target_cols, axis=1)
        self.num_cont = len(ACTIVE_RULES)

        # Compute deltas of position, time, and rotation. Conver quaternions to rotation matrices
        p_delta = np.diff(raw_coords, axis=0)
        t_delta = np.diff(raw_time, axis=0)
        R_matrices = R.from_quat(raw_quaternions).as_matrix()
        R_delta = R_matrices[1:] @ np.transpose(R_matrices[:-1], (0, 2, 1))
        R_delta  = R_delta[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)

        safe_t = np.where(t_delta == 0, 1e-6, t_delta)
        self.delta_time = t_delta
        features_p_raw = p_delta / safe_t
        features_R_raw = R_matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)[:-1]

        self.samples = []
        seq = []

        raw_samples = []
        for i in range(len(features_p_raw)-1):
            if ((window_id[i] == window_id[i + 1]) and (demo_id[i] == demo_id[i + 1]) and (raw_time[i] < raw_time[i+1])) or (i > len(features_p_raw)-2):
                seq.append(np.concatenate([self.delta_time[i], features_p_raw[i], features_R_raw[i]]))
            else:
                target = targets_raw_vals[i, :]
                raw_samples.append((copy.deepcopy(seq), target))
                seq = []
        



        if feature_stats is None:
            self.feat_mean = np.mean(features_p_raw, axis=0)
            self.feat_std = np.std(features_p_raw, axis=0) + 1e-6

            self.time_mean = np.mean(self.delta_time)
            self.time_std = np.std(self.delta_time) + 1e-6

            self.feature_stats = (self.feat_mean, self.feat_std, self.time_mean, self.time_std)
        else:
            self.feat_mean, self.feat_std = feature_stats[0], feature_stats[1]
            self.time_mean, self.time_std = feature_stats[2], feature_stats[3]
        
        target_p = np.array([s[1][0:self.num_cont] for s in raw_samples])
        if target_stats is None:
            self.target_mean = np.mean(target_p, axis=0)
            self.target_std = np.std(target_p, axis=0) + 1e-6
        else:
            self.target_mean, self.target_std = target_stats
        
        
        self.samples = []
        longest_len = np.max([len(s[0]) for s in raw_samples]) if raw_samples else 0 

        for i in range(len(raw_samples)):
            seq, target = raw_samples[i]
            actual_len = len(seq)
            seq = np.array(seq)
            speed = np.linalg.norm(seq[:, 1:4], axis=1)
            p90_speed = np.percentile(speed, 90) if len(speed) > 0 else 0.0

            seq[:, 1:4] = (seq[:, 1:4] - self.feat_mean) / self.feat_std
            seq[:, 0] = (seq[:, 0] - self.time_mean) / self.time_std
            target[0:self.num_cont] = (target[0:self.num_cont] - self.target_mean) / self.target_std
            pad_len = longest_len - len(seq)
            if pad_len > 0:
                seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            self.samples.append((seq, target, actual_len, p90_speed))


        print(f"Loaded {csv_file}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target, actual_len, p90_speed = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32).transpose(0, 1), torch.tensor(target, dtype=torch.float32), actual_len, torch.tensor(p90_speed, dtype=torch.float32)

    def _get_name__(self):  
        return "TrajDataset"



class SpeedRNN(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        
        # 1. Shared Feature Extractors (Same as yours)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.rnn = nn.GRU(
            input_size=32, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.num_cont_rules = len(ACTIVE_RULES)
        self.continuous_head = nn.Linear(16, max(1, self.num_cont_rules))
        
        # Classification heads for the geometry types (straight, corner, none) for each active rule
        self.discrete_heads = nn.ModuleList([nn.Linear(16, 3) for _ in range(self.num_cont_rules)])

    def forward(self, x, lengths):
        # Extract features
        x = self.feature_extractor(x) 
        x = x.transpose(1, 2)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, h_n = self.rnn(packed_x)   
        last_state = h_n[0] 
        
        # Pass through shared dense layer
        shared_features = self.shared_fc(last_state)
        
        # Get predictions from all heads
        cont_preds = self.continuous_head(shared_features)
        disc_preds = [head(shared_features) for head in self.discrete_heads]
        
        # Return all predictions
        return cont_preds, disc_preds
    
    def _get_name__(self):
        return "SpeedRNN"
    
# ---------------------------------------------------------
# 1. Trajectory Encoder (Modified SpeedRNN)
# ---------------------------------------------------------
class TrajectoryEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256, out_dim=1024):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.rnn = nn.GRU(
            input_size=32, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        # Project the RNN hidden dimension up to the embedding space (1024)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, lengths):
        x = self.feature_extractor(x).transpose(1, 2)
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # We take the full output sequence instead of just the last state
        packed_out, _ = self.rnn(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # out shape: (Batch, Sequence_Length, Hidden_Dim)
        return self.proj(out) 


# ---------------------------------------------------------
# 2. CAD Encoder (Simplified PointNet)
# ---------------------------------------------------------
class CADEncoder(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        # A very basic PointNet feature extractor
        # Input shape: (Batch, 3, Num_Points)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 256, 1)
        self.conv3 = nn.Conv1d(256, out_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, pc):
        # Extract per-point features
        x = self.relu(self.conv1(pc))
        x = self.relu(self.conv2(x))
        x = self.conv3(x) 
        
        # x shape: (Batch, out_dim, Num_Points)
        # Transpose so it matches attention format: (Batch, Num_Points, out_dim)
        return x.transpose(1, 2)

# ---------------------------------------------------------
# 3. Master Fusion Model
# ---------------------------------------------------------
class TrajectoryCADFusionModel(nn.Module):
    def __init__(self, num_cont_rules, embed_dim=1024, num_heads=8):
        super().__init__()
        
        self.traj_encoder = TrajectoryEncoder(out_dim=embed_dim)
        self.cad_encoder = CADEncoder(out_dim=embed_dim)
        
        # Cross Attention: Trajectory attends to CAD points
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Self Attention over the fused trajectory to aggregate time steps
        self.self_attention = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        
        # Final shared dense layers
        self.shared_fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Heads (Matching your original design)
        self.num_cont_rules = num_cont_rules
        self.continuous_head = nn.Linear(64, max(1, self.num_cont_rules))
        self.discrete_heads = nn.ModuleList([nn.Linear(64, 3) for _ in range(self.num_cont_rules)])

    def forward(self, traj, lengths, cad_points):
        # 1. Encode both modalities
        # traj_feat: (B, Seq_Len, 1024)
        traj_feat = self.traj_encoder(traj, lengths) 
        
        # cad_feat: (B, Num_Points, 1024)
        cad_feat = self.cad_encoder(cad_points)      
        
        # 2. Cross-Attention
        # Query = Trajectory, Key = CAD, Value = CAD
        # This gives us context-aware trajectory features based on geometry
        fused_feat, _ = self.cross_attention(query=traj_feat, key=cad_feat, value=cad_feat)
        
        # 3. Self-Attention (Optional, but helps the sequence figure out its own temporal dependencies after looking at the CAD)
        fused_feat = self.self_attention(fused_feat)
        
        # 4. Pooling (Collapse the sequence down to a single vector for classification)
        # Here we just take the max-pool over the sequence length. 
        # Alternatively, you could extract the specific valid 'last' state using `lengths`
        global_feat, _ = torch.max(fused_feat, dim=1) 
        
        # 5. Prediction Heads
        shared_features = self.shared_fc(global_feat)
        
        cont_preds = self.continuous_head(shared_features)
        disc_preds = [head(shared_features) for head in self.discrete_heads]
        
        return cont_preds, disc_preds



learning_rate = 1e-3
weight_decay = 1e-2
train_batch_size = 128
val_batch_size = 64
test_batch_size = 1
datset_padding = True
dataset_path = "datasets/L_shape"
input_shape = 10
epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    train_ds = TrajDataset(f"{dataset_path}/train.csv", padding=datset_padding)
    val_ds = TrajDataset(f"{dataset_path}/val.csv", feature_stats=(train_ds.feat_mean, train_ds.feat_std, train_ds.time_mean, train_ds.time_std), target_stats=(train_ds.target_mean, train_ds.target_std), padding=datset_padding)
    test_ds = TrajDataset(f"{dataset_path}/test.csv", feature_stats=(train_ds.feat_mean, train_ds.feat_std, train_ds.time_mean, train_ds.time_std), target_stats=(train_ds.target_mean, train_ds.target_std), padding=datset_padding)
    
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size)

    model = SpeedRNN(in_channels=input_shape).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=1.0)
    classification_criterion = nn.CrossEntropyLoss()

    best_mae = float('inf')
    no_progress_epochs = 0
    max_no_progress = 20

    # targets = [s[1][0] for s in train_ds.samples] # Extract just the speed target
    # plt.hist(targets, bins=50)
    # print("Histogram values:", np.histogram(targets, bins=50)[0])
    # plt.xlabel("Speed (normalized)")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of Training Targets")
    # plt.show()
    # plt.show()

    wandb.init(
        project="traj-speed-prediction",
        config={
            "architecture": model._get_name(),
            "learning_rate": learning_rate,
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "test_batch_size": test_batch_size,
            "datset_padding": datset_padding,
            "weight_decay": weight_decay,
            "criterion": criterion._get_name(),
            "dataset": dataset_path,
            "input_shape": input_shape
        }
    )
    
    t_mean = train_ds.target_mean
    t_std = train_ds.target_std

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y, lengths, _ in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad()
            cont_preds, disc_preds = model(x, lengths)
            
            num_rules = len(ACTIVE_RULES)
            cont_targets = y[:, 0:num_rules]
            
            loss_cont = criterion(cont_preds, cont_targets)
            loss = loss_cont
            
            for i in range(num_rules):
                disc_targets = y[:, num_rules + i].long()
                loss += classification_criterion(disc_preds[i], disc_targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()

        val_mae = 0
        with torch.no_grad():
            for x, y, lengths, _ in val_loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                cont_preds, disc_preds = model(x, lengths)
                num_rules = len(ACTIVE_RULES)

                actual_val = torch.from_numpy(y.cpu().numpy()).to(device)
                actual_val[:, 0:num_rules] = actual_val[:, 0:num_rules] * torch.tensor(t_std).to(device) + torch.tensor(t_mean).to(device)
                
                pred_val = cont_preds.cpu()
                pred_val[:, 0:num_rules] = pred_val[:, 0:num_rules] * torch.tensor(t_std) + torch.tensor(t_mean)
                pred_val = pred_val.to(device)
                
                error = torch.abs(actual_val[:, 0:num_rules] - pred_val[:, 0:num_rules])
                val_mae += torch.mean(error)
        
        wandb.log({"epoch": epoch+1, "train_loss": total_loss/len(train_loader), "val_mae": val_mae/len(val_loader)})
        
        val_mae /= len(val_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val MAE: {val_mae:.4f}")
            
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_speed_model.pth")
            no_progress_epochs = 0
        else:
            no_progress_epochs += 1
            # if no_progress_epochs >= max_no_progress:
            #     print(f"No improvement for {max_no_progress} epochs, stopping training.")
                # break

    # Evaluation
    model.load_state_dict(torch.load("best_speed_model.pth"))
    model.eval()
    
    print("\n--- Running Comprehensive Test Evaluation ---")
    
    all_actual_cont = {rule: [] for rule in ACTIVE_RULES}
    all_pred_cont = {rule: [] for rule in ACTIVE_RULES}
    all_actual_bin = {rule: [] for rule in ACTIVE_RULES}
    all_pred_bin = {rule: [] for rule in ACTIVE_RULES}
    all_p90 = {rule: [] for rule in ACTIVE_RULES}

    test_indices = np.random.choice(len(test_ds), size=10, replace=False)
    with torch.no_grad():
        for i, (x, y, lengths, p90) in enumerate(test_loader):
            cont_pred, disc_preds = model(x.to(device), lengths.to(device))
            
            # Move to CPU and numpy
            actual_val = y.cpu().numpy()
            pred_cont_val = cont_pred.cpu().numpy()
            pred_disc_vals = [p.cpu().numpy() for p in disc_preds]

            num_rules = len(ACTIVE_RULES)
            # Denormalize ONLY the continuous variables
            actual_val[:, 0:num_rules] = actual_val[:, 0:num_rules] * t_std + t_mean
            pred_cont_val[:, 0:num_rules] = pred_cont_val[:, 0:num_rules] * t_std + t_mean
            
            error = abs(actual_val[:, 0:num_rules] - pred_cont_val[:, 0:num_rules])

            for j, rule in enumerate(ACTIVE_RULES):
                all_actual_cont[rule].extend(actual_val[:, j])
                all_pred_cont[rule].extend(pred_cont_val[:, j])
                all_actual_bin[rule].extend(actual_val[:, num_rules + j])
                all_pred_bin[rule].extend(np.argmax(pred_disc_vals[j], axis=1))
                all_p90[rule].extend(p90.cpu().numpy())

            if i in test_indices:
                print(f"Sample {i}:")
                for j, rule in enumerate(ACTIVE_RULES):
                    print(f"  {rule} cont: Actual {float(actual_val[0][j]):.4f} | Pred {float(pred_cont_val[0][j]):.4f} | Err {float(error[0][j]):.4f}")
                    print(f"  {rule} disc: Actual {int(actual_val[0][num_rules+j])} | Pred {int(np.argmax(pred_disc_vals[j], axis=1)[0])}")

    # --- PART 3: Plot Generation Functions ---
    os.makedirs("outputs/plots", exist_ok=True)

    def generate_eval_plots(df_res, title_suffix, prefix, rule_name):
        sns.set_theme(style="whitegrid")
        
        # Plot 1: Box plot of MAE over target ranges
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=df_res, x='Value_Bin', y='Absolute_Error', ax=ax, color='lightblue')
        ax.set_title(f'Abs Error Dist per Range ({title_suffix})')
        ax.set_ylabel('Absolute Error')
        ax.set_xlabel('Actual Target Range')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        filename1 = f"outputs/plots/L_shape_{rule_name}_{prefix}_MAE_Boxplot.png"
        plt.savefig(filename1)
        wandb.log({f"Eval - Boxplot - {title_suffix}": wandb.Image(filename1)})
        plt.close(fig)

        # Plot 2: Continuous plot MAE with Variance lines
        fig, ax = plt.subplots(figsize=(6, 5))
        grouped = df_res.groupby('Value_Bin', observed=False)['Absolute_Error']
        means = grouped.mean()
        variances = grouped.var().fillna(0)
        
        x_vals = np.arange(len(means))
        ax.plot(x_vals, means.values, label='MAE', color='blue')
        ax.plot(x_vals, means.values + variances.values, label='MAE + Var', color='red', linestyle='--')
        lower_bound = np.clip(means.values - variances.values, a_min=0, a_max=None)
        ax.plot(x_vals, lower_bound, label='MAE - Var', color='red', linestyle='--')

        ax.set_xticks(x_vals)
        ax.set_xticklabels(means.index, rotation=45)
        ax.set_title(f'MAE Curve with Variance ({title_suffix})')
        ax.set_ylabel('Absolute Error')
        ax.set_xlabel('Actual Target Range')
        ax.legend()
        plt.tight_layout()
        filename2 = f"outputs/plots/L_shape_{rule_name}_{prefix}_MAE_Variance.png"
        plt.savefig(filename2)
        wandb.log({f"Eval - MAE Curve - {title_suffix}": wandb.Image(filename2)})
        plt.close(fig)

        # Plot 3: Classification Accuracy Curve per target range
        fig, ax = plt.subplots(figsize=(6, 5))
        acc_per_bin = df_res.groupby('Value_Bin', observed=False)['Correct_Class'].mean().reset_index()
        sns.lineplot(data=acc_per_bin, x='Value_Bin', y='Correct_Class', ax=ax, marker='o')
        ax.set_title(f'Classification Accuracy ({title_suffix})')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Actual Target Range')
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        filename3 = f"outputs/plots/L_shape_{rule_name}_{prefix}_Classification_Accuracy.png"
        plt.savefig(filename3)
        wandb.log({f"Eval - Accuracy - {title_suffix}": wandb.Image(filename3)})
        plt.close(fig)

    def generate_scatter_plots(df_res_dict, rule_name):
        titles = ["All", "Straight", "Corner", "None"]
        keys = ["All", "Straight", "Corner", "None"]
        for i, key in enumerate(keys):
            if key in df_res_dict and not df_res_dict[key].empty:
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.scatterplot(data=df_res_dict[key], x='Actual_Value', y='Predicted_Value', ax=ax, alpha=0.5)
                # Plot diagonal line
                min_val = min(df_res_dict[key]['Actual_Value'].min(), df_res_dict[key]['Predicted_Value'].min())
                max_val = max(df_res_dict[key]['Actual_Value'].max(), df_res_dict[key]['Predicted_Value'].max())
                if pd.notna(min_val) and pd.notna(max_val):
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                ax.set_title(f'Predicted vs Actual ({titles[i]}) - {rule_name}')
                ax.set_xlabel('Actual Value')
                ax.set_ylabel('Predicted Value')
                plt.tight_layout()
                filename = f"outputs/plots/L_shape_{rule_name}_{key}_Scatter_Pred_vs_Actual.png"
                plt.savefig(filename)
                wandb.log({f"Scatter Pred vs Actual - {key} - {rule_name}": wandb.Image(filename)})
                plt.close(fig)

    def generate_p90_scatter(df_res, rule_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_res, x='P90_Speed', y='Predicted_Value', ax=ax, alpha=0.5)
        ax.set_title(f'Predicted Scale vs 90th Pct Original Velocity ({rule_name})')
        ax.set_xlabel('90th Percentile Velocity')
        ax.set_ylabel('Predicted Value')
        plt.tight_layout()
        filename = f"outputs/plots/L_shape_{rule_name}_P90_Scatter.png"
        plt.savefig(filename)
        wandb.log({f"Scatter Predicted vs P90 Speed - {rule_name}": wandb.Image(filename)})
        plt.close(fig)

    def generate_2x2_scatter(l_straight, l_corner, w_straight, w_corner):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        panels = [
            (axes[0, 0], l_straight, "L-shape – Straight"),
            (axes[0, 1], l_corner,   "L-shape – Corner"),
            (axes[1, 0], w_straight, "Window – Straight"),
            (axes[1, 1], w_corner,   "Window – Corner"),
        ]
        lims = [0.0, 2.5]
        for ax, df, title in panels:
            if df is None or df.empty:
                ax.set_visible(False)
                continue
            ax.scatter(df['Actual_Value'], df['Predicted_Value'], alpha=0.4, s=12)
            ax.plot(lims, lims, 'r--', linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(title)
            ax.set_xlabel('Actual Velocity Scale')
            ax.set_ylabel('Predicted Velocity Scale')
        plt.tight_layout()
        filename = "outputs/plots/2x2_scatter_vel_scale.png"
        plt.savefig(filename, dpi=200)
        wandb.log({"2x2 Scatter - vel_scale": wandb.Image(filename)})
        plt.close(fig)

    l_df_straight = None
    l_df_corner = None

    all_actual_cont_combined = []
    all_pred_cont_combined = []
    all_actual_bin_combined = []
    all_pred_bin_combined = []
    all_p90_combined = []

    # Evaluate all active rules
    for eval_rule in ACTIVE_RULES:
        # Convert lists to numpy arrays
        actual_cont = np.array(all_actual_cont[eval_rule])
        pred_cont = np.array(all_pred_cont[eval_rule])
        actual_bin = np.array(all_actual_bin[eval_rule])
        pred_bin_class = np.array(all_pred_bin[eval_rule]) # Already class indices from argmax
        p90_speeds = np.array(all_p90[eval_rule])

        all_actual_cont_combined.extend(actual_cont)
        all_pred_cont_combined.extend(pred_cont)
        all_actual_bin_combined.extend(actual_bin)
        all_pred_bin_combined.extend(pred_bin_class)
        all_p90_combined.extend(p90_speeds)

        # Calculate absolute errors
        abs_errors = np.abs(actual_cont - pred_cont)

        # --- PART 1: Classification Metrics ---
        print(f"\n[ Classification Metrics ({eval_rule}) ]")
        print(f"Accuracy:  {accuracy_score(actual_bin, pred_bin_class):.4f}")
        print(f"Precision: {precision_score(actual_bin, pred_bin_class, average='weighted', zero_division=0):.4f}")
        print(f"Recall:    {recall_score(actual_bin, pred_bin_class, average='weighted', zero_division=0):.4f}")
        print(f"F1 Score:  {f1_score(actual_bin, pred_bin_class, average='weighted', zero_division=0):.4f}")

        # --- PART 2: Binning Data for Plots ---
        # Adjust bins dynamically based on rule value ranges to ensure meaningful plots
        min_val = min(actual_cont.min(), pred_cont.min())
        max_val = max(actual_cont.max(), pred_cont.max())
        if max_val - min_val == 0:
            bin_width = 0.1
        else:
            bin_width = max((max_val - min_val) / 10, 0.1)
        bins = np.arange(min_val - bin_width, max_val + bin_width * 2, bin_width)
        labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        
        df_results = pd.DataFrame({
            'Actual_Value': actual_cont,
            'Predicted_Value': pred_cont,
            'Absolute_Error': abs_errors,
            'Correct_Class': (actual_bin == pred_bin_class).astype(float),
            'Actual_Class': actual_bin,
            'P90_Speed': p90_speeds
        })
        
        df_results['Value_Bin'] = pd.cut(df_results['Actual_Value'], bins=bins, labels=labels, include_lowest=True, duplicates='drop')

        # --- PART 4: Execution ---
        df_straight = df_results[df_results['Actual_Class'] == 0]
        df_corner = df_results[df_results['Actual_Class'] == 1]
        df_none = df_results[df_results['Actual_Class'] == 2]

        df_dict = {
            "All": df_results,
            "Straight": df_straight,
            "Corner": df_corner,
            "None": df_none
        }

        if not df_results.empty:
            generate_eval_plots(df_results, f"All - {eval_rule}", "All", eval_rule)
        if not df_straight.empty:
            generate_eval_plots(df_straight, f"Straight - {eval_rule}", "Straight", eval_rule)
        if not df_corner.empty:
            generate_eval_plots(df_corner, f"Corner - {eval_rule}", "Corner", eval_rule)

        generate_scatter_plots(df_dict, eval_rule)
        generate_p90_scatter(df_results, eval_rule)

        if eval_rule == 'vel_scale':
            l_df_straight = df_straight.copy()
            l_df_corner = df_corner.copy()

    # --- PART 5: COMBINED EVALUATION ---
    if len(ACTIVE_RULES) > 1:
        print(f"\n[ Classification Metrics (All Rules Combined) ]")
        act_comb = np.array(all_actual_bin_combined)
        pred_comb = np.array(all_pred_bin_combined)
        print(f"Accuracy:  {accuracy_score(act_comb, pred_comb):.4f}")
        print(f"Precision: {precision_score(act_comb, pred_comb, average='weighted', zero_division=0):.4f}")
        print(f"Recall:    {recall_score(act_comb, pred_comb, average='weighted', zero_division=0):.4f}")
        print(f"F1 Score:  {f1_score(act_comb, pred_comb, average='weighted', zero_division=0):.4f}")

        # Generate a combined scatter plot
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x=all_actual_cont_combined, y=all_pred_cont_combined, hue=["Combined"] * len(all_actual_cont_combined), ax=ax, alpha=0.5, legend=False)
        min_val = min(min(all_actual_cont_combined), min(all_pred_cont_combined))
        max_val = max(max(all_actual_cont_combined), max(all_pred_cont_combined))
        if pd.notna(min_val) and pd.notna(max_val):
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title('Predicted vs Actual (All Rules Combined)')
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Predicted Value')
        plt.tight_layout()
        filename = "outputs/plots/L_shape_Combined_Scatter_Pred_vs_Actual.png"
        plt.savefig(filename)
        wandb.log({"Scatter Pred vs Actual - Combined": wandb.Image(filename)})
        plt.close(fig)
    
    # --- PART 6: 2x2 SCATTER (L-shape + Window) ---
    if 'vel_scale' in ACTIVE_RULES and l_df_straight is not None:
        window_train_ds = TrajDataset("datasets/windows-v2/train.csv", padding=datset_padding)
        window_test_ds = TrajDataset(
            "datasets/windows-v2/test.csv",
            feature_stats=(window_train_ds.feat_mean, window_train_ds.feat_std,
                           window_train_ds.time_mean, window_train_ds.time_std),
            target_stats=(window_train_ds.target_mean, window_train_ds.target_std),
            padding=datset_padding,
        )
        window_test_loader = DataLoader(window_test_ds, batch_size=test_batch_size)
        w_t_mean = window_train_ds.target_mean
        w_t_std = window_train_ds.target_std

        window_model = SpeedRNN(in_channels=input_shape).to(device)
        window_model.load_state_dict(torch.load("best_speed_model_window.pth"))
        window_model.eval()

        w_actual_cont, w_pred_cont, w_actual_bin = [], [], []
        with torch.no_grad():
            for x, y, lengths, p90 in window_test_loader:
                cont_pred, disc_preds = window_model(x.to(device), lengths.to(device))
                actual_val = y.cpu().numpy()
                pred_cont_val = cont_pred.cpu().numpy()
                num_rules = len(ACTIVE_RULES)
                actual_val[:, 0:num_rules] = actual_val[:, 0:num_rules] * w_t_std + w_t_mean
                pred_cont_val[:, 0:num_rules] = pred_cont_val[:, 0:num_rules] * w_t_std + w_t_mean
                vel_idx = ACTIVE_RULES.index('vel_scale')
                w_actual_cont.extend(actual_val[:, vel_idx])
                w_pred_cont.extend(pred_cont_val[:, vel_idx])
                w_actual_bin.extend(actual_val[:, num_rules + vel_idx])

        df_w = pd.DataFrame({
            'Actual_Value': np.array(w_actual_cont),
            'Predicted_Value': np.array(w_pred_cont),
            'Actual_Class': np.array(w_actual_bin, dtype=int),
        })
        w_df_straight = df_w[df_w['Actual_Class'] == 0]
        w_df_corner   = df_w[df_w['Actual_Class'] == 1]

        generate_2x2_scatter(l_df_straight, l_df_corner, w_df_straight, w_df_corner)

    # Optional blocking show
    # plt.show()

    wandb.finish()

    

def test_inference():
    train_ds = TrajDataset(f"{dataset_path}/train.csv", padding=datset_padding)
    model = SpeedRNN(in_channels=input_shape).to(device)
    model.load_state_dict(torch.load("best_speed_model.pth"))

    test_ds = TrajDataset(f"datasets/real_experiment/combined.csv", feature_stats=(train_ds.feat_mean, train_ds.feat_std, train_ds.time_mean, train_ds.time_std), target_stats=(train_ds.target_mean, train_ds.target_std), padding=datset_padding)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size)

    t_mean = train_ds.target_mean
    t_std = train_ds.target_std

    with torch.no_grad():
        for i, (x, y, lengths, p90) in enumerate(test_loader):
            cont_pred, disc_preds = model(x.to(device), lengths.to(device))
            
            # Move to CPU and numpy
            actual_val = y.cpu().numpy()
            pred_cont_val = cont_pred.cpu().numpy()
            pred_disc_vals = [p.cpu().numpy() for p in disc_preds]

            num_rules = len(ACTIVE_RULES)
            # Denormalize ONLY the continuous variables
            actual_val[:, 0:num_rules] = actual_val[:, 0:num_rules] * t_std + t_mean
            pred_cont_val[:, 0:num_rules] = pred_cont_val[:, 0:num_rules] * t_std + t_mean
            
            error = abs(actual_val[:, 0:num_rules] - pred_cont_val[:, 0:num_rules])


            print(f"Sample {i}:")
            for j, rule in enumerate(ACTIVE_RULES):
                print(f"  {rule} cont: Actual {float(actual_val[0][j]):.4f} | Pred {float(pred_cont_val[0][j]):.4f} | Err {float(error[0][j]):.4f}")
                print(f"  {rule} disc: Actual {int(actual_val[0][num_rules+j])} | Pred {int(np.argmax(pred_disc_vals[j], axis=1)[0])}")


if __name__ == "__main__":
    train()
    # test_inference()
