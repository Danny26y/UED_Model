import torch
import torch.nn as nn

class GFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fc1 = nn.Linear(64, 32)
        self.gate_relu = nn.ReLU()
        self.gate_fc2 = nn.Linear(32, 1)

        self.res_fc = nn.Linear(192, 64)
        self.res_relu = nn.ReLU()

    def forward(self, mag_feat, thermal_feat, gas_feat):
        # mag_feat, thermal_feat, gas_feat: (B, 64)
        stacked = torch.stack([mag_feat, thermal_feat, gas_feat], dim=1) # (B, 3, 64)

        # Apply gate MLP independently
        gate_logits = self.gate_fc2(self.gate_relu(self.gate_fc1(stacked))) # (B, 3, 1)
        gate_logits = gate_logits.squeeze(-1) # (B, 3)

        gate_weights = torch.softmax(gate_logits, dim=-1) # (B, 3)

        fused = (gate_weights[:, 0:1] * mag_feat +
                 gate_weights[:, 1:2] * thermal_feat +
                 gate_weights[:, 2:3] * gas_feat) # (B, 64)

        concat_feat = torch.cat([mag_feat, thermal_feat, gas_feat], dim=1) # (B, 192)
        res = self.res_relu(self.res_fc(concat_feat)) # (B, 64)

        fused_feat = fused + res
        return fused_feat, gate_weights
