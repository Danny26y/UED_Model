import torch
import torch.nn as nn
from .encoders import MagEncoder, ThermalEncoder, GasEncoder
from .fusion import GFM
from .heads import ClassificationHead, LocalisationHead, ConfidenceHead

class HMLFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mag_enc = MagEncoder()
        self.thermal_enc = ThermalEncoder()
        self.gas_enc = GasEncoder()

        self.fusion = GFM()

        self.cls_head = ClassificationHead()
        self.loc_head = LocalisationHead()
        self.conf_head = ConfidenceHead()

    def forward(self, mag, thermal, gas):
        mag_feat = self.mag_enc(mag)
        thermal_feat = self.thermal_enc(thermal)
        gas_feat = self.gas_enc(gas)

        fused_feat, gate_weights = self.fusion(mag_feat, thermal_feat, gas_feat)

        logits = self.cls_head(fused_feat)
        loc = self.loc_head(fused_feat)
        conf = self.conf_head(fused_feat)

        return logits, loc, conf, gate_weights
