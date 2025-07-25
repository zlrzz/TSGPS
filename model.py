import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()

        self.FC = nn.Sequential(nn.Linear(input_size, 512), nn.LayerNorm(512), nn.GELU(),
                                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
                                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
                                nn.Linear(128, 64),
                                nn.Linear(64, 32),
                                nn.Linear(32, 16),
                                nn.Linear(16, num_classes),
                                )

    def forward(self, x):
        out = self.FC(x)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerModel, self).__init__()
        self.attention = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=input_size, nhead=5, dropout=0.1, activation='gelu',
                                           layer_norm_eps=1e-5, batch_first=True, norm_first=True), num_layers=4,
                norm=nn.LayerNorm(input_size)),
        )

        self.FC = nn.Sequential(nn.Softmax(dim=1),
                                nn.Linear(input_size, 512), nn.LayerNorm(512), nn.GELU(),
                                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
                                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
                                nn.Linear(128, 64),
                                nn.Linear(64, 32),
                                nn.Linear(32, 16),
                                nn.Linear(16, num_classes),
                                )

    def forward(self, out):
        # out = out.unsqueeze(1)
        out = self.attention(out)
        out1 = self.attention(out)
        # out = out.squeeze(1)
        out = self.FC(out)
        return out


class TmpModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TmpModel, self).__init__()
        self.attention = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=input_size, nhead=5, dropout=0.1, activation='gelu',
                                           layer_norm_eps=1e-5, batch_first=True, norm_first=True), num_layers=4, norm=nn.LayerNorm(input_size)),
            nn.Linear(input_size, 1024),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=1024, nhead=2, dropout=0.1, activation='relu',
                                           layer_norm_eps=1e-5, batch_first=True, norm_first=True), num_layers=2, norm=nn.LayerNorm(1024)),

        )

        self.FC = nn.Sequential(nn.Softmax(dim=1),
                                nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
                                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
                                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
                                nn.Linear(128, 64),
                                nn.Linear(64, 32),
                                nn.Linear(32, 16),
                                nn.Linear(16, num_classes),
                                )

    def forward(self, out):
        # out = out.unsqueeze(1)
        out = self.attention(out)
        # out = out.squeeze(1)
        out = self.FC(out)
        return out
