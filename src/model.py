import torch
import torch.nn as nn
import torchvision.models.video as video_models

class TwoStreamCNNLSTM(nn.Module):
    def __init__(self, num_classes=126, hidden_size=512, num_layers=2):
        super(TwoStreamCNNLSTM, self).__init__()
        
        # RGB Stream: ResNet3D-18
        self.rgb_cnn = video_models.r3d_18(weights='KINETICS400_V1')
        self.rgb_cnn.fc = nn.Identity()  # Remove final FC layer
        
        # RDM Stream: Shared ResNet3D-18 for RDM1, RDM2, RDM3
        self.rdm_cnn = video_models.r3d_18(weights='KINETICS400_V1')
        self.rdm_cnn.fc = nn.Identity()
        
        # LSTM for temporal modeling
        self.rgb_lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.rdm_lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fusion and classification
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, rgb, rdm):
        # rgb: [B, T, C, H, W]
        # rdm: [B, 3, T, C, H, W]
        batch_size = rgb.size(0)
        
        # RGB Stream
        rgb = rgb.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] for 3D CNN
        rgb_features = self.rgb_cnn(rgb)  # [B, 512]
        rgb_features = rgb_features.unsqueeze(1)  # [B, 1, 512]
        rgb_out, _ = self.rgb_lstm(rgb_features)  # [B, 1, hidden_size]
        rgb_out = rgb_out[:, -1, :]  # [B, hidden_size]
        
        # RDM Stream
        rdm_features = []
        for i in range(3):  # Process RDM1, RDM2, RDM3
            rdm_i = rdm[:, i].permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            rdm_feat = self.rdm_cnn(rdm_i)  # [B, 512]
            rdm_features.append(rdm_feat)
        rdm_features = torch.stack(rdm_features, dim=1)  # [B, 3, 512]
        rdm_features = rdm_features.mean(dim=1)  # [B, 512]
        rdm_features = rdm_features.unsqueeze(1)  # [B, 1, 512]
        rdm_out, _ = self.rdm_lstm(rdm_features)  # [B, 1, hidden_size]
        rdm_out = rdm_out[:, -1, :]  # [B, hidden_size]
        
        # Fusion
        fused = torch.cat([rgb_out, rdm_out], dim=1)  # [B, hidden_size*2]
        fused = self.relu(self.fusion(fused))  # [B, hidden_size]
        fused = self.dropout(fused)
        logits = self.fc(fused)  # [B, num_classes]
        
        return logits
    

class AdvancedTwoStreamModel(nn.Module):
    def __init__(self, num_classes=126, hidden_size=512, num_layers=2):
        super(AdvancedTwoStreamModel, self).__init__()
        
        # Use a better video backbone
        self.rgb_cnn = video_models.mc3_18(weights='KINETICS400_V1')
        self.rgb_cnn.fc = nn.Identity()

        self.rdm_cnn = video_models.mc3_18(weights='KINETICS400_V1')
        self.rdm_cnn.fc = nn.Identity()
        
        # Temporal Transformer instead of LSTM
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024)
        self.rgb_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.rdm_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention fusion
        self.fusion_attention = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, rgb, rdm):
        batch_size = rgb.size(0)
        
        # RGB: [B, T, C, H, W] → [B, C, T, H, W]
        rgb = rgb.permute(0, 2, 1, 3, 4)
        rgb_feat = self.rgb_cnn(rgb).unsqueeze(1)  # [B, 1, 512]
        rgb_feat = self.rgb_transformer(rgb_feat.permute(1, 0, 2)).permute(1, 0, 2)  # [B, 1, 512]
        rgb_out = rgb_feat[:, -1, :]  # [B, 512]

        rdm_features = []
        for i in range(3):
            rdm_i = rdm[:, i].permute(0, 2, 1, 3, 4)
            rdm_feat = self.rdm_cnn(rdm_i)
            rdm_features.append(rdm_feat)
        rdm_features = torch.stack(rdm_features, dim=1).mean(dim=1).unsqueeze(1)  # [B, 1, 512]
        rdm_feat = self.rdm_transformer(rdm_features.permute(1, 0, 2)).permute(1, 0, 2)  # [B, 1, 512]
        rdm_out = rdm_feat[:, -1, :]  # [B, 512]

        # Fusion + Attention
        fused = torch.cat([rgb_out, rdm_out], dim=1)  # [B, 1024]
        fused = self.fusion_attention(fused)
        logits = self.classifier(fused)  # [B, num_classes]
        return logits
