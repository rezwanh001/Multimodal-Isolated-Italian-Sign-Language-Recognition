import torch
import torch.nn as nn
import torchvision.models.video as video_models
from timm.models import create_model

##---------------------------------------------------------------------------------------------
class TwoStreamCNNLSTM(nn.Module):
    '''
    Backbone Method: `3D ResNet`
    '''
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
    
##---------------------------------------------------------------------------------------------
class AdvancedTwoStreamModel(nn.Module):
    '''
    Backbone Method: `MC3`
    '''
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
    
##---------------------------------------------------------------------------------------------
class FusionAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.fc(attn_output.mean(dim=1))

class UltraAdvancedTwoStreamModel(nn.Module):
    '''
    Backbone Method: `R(2+1)D`
    '''
    def __init__(self, num_classes=126, hidden_size=512, num_layers=2):
        super().__init__()
        self.rgb_cnn = video_models.r2plus1d_18(weights='KINETICS400_V1')
        self.rgb_cnn.fc = nn.Identity()
        self.rdm_cnn = video_models.r2plus1d_18(weights='KINETICS400_V1')
        self.rdm_cnn.fc = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024)
        self.rgb_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.rdm_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fusion_attention = FusionAttention(1024, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, rgb, rdm):
        rgb = rgb.permute(0, 2, 1, 3, 4)
        rgb_feat = self.rgb_cnn(rgb).unsqueeze(1)
        rgb_feat = self.rgb_transformer(rgb_feat.permute(1, 0, 2)).permute(1, 0, 2)
        rgb_out = rgb_feat[:, -1, :]
        rdm_features = []
        for i in range(3):
            rdm_i = rdm[:, i].permute(0, 2, 1, 3, 4)
            rdm_feat = self.rdm_cnn(rdm_i)
            rdm_features.append(rdm_feat)
        rdm_features = torch.stack(rdm_features, dim=1).mean(dim=1).unsqueeze(1)
        rdm_feat = self.rdm_transformer(rdm_features.permute(1, 0, 2)).permute(1, 0, 2)
        rdm_out = rdm_feat[:, -1, :]
        fused = torch.cat([rgb_out, rdm_out], dim=1).unsqueeze(1)
        fused = self.fusion_attention(fused)
        logits = self.classifier(fused)
        return logits
    
##---------------------------------------------------------------------------------------------
class SwinTwoStreamModel(nn.Module):
    '''
    Backbone Method: `Swin-B`
    '''
    def __init__(self, num_classes=126, hidden_size=512, swin_model_name='swin_base_patch4_window7_224'):
        """
        Args:
            num_classes (int): Number of output classes.
            hidden_size (int): Size of the hidden layer after fusion.
            swin_model_name (str): Name of the Swin model in timm (must be available in your timm version).
        """
        super().__init__()
        # Load pretrained Swin Transformer for RGB and RDM streams
        self.rgb_swin = create_model(
            swin_model_name, pretrained=True, num_classes=0  # no final head
        )
        self.rdm_swin = create_model(
            swin_model_name, pretrained=True, num_classes=0
        )
        # Project Swin output to hidden_size if needed
        self.rgb_proj = nn.Linear(self.rgb_swin.num_features, hidden_size)
        self.rdm_proj = nn.Linear(self.rdm_swin.num_features, hidden_size)
        # Fusion and classifier
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, rgb, rdm):
        # rgb: [B, T, C, H, W]
        # rdm: [B, 3, T, C, H, W]
        B, T, C, H, W = rgb.shape
        rgb = rgb.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        rgb = rgb.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        rgb = rgb.reshape(B * T, C, H, W)  # [B*T, C, H, W]
        rgb_feat = self.rgb_swin(rgb)      # [B*T, feat_dim]
        rgb_feat = rgb_feat.view(B, T, -1).mean(dim=1)  # [B, feat_dim]
        rgb_feat = self.rgb_proj(rgb_feat)  # [B, hidden_size]

        rdm_features = []
        for i in range(3):
            rdm_i = rdm[:, i]  # [B, T, C, H, W]
            rdm_i = rdm_i.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            rdm_i = rdm_i.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            rdm_i = rdm_i.reshape(B * T, C, H, W)  # [B*T, C, H, W]
            rdm_feat = self.rdm_swin(rdm_i)        # [B*T, feat_dim]
            rdm_feat = rdm_feat.view(B, T, -1).mean(dim=1)  # [B, feat_dim]
            rdm_feat = self.rdm_proj(rdm_feat)
            rdm_features.append(rdm_feat)
        rdm_feat = torch.stack(rdm_features, dim=1).mean(dim=1)  # [B, hidden_size]

        fused = torch.cat([rgb_feat, rdm_feat], dim=1)  # [B, hidden_size*2]
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return logits