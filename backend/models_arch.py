import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ScalePredictor(nn.Module):
    """Predicts importance weights for different scale levels"""
    def __init__(self, in_channels, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, num_scales),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        pooled = self.global_pool(x).view(b, c)
        weights = self.fc(pooled)
        return weights

class DynamicScaleAwareFPN(nn.Module):
    """Dynamic Scale-Aware Feature Pyramid"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.num_scales = len(in_channels_list)
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        
        # Scale predictor
        self.scale_predictor = ScalePredictor(in_channels_list[-1], self.num_scales)
        
    def forward(self, features):
        # features: list of feature maps from different scales
        # Get scale weights
        scale_weights = self.scale_predictor(features[-1])
        
        # Apply lateral connections
        lateral_features = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Weighted combination
        # Upsample all to the largest size
        target_size = lateral_features[0].shape[2:]
        upsampled = []
        for feat in lateral_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(feat)
        
        # Weight and combine
        weighted_features = []
        for i, feat in enumerate(upsampled):
            weighted_features.append(feat * scale_weights[:, i].view(-1, 1, 1, 1))
        
        output = sum(weighted_features)
        return output, scale_weights

class GraphAttentionLayer(nn.Module):
    """Graph Attention for label correlation"""
    def __init__(self, in_features, out_features, num_labels):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_labels = num_labels
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x, adj=None):
        # x: [batch, num_labels, in_features]
        batch_size = x.size(0)

        Wx = self.W(x)  # [batch, num_labels, out_features]

        # Compute attention coefficients
        a_input = self._prepare_attentional_mechanism_input(Wx)
        # a_input: [batch, num_labels, num_labels, 2*out_features]

        e = self.leakyrelu(self.a(a_input))  # [batch, num_labels, num_labels, 1]
        e = e.squeeze(-1)  # [batch, num_labels, num_labels]

        attention = F.softmax(e, dim=2)  # [batch, num_labels, num_labels]

        # Apply attention
        h_prime = torch.bmm(attention, Wx)  # [batch, num_labels, out_features]

        return h_prime, attention
    
    def _prepare_attentional_mechanism_input(self, Wx):
        num_labels = Wx.size(1)

        # Wx: [batch, num_labels, out_features]
        # We need to create pairs: [batch, num_labels, num_labels, 2*out_features]

        # Repeat for all pairs
        Wx_i = Wx.unsqueeze(2).repeat(1, 1, num_labels, 1)  # [batch, num_labels, num_labels, out_features]
        Wx_j = Wx.unsqueeze(1).repeat(1, num_labels, 1, 1)  # [batch, num_labels, num_labels, out_features]

        # Concatenate pairs
        all_combinations = torch.cat([Wx_i, Wx_j], dim=3)  # [batch, num_labels, num_labels, 2*out_features]

        return all_combinations

class CrossLabelSemanticGraph(nn.Module):
    """Cross-Label Semantic Correlation Graph"""
    def __init__(self, feature_dim, num_labels, num_heads=4):
        super().__init__()
        self.num_labels = num_labels
        self.num_heads = num_heads
        
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(feature_dim, feature_dim, num_labels)
            for _ in range(num_heads)
        ])
        
        self.output_projection = nn.Linear(feature_dim * num_heads, feature_dim)
        
    def forward(self, label_features):
        # label_features: [batch, num_labels, feature_dim]
        
        multi_head_outputs = []
        attentions = []
        
        for gat in self.gat_layers:
            out, att = gat(label_features)
            multi_head_outputs.append(out)
            attentions.append(att)
        
        # Concatenate multi-head outputs
        concatenated = torch.cat(multi_head_outputs, dim=2)
        output = self.output_projection(concatenated)
        
        return output, attentions

class DeformableAttention(nn.Module):
    """Memory-efficient Deformable Attention with spatial downsampling"""
    def __init__(self, dim, num_heads=8, max_tokens=196):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.max_tokens = max_tokens  # Reduce to 14x14 = 196 tokens max

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        # Spatially downsample if needed to reduce memory
        if N > self.max_tokens:
            # Reshape to 2D, apply adaptive pooling, reshape back
            # Assuming square spatial dimensions
            h = w = int(N ** 0.5)
            target_h = target_w = int(self.max_tokens ** 0.5)

            x_2d = x.transpose(1, 2).reshape(B, C, h, w)
            x_2d = F.adaptive_avg_pool2d(x_2d, (target_h, target_w))
            x = x_2d.reshape(B, C, -1).transpose(1, 2)
            N = x.shape[1]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SpectralSpatialFusion(nn.Module):
    """Spectral-Spatial Decoupled Multi-Head Classification"""
    def __init__(self, feature_dim, num_labels):
        super().__init__()
        
        # Spectral branch
        self.spectral_branch = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_labels)
        )
        
        # Spatial branch
        self.spatial_branch = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_labels)
        )
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim, num_labels),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        spectral_out = self.spectral_branch(features)
        spatial_out = self.spatial_branch(features)
        
        gate = self.fusion_gate(features)
        
        fused_output = gate * spectral_out + (1 - gate) * spatial_out
        
        return fused_output, spectral_out, spatial_out, gate

class UncertaintyHead(nn.Module):
    """Evidential uncertainty estimation"""
    def __init__(self, feature_dim, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.fc = nn.Linear(feature_dim, num_labels)
        
    def forward(self, x):
        evidence = F.softplus(self.fc(x))
        alpha = evidence + 1
        uncertainty = self.num_labels / torch.sum(alpha, dim=1, keepdim=True)
        return evidence, uncertainty

class AMSINet(nn.Module):
    def __init__(self, num_labels, pretrained=True):
        super().__init__()
        self.num_labels = num_labels
        
        # Backbone: ResNet50 as feature extractor
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            backbone = resnet50(weights=weights)
        else:
            backbone = resnet50()
            
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels
        
        # Dynamic Scale-Aware FPN
        self.dsafp = DynamicScaleAwareFPN([256, 512, 1024, 2048], out_channels=256)
        
        # Deformable Attention
        self.deformable_attn = DeformableAttention(dim=256, num_heads=8)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Cross-Label Semantic Correlation Graph
        self.clscg = CrossLabelSemanticGraph(feature_dim=256, num_labels=num_labels, num_heads=4)
        
        # Create label embeddings
        self.label_embeddings = nn.Parameter(torch.randn(num_labels, 256))
        
        # Spectral-Spatial Fusion
        self.spectral_spatial_fusion = SpectralSpatialFusion(256, num_labels)
        
        # Uncertainty Head
        self.uncertainty_head = UncertaintyHead(256, num_labels)
        
    def forward(self, x):
        # Backbone feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Multi-scale features
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Dynamic FPN
        fpn_features, scale_weights = self.dsafp([c2, c3, c4, c5])
        
        # Flatten for attention
        b, c, h, w = fpn_features.shape
        fpn_flat = fpn_features.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Deformable attention
        attn_features = self.deformable_attn(fpn_flat)
        
        # Global pooling
        attn_features = attn_features.mean(dim=1)  # [B, C]
        
        # Prepare label-specific features
        batch_size = x.size(0)
        label_features = self.label_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add global context to each label
        global_context = attn_features.unsqueeze(1).expand(-1, self.num_labels, -1)
        label_features = label_features + global_context
        
        # Cross-Label Graph reasoning
        graph_features, graph_attentions = self.clscg(label_features)
        
        # Aggregate graph features
        aggregated_features = graph_features.mean(dim=1)  # [B, C]
        
        # Spectral-Spatial Fusion
        output, spectral_out, spatial_out, fusion_gate = self.spectral_spatial_fusion(aggregated_features)
        
        # Uncertainty estimation
        evidence, uncertainty = self.uncertainty_head(aggregated_features)
        
        return {
            'logits': output,
            'spectral_logits': spectral_out,
            'spatial_logits': spatial_out,
            'fusion_gate': fusion_gate,
            'uncertainty': uncertainty,
            'scale_weights': scale_weights,
            'graph_attentions': graph_attentions
        }
