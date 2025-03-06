import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Efficient Channel Attention (ECA) - Modified for Stability
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        kernel_size = max(3, int((torch.log2(torch.tensor(channels, dtype=torch.float32)) / gamma + b).item()))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)  # Global Average Pooling
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

# Lightweight Edge Feature Extractor - No In-place Operations
class EdgeFeatureExtractor(nn.Module):
    def __init__(self):
        super(EdgeFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))  # No In-place SiLU
        x = F.silu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x.view(x.size(0), -1)  # Flatten

# Feature Fusion with ECA - No Unnecessary Reshapes
class FeatureFusion(nn.Module):
    def __init__(self, backbone_dim, edge_dim):
        super(FeatureFusion, self).__init__()
        self.eca = ECA(backbone_dim + edge_dim)
        self.fc = nn.Linear(backbone_dim + edge_dim, backbone_dim)

    def forward(self, backbone_features, edge_features):
        fused_features = torch.cat((backbone_features, edge_features), dim=1)
        fused_features = self.eca(fused_features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # Apply ECA
        return F.silu(self.fc(fused_features))

# Shape Classification Model with EfficientNet-B0 and Feature Fusion
class Shape2DClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(Shape2DClassifier, self).__init__()

        # EfficientNet-B0 as Feature Extractor
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone_dim = 1280  # Output dimension of EfficientNet-B0
        self.backbone.classifier = nn.Identity()  # Remove classifier layer

        # Freezing all layers except the last block
        for param in self.backbone.features[:-2].parameters():
            param.requires_grad = False

        # Edge Feature Extractor
        self.edge_extractor = EdgeFeatureExtractor()
        self.edge_dim = 32  # Output from lightweight CNN

        # Feature Fusion using ECA
        self.fusion = FeatureFusion(self.backbone_dim, self.edge_dim)

        # Optimized Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),  # Lower dropout for faster inference
            nn.Linear(128, num_classes),
        )

    def forward(self, image):
        backbone_features = self.backbone(image)
        edge_features = self.edge_extractor(image)
        fused_features = self.fusion(backbone_features, edge_features)
        return self.classifier(fused_features)
