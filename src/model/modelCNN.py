import torch
import torch.nn as nn
from torchvision import models


class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(SkinCancerCNN, self).__init__()

        # Pretrained model
        self.pretrained_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Classifier
        num_features = self.pretrained_model.classifier[1].in_features
        self.pretrained_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.pretrained_model(x)


if __name__ == "__main__":
    model = SkinCancerCNN()
    print("model made")
