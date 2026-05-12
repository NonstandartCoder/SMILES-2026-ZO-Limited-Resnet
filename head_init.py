import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from collections import defaultdict

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)

def init_last_layer(layer: nn.Linear):
    backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    backbone.fc = nn.Identity()
    backbone.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone.to(device)

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
    ])

    dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=4
    )

    class_features = defaultdict(list)

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)
            for feat, target in zip(feats.cpu(), targets):
                class_features[target.item()].append(feat)

    prototypes = torch.zeros(100, 512)
    for c in range(100):
        if class_features[c]:
            prototypes[c] = torch.stack(class_features[c]).mean(dim=0)

    prototypes = nn.functional.normalize(prototypes, p=2, dim=1)

    with torch.no_grad():
        layer.weight.copy_(prototypes)
        if layer.bias is not None:
            layer.bias.zero_()