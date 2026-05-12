import torchvision.transforms as T

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
    else:
        return T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ])
