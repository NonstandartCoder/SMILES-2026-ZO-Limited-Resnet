import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY=True

def get_train_dataset_loader(data_dir, batch_size, generator_train):
    full_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )

    targets = np.array(full_dataset.targets)
    class_indices = [np.where(targets == c)[0] for c in range(100)]

    seed = generator_train.initial_seed() % (2**32)
    rng = np.random.RandomState(seed)

    total_budget = 8192
    per_class = total_budget // 100
    remaining = total_budget - per_class * 100

    selected_indices = []
    for c in range(100):
        indices_c = class_indices[c]
        n_sample = per_class + (1 if c < remaining else 0)
        replace = n_sample > len(indices_c)
        chosen = rng.choice(indices_c, size=n_sample, replace=replace)
        selected_indices.extend(chosen)

    rng.shuffle(selected_indices)
    subset = Subset(full_dataset, selected_indices)

    train_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train,
    )

    return subset, train_loader