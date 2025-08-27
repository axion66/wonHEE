import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, TensorDataset
import argparse
import os
from collections import Counter
from sklearn.preprocessing import StandardScaler
import random

class ECGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx])

def loader(
    features_path='dataset/MIT-BIH_features.npy',
    labels_path='dataset/MIT-BIH_labels.npy',
    batch_size=16,
    balance_labels=True,
    split_ratio=0.8,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Missing .npy files. Please run preprocessing first to generate them.")

    features = np.load(features_path)
    labels = np.load(labels_path)

    dataset = ECGDataset(features, labels)

    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_features = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    test_features = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    scaler = StandardScaler().fit(train_features.view(-1, 1).numpy())
    train_features = torch.from_numpy(
        scaler.transform(train_features.view(-1, 1).numpy()).reshape(train_features.shape).astype(np.float32)
    )
    test_features = torch.from_numpy(
        scaler.transform(test_features.view(-1, 1).numpy()).reshape(test_features.shape).astype(np.float32)
    )

    if balance_labels:
        class_counts = Counter(train_labels.numpy())
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label.item()] for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True,
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=batch_size,
            sampler=sampler
        )
    else:
        train_loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=batch_size,
            shuffle=True
        )

    test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=batch_size)

    return train_loader, test_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create PyTorch DataLoaders for MIT-BIH dataset.")
    parser.add_argument('--features_path', type=str, default='dataset/MIT-BIH_features.npy')
    parser.add_argument('--labels_path', type=str, default='dataset/MIT-BIH_labels.npy')
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--balance_labels', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_loader, test_loader = loader(
        features_path=args.features_path,
        labels_path=args.labels_path,
        batch_size=args.batch_size,
        balance_labels=args.balance_labels,
        split_ratio=args.split_ratio,
        seed=args.seed
    )

    print(f"\nTrain loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    x, y = next(iter(train_loader))
    print(f"Sample batch shape: {x.shape}, labels shape: {y.shape}")
