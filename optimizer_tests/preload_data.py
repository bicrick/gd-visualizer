"""
Script to preload and verify CIFAR-10 dataset.
"""

import argparse
from torchvision import datasets, transforms
from pathlib import Path


def preload_cifar10(data_dir='./data'):
    """
    Download and verify CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store CIFAR-10 data
    """
    print("=" * 70)
    print("CIFAR-10 Dataset Preloader")
    print("=" * 70)
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nData directory: {data_path.absolute()}")
    
    # Simple transform for verification
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download training set
    print("\nDownloading training set...")
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    print(f"Training samples: {len(train_dataset)}")
    
    # Download test set
    print("\nDownloading test set...")
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # Verify data
    print("\nVerifying dataset...")
    train_sample, train_label = train_dataset[0]
    test_sample, test_label = test_dataset[0]
    
    print(f"Sample image shape: {train_sample.shape}")
    print(f"Sample label: {train_label}")
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"\nClasses ({len(class_names)}): {', '.join(class_names)}")
    
    print("\n" + "=" * 70)
    print("Dataset preloaded successfully!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Preload CIFAR-10 dataset')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store CIFAR-10 data (default: ./data)')
    args = parser.parse_args()
    
    preload_cifar10(args.data_dir)


if __name__ == "__main__":
    main()
