"""
Evaluation script for trained CIFAR-10 models.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

from model import create_model
from utils import (
    get_device,
    get_cifar10_loaders,
    evaluate_model,
    load_checkpoint
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained CNN on CIFAR-10')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data (default: ./data)')
    
    parser.add_argument('--save-confusion-matrix', action='store_true',
                        help='Save confusion matrix plot')
    
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results (default: ./results)')
    
    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def get_predictions(model, test_loader, device):
    """
    Get all predictions from the model.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        device: Device to use for computation
        
    Returns:
        tuple: (all_predictions, all_targets)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device()
    
    # Create data loader
    print("\nLoading CIFAR-10 test dataset...")
    _, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    
    # Create model
    print("\nInitializing model...")
    model = create_model(num_classes=10)
    model = model.to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    epoch, loss, accuracy = load_checkpoint(model, None, args.checkpoint, device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    print("\nEvaluating model on test set...")
    test_loss, test_acc, per_class_acc = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Overall Test Loss:     {test_loss:.4f}")
    print(f"Overall Test Accuracy: {test_acc:.2f}%")
    print("\nPer-Class Accuracy:")
    print("-" * 70)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for class_name, acc in per_class_acc.items():
        print(f"  {class_name:12s}: {acc:5.2f}%")
    
    print("=" * 70)
    
    # Generate confusion matrix if requested
    if args.save_confusion_matrix:
        print("\nGenerating confusion matrix...")
        predictions, targets = get_predictions(model, test_loader, device)
        
        cm_path = output_dir / 'confusion_matrix.png'
        plot_confusion_matrix(targets, predictions, class_names, save_path=cm_path)
    
    # Save results to file
    results_file = output_dir / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Evaluation Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Overall Test Loss: {test_loss:.4f}\n")
        f.write(f"Overall Test Accuracy: {test_acc:.2f}%\n")
        f.write("\nPer-Class Accuracy:\n")
        f.write("-" * 70 + "\n")
        for class_name, acc in per_class_acc.items():
            f.write(f"  {class_name:12s}: {acc:5.2f}%\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
