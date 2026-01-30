"""
Training script for CIFAR-10 with custom optimizers.
"""

import argparse
import os
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model import create_model
from optimizers import BatchGD, MomentumGD, AdamGD, SGD, WheelGD
from utils import (
    get_device,
    get_cifar10_loaders,
    train_epoch,
    evaluate_model,
    save_checkpoint
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CNN on CIFAR-10')
    
    parser.add_argument('--optimizer', type=str, default='batch_gd',
                        choices=['batch_gd', 'momentum', 'adam', 'sgd', 'wheel'],
                        help='Optimizer to use (default: batch_gd)')
    
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum coefficient (default: 0.9, for momentum optimizer)')
    
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam optimizer (default: 0.9)')
    
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam optimizer (default: 0.999)')
    
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Epsilon for Adam optimizer (default: 1e-8)')
    
    parser.add_argument('--step-multiplier', type=float, default=3.0,
                        help='Step multiplier for SGD optimizer (default: 3.0)')
    
    parser.add_argument('--noise-scale', type=float, default=0.8,
                        help='Initial noise scale for SGD optimizer (default: 0.8)')
    
    parser.add_argument('--noise-decay', type=float, default=0.995,
                        help='Noise decay factor for SGD optimizer (default: 0.995)')
    
    parser.add_argument('--beta', type=float, default=0.98,
                        help='Beta (momentum decay) for Wheel optimizer (default: 0.98)')
    
    parser.add_argument('--inertia', type=float, default=5.0,
                        help='Moment of inertia for Wheel optimizer (default: 5.0)')
    
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data (default: ./data)')
    
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    parser.add_argument('--tensorboard-dir', type=str, default='./runs',
                        help='Directory for TensorBoard logs (default: ./runs)')
    
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')
    
    return parser.parse_args()


def create_optimizer(model, args):
    """
    Create optimizer based on command line arguments.
    
    Args:
        model: Neural network model
        args: Parsed command line arguments
        
    Returns:
        Optimizer instance
    """
    if args.optimizer == 'batch_gd':
        optimizer = BatchGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'momentum':
        optimizer = MomentumGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = AdamGD(
            model.parameters(),
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon
        )
    elif args.optimizer == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            step_multiplier=args.step_multiplier,
            noise_scale=args.noise_scale,
            noise_decay=args.noise_decay
        )
    elif args.optimizer == 'wheel':
        optimizer = WheelGD(
            model.parameters(),
            lr=args.lr,
            beta=args.beta,
            inertia=args.inertia
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    return optimizer


def main():
    """Main training loop."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device()
    
    # Create data loaders
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    
    # Create model
    print("\nInitializing model...")
    model = create_model(num_classes=10)
    model = model.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(model, args)
    print(f"Using optimizer: {optimizer}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # TensorBoard setup
    writer = None
    if not args.no_tensorboard:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
        log_dir = Path(args.tensorboard_dir) / run_name
        writer = SummaryWriter(log_dir=log_dir)
        print(f"\nTensorBoard logging enabled: {log_dir}")
        print(f"View with: tensorboard --logdir={args.tensorboard_dir}")
        
        # Log hyperparameters
        hparams = {
            'optimizer': args.optimizer,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'seed': args.seed,
        }
        if args.optimizer == 'momentum':
            hparams['momentum'] = args.momentum
        if args.optimizer == 'adam':
            hparams['beta1'] = args.beta1
            hparams['beta2'] = args.beta2
            hparams['epsilon'] = args.epsilon
        if args.optimizer == 'sgd':
            hparams['step_multiplier'] = args.step_multiplier
            hparams['noise_scale'] = args.noise_scale
            hparams['noise_decay'] = args.noise_decay
        if args.optimizer == 'wheel':
            hparams['beta'] = args.beta
            hparams['inertia'] = args.inertia
        
        # Log model graph (first batch)
        sample_batch, _ = next(iter(train_loader))
        writer.add_graph(model, sample_batch.to(device))
        writer.flush()
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    best_accuracy = 0.0
    train_history = []
    test_history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc, per_class_acc = evaluate_model(
            model, test_loader, criterion, device
        )
        
        # Print results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Track history
        train_history.append({'epoch': epoch, 'loss': train_loss, 'accuracy': train_acc})
        test_history.append({'epoch': epoch, 'loss': test_loss, 'accuracy': test_acc})
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            writer.add_scalar('Learning_Rate', args.lr, epoch)
            
            # Log per-class accuracy
            for class_name, acc in per_class_acc.items():
                writer.add_scalar(f'Per_Class_Accuracy/{class_name}', acc, epoch)
            
            writer.flush()
        
        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = checkpoint_dir / f"model_{args.optimizer}_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, test_loss, test_acc, checkpoint_path)
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_checkpoint_path = checkpoint_dir / f"model_{args.optimizer}_best.pt"
            save_checkpoint(model, optimizer, epoch, test_loss, test_acc, best_checkpoint_path)
            print(f"  New best accuracy: {best_accuracy:.2f}%")
    
    # Final results
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print("\nPer-class accuracy (final epoch):")
    for class_name, acc in per_class_acc.items():
        print(f"  {class_name:12s}: {acc:5.2f}%")
    print("=" * 70)
    
    # Log final metrics to TensorBoard
    if writer is not None:
        writer.add_hparams(
            {
                'optimizer': args.optimizer,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
            },
            {
                'final_test_accuracy': test_acc,
                'best_test_accuracy': best_accuracy,
                'final_test_loss': test_loss,
            }
        )
        writer.close()
        print(f"\nTensorBoard logs saved to: {args.tensorboard_dir}")
        print(f"View with: tensorboard --logdir={args.tensorboard_dir}")


if __name__ == "__main__":
    main()
