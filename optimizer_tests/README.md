# CIFAR-10 Optimizer Testing Suite

A standalone testing suite for comparing gradient descent optimizers on CIFAR-10 image classification using PyTorch.

## Overview

This project implements custom gradient descent optimizers from scratch and evaluates their performance on the CIFAR-10 dataset. The goal is to compare how different optimization algorithms converge and achieve accuracy on a real-world deep learning task.

## Features

- Custom PyTorch optimizer implementations:
  - Batch Gradient Descent
  - Gradient Descent with Momentum
  - ADAM (Adaptive Moment Estimation)
  - SGD (Stochastic Gradient Descent)
- Simple CNN architecture (~50K parameters)
- Training and evaluation scripts with comprehensive metrics
- Apple Silicon (MPS) GPU support
- Confusion matrix visualization
- Per-class accuracy analysis

## Project Structure

```
optimizer_tests/
├── environment.yml          # Conda environment specification
├── requirements.txt         # Python dependencies
├── preload_data.py         # Dataset preloader
├── model.py                # SimpleCNN architecture
├── optimizers/             # Custom optimizer implementations
│   ├── __init__.py
│   ├── batch_gd.py        # Batch gradient descent
│   ├── momentum_gd.py     # GD with momentum
│   ├── adam_gd.py         # ADAM optimizer
│   └── sgd.py             # Stochastic gradient descent
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── utils.py               # Helper functions
├── data/                  # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/           # Saved model checkpoints
├── runs/                  # TensorBoard logs
└── results/               # Evaluation results and plots
```

## Setup

### Prerequisites

- macOS with Apple Silicon (for MPS support) or any system with Python 3.11+
- Conda package manager

### Installation

1. Create and activate the conda environment:

```bash
cd optimizer_tests
conda env create -f environment.yml
conda activate cifar10-opt
```

Alternatively, use pip:

```bash
pip install -r requirements.txt
```

## Usage

### Preload Dataset (Recommended)

Download CIFAR-10 before training to avoid delays:

```bash
python preload_data.py
```

This will download ~170MB of data to the `./data` directory.

### Training

Train a model with batch gradient descent:

```bash
python train.py --optimizer batch_gd --epochs 10 --lr 0.01
```

Train a model with momentum:

```bash
python train.py --optimizer momentum --epochs 10 --lr 0.01 --momentum 0.9
```

Train a model with ADAM:

```bash
python train.py --optimizer adam --epochs 10 --lr 0.001 --beta1 0.9 --beta2 0.999
```

Train a model with SGD (with gradient noise simulation):

```bash
python train.py --optimizer sgd --epochs 10 --lr 0.01 --step-multiplier 3.0 --noise-scale 0.8
```

#### Training Arguments

- `--optimizer`: Optimizer to use (`batch_gd`, `momentum`, `adam`, or `sgd`)
- `--lr`: Learning rate (default: 0.01)
- `--momentum`: Momentum coefficient for momentum optimizer (default: 0.9)
- `--beta1`: Beta1 for Adam optimizer (default: 0.9)
- `--beta2`: Beta2 for Adam optimizer (default: 0.999)
- `--epsilon`: Epsilon for Adam optimizer (default: 1e-8)
- `--step-multiplier`: Step multiplier for SGD optimizer (default: 3.0)
- `--noise-scale`: Initial noise scale for SGD optimizer (default: 0.8)
- `--noise-decay`: Noise decay factor for SGD optimizer (default: 0.995)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 64)
- `--data-dir`: Directory for CIFAR-10 data (default: ./data)
- `--checkpoint-dir`: Directory to save checkpoints (default: ./checkpoints)
- `--save-every`: Save checkpoint every N epochs (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)

#### Optimizer Details

**Batch GD**: Classic gradient descent with full batch updates. Stable but can be slow.

**Momentum GD**: Adds momentum to accelerate convergence and dampen oscillations.

**ADAM**: Adaptive learning rates with bias-corrected momentum. Often the best default choice for deep learning.

**SGD**: Simulates stochastic mini-batch behavior by adding controlled gradient noise. Features:
- Adds perpendicular and magnitude noise to gradients
- Uses a step multiplier for faster convergence
- Gradually reduces noise over time (simulating learning rate annealing)
- Shows the trade-off: faster convergence but noisier path
- `--tensorboard-dir`: Directory for TensorBoard logs (default: ./runs)
- `--no-tensorboard`: Disable TensorBoard logging

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --checkpoint checkpoints/model_momentum_best.pt
```

Generate confusion matrix:

```bash
python evaluate.py --checkpoint checkpoints/model_momentum_best.pt --save-confusion-matrix
```

#### Evaluation Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--batch-size`: Batch size for evaluation (default: 64)
- `--data-dir`: Directory for CIFAR-10 data (default: ./data)
- `--save-confusion-matrix`: Generate and save confusion matrix plot
- `--output-dir`: Directory to save results (default: ./results)

### TensorBoard Visualization

Monitor training in real-time with TensorBoard:

```bash
# Start TensorBoard (in a separate terminal)
tensorboard --logdir=./runs

# Then open http://localhost:6006 in your browser
```

TensorBoard automatically logs:
- Training and test loss
- Training and test accuracy
- Per-class accuracy for all 10 classes
- Model graph
- Hyperparameters

You can compare multiple runs side-by-side by training with different optimizers or hyperparameters.

## Model Architecture

The SimpleCNN architecture consists of:

```
Input (3x32x32)
├── Conv2D (3 -> 32, 3x3) + ReLU + MaxPool (2x2) -> 32x16x16
├── Conv2D (32 -> 64, 3x3) + ReLU + MaxPool (2x2) -> 64x8x8
├── Flatten -> 4096
├── Linear (4096 -> 128) + ReLU
└── Linear (128 -> 10)
```

Total parameters: ~50,000

## Optimizers

### Batch Gradient Descent

Simple gradient descent with fixed learning rate:

```
param = param - lr * grad
```

**Hyperparameters:**
- Learning rate: 0.01

### Momentum Gradient Descent

Accumulates velocity to accelerate convergence:

```
v = momentum * v + grad
param = param - lr * v
```

**Hyperparameters:**
- Learning rate: 0.01
- Momentum: 0.9

## Expected Results

With the default configuration (10 epochs):

- **Batch GD**: ~55-65% test accuracy
- **Momentum GD**: ~60-70% test accuracy

Momentum typically converges faster and achieves higher accuracy due to better handling of local minima and plateaus.

## Hardware Acceleration

The code automatically detects and uses:

1. **MPS** (Metal Performance Shaders) for Apple Silicon Macs
2. **CUDA** for NVIDIA GPUs
3. **CPU** as fallback

Expected speedup with MPS vs CPU: 3-5x

## Tips

1. **Memory issues**: Reduce batch size if you encounter out-of-memory errors
2. **Faster training**: Increase batch size for better GPU utilization
3. **Better accuracy**: Train for more epochs (20-50) or use learning rate scheduling
4. **Reproducibility**: Set the same random seed across runs

## Example Workflow

Complete workflow from training to evaluation:

```bash
# Activate environment
conda activate cifar10-opt

# Preload dataset (one-time setup)
python preload_data.py

# Start TensorBoard in a separate terminal
tensorboard --logdir=./runs

# Train with momentum optimizer
python train.py --optimizer momentum --epochs 20 --lr 0.01

# Evaluate the best checkpoint
python evaluate.py \
    --checkpoint checkpoints/model_momentum_best.pt \
    --save-confusion-matrix

# Train with batch gradient descent for comparison
python train.py --optimizer batch_gd --epochs 20 --lr 0.01

# Evaluate batch GD
python evaluate.py \
    --checkpoint checkpoints/model_batch_gd_best.pt \
    --save-confusion-matrix

# Compare both runs in TensorBoard at http://localhost:6006
```

## Future Extensions

- Implement wheel optimizer
- Add learning rate scheduling
- Support for mini-batch SGD variants
- TensorBoard logging
- Early stopping based on validation loss
- Data augmentation strategies
- Deeper architectures (ResNet, VGG)

## CIFAR-10 Dataset

The dataset is automatically downloaded on first run (~170MB). It contains:

- **Training set**: 50,000 images
- **Test set**: 10,000 images
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Image size**: 32x32 RGB

## Troubleshooting

### MPS Not Available

If MPS is not detected on Apple Silicon:

1. Ensure macOS 12.3+ is installed
2. Update PyTorch: `conda update pytorch`
3. Verify: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Training is Slow

- Check if GPU is being used (output should say "Using MPS" or "Using CUDA")
- Increase batch size to better utilize GPU
- Reduce number of workers if CPU bottleneck: `--batch-size 128`

### Low Accuracy

- Train for more epochs (20-50)
- Try different learning rates (0.001, 0.005, 0.05)
- Ensure data augmentation is enabled (it is by default in training)
- Check for bugs in optimizer implementation

## License

MIT

## Contributing

This is part of the gradient descent experiments project. Feel free to add new optimizers in the `optimizers/` directory following the existing structure.
