"""
Gradient descent optimizer implementations for visualization.

This package contains modular optimizer implementations that can be imported
directly for backward compatibility with the original monolithic module.
"""

from .sgd import stochastic_gradient_descent
from .batch_gd import batch_gradient_descent
from .momentum import momentum_gradient_descent
from .adam import adam_optimizer
from .wheel import wheel_optimizer
from .ballistic import (
    find_collision_newton,
    ballistic_gradient_descent,
    ballistic_adam_optimizer
)

__all__ = [
    'stochastic_gradient_descent',
    'batch_gradient_descent',
    'momentum_gradient_descent',
    'adam_optimizer',
    'wheel_optimizer',
    'find_collision_newton',
    'ballistic_gradient_descent',
    'ballistic_adam_optimizer',
]
