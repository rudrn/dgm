# Base model class for deep generative models

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """
    Abstract class for generartive models.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self) -> torch.device:
        """Get the device of model parameters"""
        return next(self.parameters()).device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a model

        Args:
            x: input tensor.

        Returns:
            Output tensor.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def sample(self, n: int) -> np.ndarray | torch.Tensor:
        """
        Generate samples from the model.

        Args:
            n: Number of samples to generate.

        Returns:
            Array or tensor of generated samples.
        """
        pass
