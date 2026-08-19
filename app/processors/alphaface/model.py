"""Inference-only AlphaFace swapper architecture.

Adapted from AlphaFace's official MIT-licensed implementation at
https://github.com/andrewyu90/Alphaface_Official (commit d41fbd4).
Training-only modules and dependencies are intentionally omitted.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveInstanceNormalization(nn.Module):
    @staticmethod
    def _mean(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    @classmethod
    def _std(cls, x: torch.Tensor) -> torch.Tensor:
        centered = (x.permute(2, 3, 0, 1) - cls._mean(x)).permute(2, 3, 0, 1)
        variance = (torch.sum(centered**2, (2, 3)) + 2.3e-8) / (x.shape[2] * x.shape[3])
        return torch.sqrt(variance)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        normalized = (x.permute(2, 3, 0, 1) - self._mean(x)) / self._std(x)
        return (self._std(style) * normalized + self._mean(style)).permute(2, 3, 0, 1)


class IdentityFeedingBlock(nn.Module):
    def __init__(self, output_dim: int, identity_dim: int = 512) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(identity_dim, output_dim)
        self.AdaIN = AdaptiveInstanceNormalization()

    def forward(
        self, identity: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.fc(identity).unsqueeze(2).unsqueeze(3)
        midpoint = int(self.output_dim / 2)
        first = projected[:, 0:midpoint, :, :]
        second = projected[:, midpoint : self.output_dim, :, :]
        first = (first + self.AdaIN(first, target)) / 2.0
        second = (second + self.AdaIN(second, target)) / 2.0
        return first, second


class OperationUnit(nn.Module):
    def __init__(self, channels: int, identity_output_dim: int, activate: bool) -> None:
        super().__init__()
        self.activate = activate
        self.Conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.activation = nn.ReLU()
        self.IFF = IdentityFeedingBlock(identity_output_dim)

    def forward(self, features: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        scale, bias = self.IFF(identity, features)
        output = F.pad(features, (1, 1, 1, 1), mode="reflect")
        output = self.Conv1(output)
        output = output - torch.mean(output, dim=(2, 3), keepdim=True)
        variance = torch.mean(torch.mul(output, output), (2, 3), keepdim=True)
        inverse_std = torch.div(1.0, torch.sqrt(torch.add(variance, 1.0e-8)))
        output = torch.add(torch.mul(scale, torch.mul(output, inverse_std)), bias)
        return self.activation(output) if self.activate else output


class CrossAdaptiveIdentityInjectionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.OP1 = OperationUnit(1024, 2048, activate=True)
        self.OP2 = OperationUnit(1024, 2048, activate=False)

    def forward(self, features: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return features + self.OP2(self.OP1(features, identity), identity)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = [3, 128, 256, 512, 1024]
        kernels = [7, 3, 3, 3]
        paddings = [0, 1, 1, 1]
        strides = [1, 1, 2, 2]
        self.Encoder = nn.ModuleDict(
            {
                f"layer_{index}": nn.Sequential(
                    nn.Conv2d(
                        channels[index],
                        channels[index + 1],
                        kernel_size=kernels[index],
                        stride=strides[index],
                        padding=paddings[index],
                    ),
                    nn.LeakyReLU(0.2),
                )
                for index in range(4)
            }
        )
        self.fusion_module = nn.ModuleDict(
            {
                f"fusion_layer_{index}": CrossAdaptiveIdentityInjectionBlock()
                for index in range(6)
            }
        )

    def forward(self, target: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        output = F.pad(target, (3, 3, 3, 3), mode="reflect")
        for index in range(4):
            output = self.Encoder[f"layer_{index}"](output)
        for index in range(6):
            output = self.fusion_module[f"fusion_layer_{index}"](output, identity)
        return output


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Upsample = nn.Upsample(
            scale_factor=2, align_corners=False, mode="bilinear"
        )
        self.Conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.Conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.Conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.Conv4_new = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Conv5_new = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.Conv6_new = nn.Conv2d(128, 3, kernel_size=5, padding=0)
        self.Activation_LeakyRelu = nn.LeakyReLU(0.2)
        self.Activation_Tanh = nn.Tanh()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.Upsample(features)
        output = self.Activation_LeakyRelu(self.Conv1(output))
        output = self.Upsample(output)
        output = self.Activation_LeakyRelu(self.Conv2(output))
        output = self.Activation_LeakyRelu(self.Conv3(output))
        output = self.Activation_LeakyRelu(self.Conv4_new(output))
        output = self.Activation_LeakyRelu(self.Conv5_new(output))
        output = F.pad(output, (3, 3, 3, 3), mode="reflect")
        output = self.Activation_Tanh(self.Conv6_new(output))
        return (output + 1.0) / 2.0


class AlphaFaceSwapper(nn.Module):
    """The released 256px AlphaFace swapper without training dependencies."""

    def __init__(self) -> None:
        super().__init__()
        # Attribute names match the official checkpoint exactly.
        self.E = Encoder()
        self.G = Decoder()

    def forward(
        self, target: torch.Tensor, source_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.G(self.E(target, source_embedding))
