import torch
import torch.nn as nn
from torchvision import models as vision_models


class OutputNorm(nn.Module):
    def __init__(self, dim: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.mu = torch.zeros(dim, requires_grad=False).float()
        self.alpha = alpha
        print(f"Using output norm with alpha = {alpha}")

    def forward(self, x: torch.Tensor, mu: torch.Tensor = None, update_mu: bool = True):
        if mu is not None:
            return x - mu
        else:
            if update_mu:
                with torch.no_grad():
                    self.mu = self.mu.to(x.device)
                    self.mu = self.alpha * self.mu + (1 - self.alpha) * x.mean(axis=0)
            return x - self.mu
        
        
class StackedFrames(nn.Module):
    """
    Build representations for a stack of frames using a common architecture.
    """

    def __init__(
        self,
        n_outputs: int,
        base_architecture: str = "resnet18",
        flat_output: bool = True,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        base_model = getattr(vision_models, base_architecture)
        self.model = base_model(False, num_classes=n_outputs)
        if batch_norm:
            self.bn = nn.BatchNorm1d(n_outputs)
        else:
            self.bn = lambda x: x
        self.n_outputs = n_outputs
        self.flat_output = flat_output

    def forward(self, x):
        # assume input format [batch, time, channels, rows, cols]
        if x.ndim == 4:
            return self.model(x)
        else:
            shape = list(x.shape)
            flat_shape = [shape[0] * shape[1]] + shape[2:]
            out_shape = [shape[0], shape[1], self.n_outputs]
            output = self.model(x.reshape(*flat_shape)).reshape(*out_shape)
            output = self.bn(output.view((-1, output.shape[-1]))).view(output.shape)
            if self.flat_output:
                return torch.cat(
                    [output[:, i, :] for i in range(output.shape[1])], axis=-1
                )
            else:
                return output


            
class DifferenceModel(nn.Module):
    def __init__(
        self, n_latents, batch_norm: bool = False, detach_velocity: bool = False
    ) -> None:
        super().__init__()
        self.base_model = StackedFrames(
            n_outputs=n_latents * 10, flat_output=False, batch_norm=batch_norm
        )
        self.output = nn.Sequential(nn.Linear(n_latents * 10, 2))
        self.detach_vel = detach_velocity
        self.vel_scale = nn.parameter.Parameter(
            torch.ones(
                1,
            )
        )
        self.acc_scale = nn.parameter.Parameter(
            torch.ones(
                1,
            )
        )

    def forward(self, x):
        representation = self.base_model(x)
        output = self.output(representation)
        pos = output[:, 0, :]
        if self.detach_vel:
            vel_1 = self.vel_scale * ((output[:, 0, :] - output[:, 1, :]).detach())
            vel_2 = self.vel_scale * ((output[:, 1, :] - output[:, 2, :]).detach())
        else:
            vel_1 = self.vel_scale * ((output[:, 0, :] - output[:, 1, :]))
            vel_2 = self.vel_scale * ((output[:, 1, :] - output[:, 2, :]))
        acc = self.acc_scale * (vel_1 - vel_2)
        return torch.cat([pos, vel_1, acc], dim=1)
    
    
class Encoder(nn.Module):
    def __init__(
        self,
        n_latents,
        base_architecture,
        n_frames: int = 1,
        output_norm: bool = False,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        # TODO: layers can be specified in a hydra config file
        layers = [
            StackedFrames(
                n_outputs=n_latents * 10, base_architecture=base_architecture
            ),
            nn.LeakyReLU(),
            nn.Linear(n_latents * 10 * n_frames, n_latents * 10),
            nn.LeakyReLU(),
            nn.Linear(n_latents * 10, n_latents),
        ]
        if output_norm:
            layers += [OutputNorm(n_latents, alpha=0.5)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)