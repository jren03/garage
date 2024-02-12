import numpy as np
import torch
from gym.spaces import Discrete
from termcolor import cprint
from torch import nn

from garage.utils.nn_utils import create_mlp, init_ortho


class Discriminator(nn.Module):
    def __init__(self, env, clip_output: bool = False, print_clip_message=True) -> None:
        super(Discriminator, self).__init__()
        if isinstance(env.action_space, Discrete):
            self.net_arch = [64, 64]
            self.action_dim = env.action_space.n
        else:
            self.net_arch = [256, 256]
            self.action_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(env.observation_space.shape))
        net = create_mlp(
            self.obs_dim + self.action_dim,
            1,
            self.net_arch,
            activation_fn=nn.ReLU,
            output_activation_fn=None,
        )
        self.net = nn.Sequential(*net)
        self.net.apply(init_ortho)

        if clip_output:
            if print_clip_message:
                cprint("Clipping discriminator", color="magenta", attrs=["bold"])
            self.clip = True
            self.clip_min, self.clip_max = -40, 40
        else:
            self.clip = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.net(inputs)
        if self.clip:
            output = torch.clamp(output, self.clip_min, self.clip_max)
        return output.view(-1)


class DiscriminatorEnsemble(nn.Module):
    def __init__(self, env, ensemble_size=7, clip_output=False):
        super(DiscriminatorEnsemble, self).__init__()
        self.ensemble = nn.ModuleList(
            [
                Discriminator(env, clip_output=clip_output, print_clip_message=(i == 0))
                for i in range(ensemble_size)
            ]
        )
        cprint(
            f"Creating ensemble of {ensemble_size} discriminators",
            color="magenta",
            attrs=["bold"],
        )

    def forward(self, inputs):
        outputs = torch.stack([disc(inputs) for disc in self.ensemble])
        return torch.min(outputs, dim=0)[0]
