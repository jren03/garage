from typing import Callable, List, Optional, Type, Union

import torch
from termcolor import cprint
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def gradient_penalty(
    learner_sa: torch.Tensor,
    expert_sa: torch.Tensor,
    f: nn.Module,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Calculates the gradient penalty for the given learner and expert state-action tensors.

    Args:
        learner_sa (torch.Tensor): The state-action tensor from the learner.
        expert_sa (torch.Tensor): The state-action tensor from the expert.
        f (nn.Module): The discriminator network.
        device (str, optional): The device to use. Defaults to "cuda".

    Returns:
        torch.Tensor: The gradient penalty.
    """
    batch_size = expert_sa.size()[0]

    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(expert_sa)

    interpolated = alpha * expert_sa.data + (1 - alpha) * learner_sa.data

    interpolated = Variable(interpolated, requires_grad=True).to(device)

    f_interpolated = f(interpolated.float()).to(device)

    gradients = torch_grad(
        outputs=f_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(f_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    gradients = gradients.view(batch_size, -1)

    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    # 2 * |f'(x_0)|
    return ((gradients_norm - 0.4) ** 2).mean()


def init_ortho(layer: nn.Module) -> None:
    """
    Initialize the weight of a linear layer with orthogonal initialization.

    Args:
        layer (nn.Module): The linear layer to be initialized.

    Returns:
        None
    """
    if type(layer) == nn.Linear:
        nn.init.orthogonal_(layer.weight)


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    output_activation_fn: Optional[Type[nn.Module]] = None,
) -> List[nn.Module]:
    """
    Create a multi-layer perceptron (MLP) neural network.

    Args:
        input_dim (int): The dimension of the input layer.
        output_dim (int): The dimension of the output layer.
        net_arch (List[int]): A list of integers representing the number of units in each hidden layer.
        activation_fn (Type[nn.Module], optional): The activation function to be used in the hidden layers. Defaults to nn.ReLU.
        output_activation_fn (Optional[Type[nn.Module]], optional): The activation function to be used in the output layer. Defaults to None.

    Returns:
        List[nn.Module]: A list of nn.Module objects representing the layers of the MLP.
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
        if output_activation_fn is not None:
            cprint(
                f"Appending {output_activation_fn} to MLP", color="cyan", attrs=["bold"]
            )
            modules.append(output_activation_fn())
    return modules
