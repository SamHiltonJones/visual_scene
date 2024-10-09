import torch
import torch.nn as nn
from copy import deepcopy
from collections import defaultdict
import numpy as np
import re
from torch import distributions as pyd
import matplotlib.pyplot as plt
from networks import FactorizedNoisyLinear

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class LinearOutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

def cal_dormant_ratio(model, *inputs, percentage=0.025):
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(*inputs)

    for module, hook in zip((module for module in model.modules() if isinstance(module, nn.Linear)), hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                #print(module)
                #print("hey")
                mean_output = output_data.abs().mean(0)
                #print(mean_output)
                avg_neuron_output = mean_output.mean()
                #print(avg_neuron_output)
                dormant_indices = (mean_output < avg_neuron_output *
                                   percentage).nonzero(as_tuple=True)[0]
                total_neurons += module.weight.shape[0]
                #print(len(dormant_indices))
                dormant_neurons += len(dormant_indices)

                # # Convert the tensor to a NumPy array
                # tensor_np = mean_output.cpu().detach().numpy()
                #
                # # Plot the histogram
                # plt.figure(figsize=(8, 6))
                # plt.hist(tensor_np, bins=30, color='blue', edgecolor='black', alpha=0.7)
                #
                # # Set title and labels
                # plt.title('Histogram of Tensor Values')
                # plt.xlabel('Value')
                # plt.ylabel('Frequency')
                #
                # # Show the plot
                # plt.show()



    for hook in hooks:
        hook.outputs.clear()

    for hook_handler in hook_handlers:
        hook_handler.remove()

    return dormant_neurons / total_neurons


def perturb(net, optimizer, perturb_factor):
    # Include both torch.nn.Linear and FactorizedNoisyLinear in the search
    linear_keys = [
        name for name, mod in net.named_modules()
        if isinstance(mod, (torch.nn.Linear, FactorizedNoisyLinear))
    ]

    new_net = deepcopy(net)
    new_net.apply(weight_init)

    # Update both parameters and buffers
    for name, param in net.named_parameters():
        if any(key in name for key in linear_keys):
            noise = new_net.state_dict()[name] * (1 - perturb_factor)
            param.data = param.data * perturb_factor + noise
        else:
            param.data = net.state_dict()[name]

    # Handle buffers, particularly for FactorizedNoisyLinear
    for name, buffer in net.named_buffers():
        if any(key in name for key in linear_keys):
            noise = new_net.state_dict()[name] * (1 - perturb_factor)
            buffer.data = buffer.data * perturb_factor + noise
        else:
            buffer.data = net.state_dict()[name]

    optimizer.state = defaultdict(dict)
    return net, optimizer