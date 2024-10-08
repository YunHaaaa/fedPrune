import sys
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils import prune
import torchvision
import prune as torch_prune
import warnings


# Utility functions

def needs_mask(name):
    return name.endswith('weight')


def initialize_mask(model, dtype=torch.bool):
    layers_to_prune = (layer for _, layer in model.named_children())
    for layer in layers_to_prune:
        for name, param in layer.named_parameters():
            if name.endswith('weight'):
                if hasattr(layer, name + '_mask'):
                    warnings.warn(
                        'Parameter has a pruning mask already. '
                        'Reinitialize to an all-one mask.'
                    )
                layer.register_buffer(name + '_mask', torch.ones_like(param, dtype=dtype))
                continue
                parent = name[:name.rfind('.')]

                for mname, module in layer.named_modules():
                    if mname != parent:
                        continue
                    module.register_buffer(name[name.rfind('.')+1:] + '_mask', torch.ones_like(param, dtype=dtype))


class PrunableNet(nn.Module):
    '''Common functionality for all networks in this experiment.'''

    def __init__(self, device='cpu'):
        super(PrunableNet, self).__init__()
        self.device = torch.device(device)

        self.communication_sparsity = 0


    def init_param_sizes(self):
        # bits required to transmit mask and parameters?
        self.mask_size = 0
        self.param_size = 0
        for _, layer in self.named_children():
            for name, param in layer.named_parameters():
                param_size = np.prod(param.size())
                self.param_size += param_size * 32 # FIXME: param.dtype.size?
                if needs_mask(name):
                    self.mask_size += param_size
        #print(f'Masks require {self.mask_size} bits.')
        #print(f'Weights require {self.param_size} bits.')
        #print(f'Unmasked weights require {self.param_size - self.mask_size*32} bits.')


    def clear_gradients(self):
        for _, layer in self.named_children():
            for _, param in layer.named_parameters():
                del param.grad
        torch.cuda.empty_cache()


    def infer_mask(self, masking):
        for name, param in self.state_dict().items():
            if needs_mask(name) and name in masking.masks:
                mask_name = name + "_mask"
                mask = self.state_dict()[mask_name]
                mask.copy_(masking.masks[name])


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def _decay(self, t, alpha=0.3, t_end=400):
        if t >= t_end:
            return 0
        return alpha/2 * (1 + np.cos(t*np.pi / t_end))


    def _weights_by_layer(self, sparsity=0.1, sparsity_distribution='erk'):
        with torch.no_grad():
            layer_names = []
            sparsities = np.empty(len(list(self.named_children())))
            n_weights = np.zeros_like(sparsities, dtype=int)

            for i, (name, layer) in enumerate(self.named_children()):

                layer_names.append(name)
                for pname, param in layer.named_parameters():
                    n_weights[i] += param.numel()

                if sparsity_distribution == 'uniform':
                    sparsities[i] = sparsity
                    continue
                
                kernel_size = None
                if isinstance(layer, nn.modules.conv._ConvNd):
                    neur_out = layer.out_channels
                    neur_in = layer.in_channels
                    kernel_size = layer.kernel_size
                elif isinstance(layer, nn.Linear):
                    neur_out = layer.out_features
                    neur_in = layer.in_features
                else:
                    raise ValueError('Unsupported layer type ' + type(layer))

                if sparsity_distribution == 'er':
                    sparsities[i] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                elif sparsity_distribution == 'erk':
                    if isinstance(layer, nn.modules.conv._ConvNd):
                        sparsities[i] = 1 - (neur_in + neur_out + np.sum(kernel_size)) / (neur_in * neur_out * np.prod(kernel_size))
                    else:
                        sparsities[i] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                else:
                    raise ValueError('Unsupported sparsity distribution ' + sparsity_distribution)
                
            # Now we need to renormalize sparsities.
            # We need global sparsity S = sum(s * n) / sum(n) equal to desired
            # sparsity, and s[i] = C n[i]
            sparsities *= sparsity * np.sum(n_weights) / np.sum(sparsities * n_weights)
            n_weights = np.floor((1-sparsities) * n_weights)

            return {layer_names[i]: n_weights[i] for i in range(len(layer_names))}


    def layer_prune(self, sparsity=0.1, sparsity_distribution='erk', pruning_type='hard'):
        '''
        Prune the network to the desired sparsity, following the specified
        sparsity distribution. The weight magnitude is used for pruning.

        uniform: layer sparsity = global sparsity
        er: Erdos-Renyi
        erk: Erdos-Renyi Kernel
        
        pruning_type: 'hard' -> set weight to 0
                    'soft' -> set mask to 0 but keep weight
        '''

        #print('desired sparsity', sparsity)
        with torch.no_grad():
            weights_by_layer = self._weights_by_layer(sparsity=sparsity, sparsity_distribution=sparsity_distribution)
            for name, layer in self.named_children():

                # We need to figure out how many to prune
                n_total = 0
                for bname, buf in layer.named_buffers():
                    n_total += buf.numel()
                n_prune = int(n_total - weights_by_layer[name])
                if n_prune >= n_total or n_prune < 0:
                    continue
                #print('prune out', n_prune)

                for pname, param in layer.named_parameters():
                    if not needs_mask(pname):
                        continue

                    # Determine smallest indices
                    _, prune_indices = torch.topk(torch.abs(param.data.flatten()), n_prune, largest=False)

                    # Hard pruning: set the weights to 0
                    if pruning_type == 'hard':
                        param.data.view(param.data.numel())[prune_indices] = 0

                    # Soft pruning: only set the mask to 0, keep the weights
                    for bname, buf in layer.named_buffers():
                        if bname == pname + '_mask':
                            buf.view(buf.numel())[prune_indices] = 0  # Mask는 0으로 설정
            #print('pruned sparsity', self.sparsity())


    def layer_grow(self, sparsity=0.1, sparsity_distribution='erk'):
        '''
        Grow the network to the desired sparsity, following the specified
        sparsity distribution.
        The gradient magnitude is used for growing weights.

        uniform: layer sparsity = global sparsity
        er: Erdos-Renyi
        erk: Erdos-Renyi Kernel
        '''

        #print('desired sparsity', sparsity)
        with torch.no_grad():
            weights_by_layer = self._weights_by_layer(sparsity=sparsity, sparsity_distribution=sparsity_distribution)
            for name, layer in self.named_children():

                # We need to figure out how many to grow
                n_nonzero = 0
                for bname, buf in layer.named_buffers():
                    n_nonzero += buf.count_nonzero().item()
                n_grow = int(weights_by_layer[name] - n_nonzero)
                if n_grow < 0:
                    continue
                #print('grow from', n_nonzero, 'to', weights_by_layer[name])

                for pname, param in layer.named_parameters():
                    if not needs_mask(pname):
                        continue

                    # Determine largest gradient indices
                    _, grow_indices = torch.topk(torch.abs(param.grad.flatten()),
                                                 n_grow, largest=True)

                    # Write and apply mask
                    param.data.view(param.data.numel())[grow_indices] = 0
                    for bname, buf in layer.named_buffers():
                        if bname == pname + '_mask':
                            buf.view(buf.numel())[grow_indices] = 1
            #print('grown sparsity', self.sparsity())


    def prunefl_readjust(self, aggregate_gradients, layer_times, prunable_params=0.3):
        with torch.no_grad():
            importances = []
            for i, g in enumerate(aggregate_gradients):
                g.square_()
                g = g.div(layer_times[i])
                importances.append(g)

            t = 0.2
            delta = 0
            cat_grad = torch.cat([torch.flatten(g) for g in aggregate_gradients])
            cat_imp = torch.cat([torch.flatten(g) for g in importances])
            indices = torch.argsort(cat_grad, descending=True)
            n_required = (1 - prunable_params) * cat_grad.numel()
            n_grown = 0

            masks = []
            for i, g in enumerate(aggregate_gradients):
                masks.append(torch.zeros_like(g, dtype=torch.bool))

            for j, i in enumerate(indices):
                if cat_imp[i] >= delta/t or n_grown <= n_required:
                    index_within_layer = i.item()
                    for layer in range(len(layer_times)):
                        numel = aggregate_gradients[layer].numel()
                        if index_within_layer >= numel:
                            index_within_layer -= numel
                        else:
                            break

                    delta += cat_grad[i]
                    t += layer_times[layer]

                    shape = tuple(masks[layer].shape)
                    masks[layer][np.unravel_index(index_within_layer, shape)] = 1
                    n_grown += 1
                else:
                    break

            print('readj density', n_grown / cat_imp.numel())

            # set masks
            state = self.state_dict()
            i = 0
            n_differences = 0
            for name, param in state.items():
                if name.endswith('_mask'):
                    continue
                if not needs_mask(name):
                    continue

                n_differences += torch.count_nonzero(state[name + '_mask'].to('cpu') ^ masks[i].to('cpu'))
                state[name + '_mask'] = masks[i]
                i += 1

            print('mask changed percent', n_differences / cat_imp.numel())
                    
            self.load_state_dict(state)
            return n_differences/cat_imp.numel()


    def prune(self, pruning_rate=0.2):
        with torch.no_grad():
            # prune (self.pruning_rate) of the remaining weights
            parameters_to_prune = []
            layers_to_prune = (layer for _, layer in self.named_children())
            for layer in layers_to_prune:
                for name, param in layer.named_parameters():
                    if needs_mask(name):
                        parameters_to_prune.append((layer, name))

            # (actually perform pruning)
            torch_prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch_prune.L1Unstructured,
                amount=pruning_rate
            )


    def grow(self, indices):
        with torch.no_grad():
            state = self.state_dict()
            keys = list(state.keys())
            for grow_index in indices:
                mask_name = keys[grow_index[0]] + "_mask"
                state[mask_name].flatten()[grow_index[1]] = 1
            self.load_state_dict(state)

    def reset_weights(self, global_state=None, use_global_mask=False,
                      global_communication_mask=False, pruning_type='hard'):
        '''Reset weights to the given global state and apply the mask.
        - If global_state is None, then only apply the mask in the current state.
        - use_global_mask will reset the local mask to the global mask.
        - pruning_type allows for 'soft' or 'hard' pruning.
        '''

        with torch.no_grad():
            mask_changed = False
            local_state = {k: v.to(self.device) for k, v in self.state_dict().items()}  # local_state를 device로 이동

            # If no global parameters were specified, that just means we should
            # apply the local mask, so the local state should be used as the
            # parameter source.
            if global_state is None:
                param_source = local_state
            else:
                param_source = {k: v.to(self.device) for k, v in global_state.items()}  # global_state를 device로 이동

            # We may wish to apply the global parameters but use the local mask.
            # In these cases, we will use the local state as the mask source.
            if use_global_mask:
                apply_mask_source = param_source
            else:
                apply_mask_source = local_state

            # We may wish to apply the global mask to the global parameters,
            # but not overwrite the local mask with it.
            if global_communication_mask:
                copy_mask_source = local_state
            else:
                copy_mask_source = apply_mask_source

            self.communication_sparsity = self.sparsity(apply_mask_source.items())

            # Empty new state to start with.
            new_state = {}

            # Copy over the params, masking them off if needed.
            for name, param in param_source.items():
                if name.endswith('_mask'):
                    # Skip masks, since we will copy them with their corresponding
                    # layers, from the mask source.
                    continue

                new_state[name] = local_state[name]

                mask_name = name + '_mask'
                if needs_mask(name) and mask_name in apply_mask_source:

                    mask_to_apply = apply_mask_source[mask_name].to(self.device, dtype=torch.bool)
                    mask_to_copy = copy_mask_source[mask_name].to(self.device, dtype=torch.bool)
                    param = param.to(self.device)
                    gpu_param = param[mask_to_apply].to(self.device)

                    # Copy weights provided by the weight source, where the mask permits them to be copied
                    new_state[name][mask_to_apply] = gpu_param

                    if pruning_type == 'hard':
                        new_state[name][~mask_to_apply] = 0  # Set all non-masked weights to zero (hard pruning).

                    # Don't bother allocating a *new* mask if not needed
                    if mask_name in local_state:
                        new_state[mask_name] = local_state[mask_name]

                    new_state[mask_name].copy_(mask_to_copy)  # Copy mask from mask_source into this model's mask

                    if mask_name not in local_state or not torch.equal(local_state[mask_name], mask_to_copy):
                        mask_changed = True
                else:
                    # Biases and other unmasked things, move them to the correct device
                    gpu_param = param.to(self.device)
                    new_state[name].copy_(gpu_param)

                # clean up copies made to gpu
                if gpu_param.data_ptr() != param.data_ptr():
                    del gpu_param

            self.load_state_dict(new_state)

        return mask_changed
    
    def apply_hard_mask(self):
        """Apply the current mask to the network, setting masked weights to zero
        and ensuring no gradient updates are applied to masked weights."""
        
        masked_weights = {}
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                mask_name = name + '_mask'
                if mask_name in self.state_dict():
                    mask = self.state_dict()[mask_name].to(self.device)
                    param = param.to(self.device)
                    
                    masked_weights[name] = param.data[~mask].clone()

                    param.data[~mask] = 0

                    param.requires_grad = True
                    param.register_hook(lambda grad, mask=mask: grad * mask.float())
                    
        return masked_weights


    def proximal_loss(self, last_state):

        loss = torch.tensor(0.).to(self.device)

        state = self.state_dict()
        for i, (name, param) in enumerate(state.items()):
            if name.endswith('_mask'):
                continue
            gpu_param = last_state[name].to(self.device)
            loss += torch.sum(torch.square(param - gpu_param))
            if gpu_param.data_ptr != last_state[name].data_ptr:
                del gpu_param

        return loss


    def topk_changes(self, last_state, count=5, mask_behavior='invert'):
        '''Find the top `count` changed indices and their values
        since the given last_state.
        - mask_behavior determines how the layer mask is used:
          'normal' means to take the top-k which are masked 1 (masked in)
          'invert' means to take the top-k which are masked 0 (masked out)
          'all' means to ignore the mask

        returns (values, final_indices) tuple. Where values has zeroes,
        we were only able to find top k0 < k.
        '''

        with torch.no_grad():
            state = self.state_dict()
            topk_values = torch.zeros(len(state), count)
            topk_indices = torch.zeros_like(topk_values)

            for i, (name, param) in enumerate(state.items()):
                if name.endswith('_mask') or not needs_mask(name):
                    continue
                mask_name = name + '_mask'
                haystack = param - last_state[name]
                if mask_name in state and mask_behavior != 'all':
                    mask = state[mask_name]
                    if mask_behavior == 'invert':
                        mask = 1 - mask
                    haystack *= mask

                haystack = haystack.flatten()
                layer_count = min((count, haystack.numel()))
                vals, idxs = torch.topk(torch.abs(haystack), k=layer_count, largest=True, sorted=False)
                topk_indices[i, :layer_count] = idxs
                topk_values[i, :layer_count] = haystack[idxs]

            # Get the top-k collected
            vals, idxs = torch.topk(torch.abs(topk_values).flatten(), k=count, largest=True, sorted=False)
            vals = topk_values.flatten()[idxs]
            final_indices = torch.zeros(count, 2)
            final_indices[:, 0] = idxs // count # which parameters do they belong to?
            final_indices[:, 1] = topk_indices.flatten()[idxs] # indices within

        return vals, final_indices


    def sparsity(self, buffers=None):

        if buffers is None:
            buffers = self.named_buffers()

        n_ones = 0
        mask_size = 0
        for name, buf in buffers:
            if name.endswith('mask'):
                n_ones += torch.sum(buf)
                mask_size += buf.nelement()

        return 1 - (n_ones / mask_size).item()


#############################
# Subclasses of PrunableNet #
#############################


class MNISTNet(PrunableNet):
    def __init__(self, in_channels=1, out_features=50, hidden_size=[10, 20], wh_size=16, num_classes=10):
        super(MNISTNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_size[0], 5)
        self.conv2 = nn.Conv2d(hidden_size[0], hidden_size[1], 5)

        self.fc1 = nn.Linear(hidden_size[1] * wh_size * wh_size, out_features)
        self.fc2 = nn.Linear(out_features, num_classes)

        self.init_param_sizes()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        feature = x

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        logits = F.softmax(self.fc2(x), dim=1)

        return feature, logits


class CIFAR10Net(PrunableNet):
    def __init__(self, in_channels=3, out_features=120, hidden_size=[6, 16], wh_size=20, num_classes=10):
        super(CIFAR10Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_size[0], 5)
        self.conv2 = nn.Conv2d(hidden_size[0], hidden_size[1], 5)

        self.fc1 = nn.Linear(hidden_size[1] * wh_size * wh_size, out_features)
        self.fc2 = nn.Linear(out_features, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.init_param_sizes()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        feature = x

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = F.softmax(self.fc3(x), dim=1)

        return feature, logits


class CIFAR100Net(PrunableNet):
    def __init__(self, in_channels=3, out_features=120, hidden_size=[6, 16], wh_size=20, num_classes=100):
        super(CIFAR100Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_size[0], 5)
        self.conv2 = nn.Conv2d(hidden_size[0], hidden_size[1], 5)

        self.fc1 = nn.Linear(hidden_size[1] * wh_size * wh_size, out_features)
        self.fc2 = nn.Linear(out_features, num_classes)

        self.init_param_sizes()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        feature = x

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        logits = F.softmax(self.fc2(x), dim=1)

        return feature, logits


class Conv2(PrunableNet):
    def __init__(self, in_channels=1, out_features=2048, hidden_size=[32, 64], wh_size=7, num_classes=62):
        super(Conv2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_size[0], 5, padding=2)
        self.conv2 = nn.Conv2d(hidden_size[0], hidden_size[1], 5, padding=2)

        self.fc1 = nn.Linear(hidden_size[1] * wh_size * wh_size, out_features)
        self.fc2 = nn.Linear(out_features, num_classes)

        self.init_param_sizes()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))
        feature = x

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        logits = F.softmax(self.fc2(x), dim=1)

        return feature, logits


class CoLearner(PrunableNet):
    def __init__(self, in_channels, out_features, hidden_size, wh_size):
        super(CoLearner, self).__init__()

        # Co-learner
        self.conv1 = nn.Conv2d(in_channels=hidden_size[1], out_channels=hidden_size[1], kernel_size=5)
        self.conv2 = nn.Conv2d(hidden_size[1], hidden_size[1], 5)
        
        # To determine the input size for fc1, pass a dummy input through conv layers
        dummy_input = torch.zeros(1, hidden_size[1], wh_size, wh_size)
        x = F.relu(F.max_pool2d(self.conv1(dummy_input), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        flattened_size = x.numel()

        self.fc1 = nn.Linear(flattened_size, out_features)

    def forward(self, inputs):
        # print(inputs.shape)
        x = F.relu(F.max_pool2d(self.conv1(inputs), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        # print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        logits = F.softmax(self.fc1(x), dim=1)

        # torch.Size([32, 16, 20, 20])
        # torch.Size([32, 16, 8, 8])
        # torch.Size([32, 1024])
        return logits


all_models = {
        'mnist': MNISTNet,
        'emnist': Conv2,
        'cifar10': CIFAR10Net,
        'cifar100': CIFAR100Net
}

