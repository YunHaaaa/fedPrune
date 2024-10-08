import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F
import argparse
import numpy as np
import os
import sys
import time
from copy import deepcopy
import csv

from tqdm import tqdm

from datasets import get_dataset
import models
from models import all_models, needs_mask, initialize_mask

def device_list(x):
    if x == 'cpu':
        return [x]
    return [int(y) for y in x.split(',')]


parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, help='learning rate', default=0.01)
parser.add_argument('--clients', type=int, help='number of clients per round', default=20)
parser.add_argument('--rounds', type=int, help='number of global rounds', default=400)
parser.add_argument('--epochs', type=int, help='number of local epochs', default=10)
parser.add_argument('--dataset', type=str, choices=('mnist', 'emnist', 'cifar10', 'cifar100'),
                    default='mnist', help='Dataset to use')
parser.add_argument('--distribution', type=str, choices=('dirichlet', 'lotteryfl', 'iid'), default='dirichlet',
                    help='how should the dataset be distributed?')
parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter (unbalance rate) for Dirichlet distribution')
parser.add_argument('--total-clients', type=int, help='split the dataset between this many clients. Ignored for EMNIST.', default=400)
parser.add_argument('--min-samples', type=int, default=0, help='minimum number of samples required to allow a client to participate')
parser.add_argument('--samples-per-client', type=int, default=20, help='samples to allocate to each client (per class, for lotteryfl, or per client, for iid)')
parser.add_argument('--prox', type=float, default=0, help='coefficient to proximal term (i.e. in FedProx)')
parser.add_argument('--hidden-size', nargs='+', type=int, default=32, help='Number of channels for each convolutional layer (default: 64).')
parser.add_argument('--num-ways', type=int, default=10, help='Number of classes per task (N in "N-way", default: 5).')
parser.add_argument('--wh-size', type=int, default=7, help='Size of the hidden layer in the network (default: 7).')

# Pruning and regrowth options
parser.add_argument('--sparsity', type=float, default=0.1, help='sparsity from 0 to 1')
parser.add_argument('--rate-decay-method', default='cosine', choices=('constant', 'cosine'), help='annealing for readjustment ratio')
parser.add_argument('--rate-decay-end', default=None, type=int, help='round to end annealing')
parser.add_argument('--readjustment-ratio', type=float, default=0.5, help='readjust this many of the weights each time')
parser.add_argument('--pruning-begin', type=int, default=9, help='first epoch number when we should readjust')
parser.add_argument('--pruning-interval', type=int, default=10, help='epochs between readjustments')
parser.add_argument('--rounds-between-readjustments', type=int, default=10, help='rounds between readjustments')
parser.add_argument('--remember-old', default=False, action='store_true', help="remember client's old weights when aggregating missing ones")
parser.add_argument('--sparsity-distribution', default='erk', choices=('uniform', 'er', 'erk'))
parser.add_argument('--final-sparsity', type=float, default=None, help='final sparsity to grow to, from 0 to 1. default is the same as --sparsity')
parser.add_argument('--pruning-ratio', type=float, default=0.7, help='pruning ratio for each round')
parser.add_argument('--pruning-type', type=str, default='hard', choices=['hard', 'soft'], help='Pruning type: hard or soft pruning')

parser.add_argument('--batch-size', type=int, default=32,
                    help='local client batch size')
parser.add_argument('--l2', default=0.001, type=float, help='L2 regularization strength')
parser.add_argument('--momentum', default=0.9, type=float, help='Local client SGD momentum parameter')
parser.add_argument('--cache-test-set', default=False, action='store_true', help='Load test sets into memory')
parser.add_argument('--cache-test-set-gpu', default=False, action='store_true', help='Load test sets into GPU memory')
parser.add_argument('--test-batches', default=0, type=int, help='Number of minibatches to test on, or 0 for all of them')
parser.add_argument('--eval-every', default=10, type=int, help='Evaluate on test set every N rounds')
parser.add_argument('--device', default='0', type=device_list, help='Device to use for compute. Use "cpu" to force CPU. Otherwise, separate with commas to allow multi-GPU.')
parser.add_argument('--min-votes', default=0, type=int, help='Minimum votes required to keep a weight')
parser.add_argument('--no-eval', default=True, action='store_false', dest='eval')
parser.add_argument('--fp16', default=False, action='store_true', help='upload as fp16')
parser.add_argument('-o', '--outfile', default='output.log', type=argparse.FileType('a', encoding='ascii'))
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--loss-scaling', type=float, default=1.0, help='Loss scaling factor for Co-learner (default: 1.0).')

args = parser.parse_args()
devices = [torch.device(x) for x in args.device]
args.pid = os.getpid()

rng = np.random.default_rng(args.seed)

if args.rate_decay_end is None:
    args.rate_decay_end = args.rounds // 2
if args.final_sparsity is None:
    args.final_sparsity = args.sparsity

def print2(*arg, **kwargs):
    print(*arg, **kwargs, file=args.outfile)
    print(*arg, **kwargs)

def dprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

def print_csv_line(**kwargs):
    print2(','.join(str(x) for x in kwargs.values()))

    # for key, value in kwargs.items():
    #     print(f"{key}: {value}")

def nan_to_num(x, nan=0, posinf=0, neginf=0):
    x = x.clone()
    x[x != x] = nan
    x[x == -float('inf')] = neginf
    x[x == float('inf')] = posinf
    return x.clone()


def evaluate_global(clients, global_model, progress=False, n_batches=0):
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            accuracy = client.test(model=global_model)
            accuracies[client_id] = accuracy.item()
            sparsities[client_id] = client.sparsity()

    return accuracies, sparsities


def evaluate_local(clients, global_model, progress=False, n_batches=0):

    # we need to perform an update to client's weights.
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            client.reset_weights(global_state=global_model.state_dict(), use_global_mask=True)
            accuracy = client.test(model=global_model)
            accuracies[client_id] = accuracy.item()
            sparsities[client_id] = client.sparsity()

    return accuracies, sparsities


# Fetch and cache the dataset
dprint('Fetching dataset...')
cache_devices = devices

loaders = get_dataset(args.dataset, clients=args.total_clients, mode=args.distribution,
                      beta=args.beta, batch_size=args.batch_size, devices=cache_devices,
                      min_samples=args.min_samples, samples=args.samples_per_client)

class Client:

    def __init__(self, id, device, train_data, test_data, net=models.MNISTNet, co_net=models.MNISTNet,
                 local_epochs=10, learning_rate=0.01, target_sparsity=0.1):
        '''Construct a new client.

        Parameters:
        id : object
            a unique identifier for this client. For EMNIST, this should be
            the actual client ID.
        train_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us training samples.
        test_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us test samples.
            (we will use this as the validation set.)
        local_epochs : int
            the number of local epochs to train for each round

        Returns: a new client.
        '''

        self.id = id

        self.train_data, self.test_data = train_data, test_data

        self.device = device
        
        self.net = net(device=self.device).to(self.device)
        self.co_net = co_net(device=self.device).to(self.device)

        initialize_mask(self.net)
        initialize_mask(self.co_net)
        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.reset_optimizer()

        self.local_epochs = local_epochs

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.initial_global_params = None


    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=args.momentum, weight_decay=args.l2)
        self.co_optimizer = torch.optim.SGD(self.co_net.parameters(), lr=self.learning_rate, momentum=args.momentum, weight_decay=args.l2)

    def reset_weights(self, *args, **kwargs):
        return self.net.reset_weights(*args, **kwargs)

    def co_reset_weights(self, *args, **kwargs):
        return self.co_net.reset_weights(*args, **kwargs)

    def apply_hard_mask(self):
        return self.net.apply_hard_mask()

    def sparsity(self, *args, **kwargs):
        return self.net.sparsity(*args, **kwargs)


    def train_size(self):
        return sum(len(x) for x in self.train_data)

    def merge_models(self):

        '''self.net과 self.co_net의 파라미터와 마스크를 서버 집계 로직과 유사하게 병합합니다.'''
        import torch.nn.functional as F

        # 집계할 파라미터와 마스크 초기화
        aggregated_params = {}
        aggregated_masks = {}
        aggregated_params_for_mask = {}

        # 두 모델의 파라미터와 마스크를 수집하여 집계
        for model in [self.net, self.co_net]:
            state_dict = model.state_dict()
            for name, param in state_dict.items():
                if name.endswith('_mask'):
                    base_name = name[:-5]
                    if base_name not in aggregated_masks:
                        aggregated_masks[base_name] = torch.zeros_like(param, dtype=torch.float)
                    aggregated_masks[base_name] += param.float()
                else:
                    if name not in aggregated_params:
                        aggregated_params[name] = torch.zeros_like(param, dtype=torch.float)
                    aggregated_params[name] += param.float()

        # 집계된 파라미터와 마스크를 평균화
        for name in aggregated_params:
            if name in aggregated_masks:
                # 마스크가 있는 경우
                # 마스크가 1인 곳만 파라미터를 합산
                aggregated_params[name] = aggregated_params[name]  # 이미 마스크가 적용된 상태로 합산됨
                aggregated_masks[name] = aggregated_masks[name]  # 마스크 합계

                # 최소 투표 수 설정 (예: 두 모델 모두에서 마스크가 1인 경우만 유지)
                # 서버 집계 로직에서 args.min_votes를 사용했으나, 클라이언트는 두 모델이므로 min_votes=2로 설정
                min_votes = 2
                aggregated_masks[name] = F.threshold(aggregated_masks[name], min_votes, 0)
                aggregated_masks[name] = (aggregated_masks[name] >= min_votes).float()

                # 마스크를 기준으로 파라미터를 평균화
                # 마스크가 1인 곳만 평균을 계산하고, 0인 곳은 0으로 설정
                valid_mask = aggregated_masks[name] > 0
                aggregated_params[name][valid_mask] /= aggregated_masks[name][valid_mask]
                aggregated_params[name][~valid_mask] = 0  # 마스크가 0인 곳은 0으로 설정

                # NaN 방지
                aggregated_params[name] = torch.nan_to_num(aggregated_params[name],
                                                        nan=0.0, posinf=0.0, neginf=0.0)
                aggregated_masks[name] = torch.nan_to_num(aggregated_masks[name],
                                                        nan=0.0, posinf=0.0, neginf=0.0)

                # 마스크는 bool 타입으로 변환
                aggregated_masks[name] = aggregated_masks[name].bool()
            else:
                # 마스크가 없는 경우 (예: 바이어스)
                aggregated_params[name] /= 2  # 두 모델의 평균

        # 마스크도 파라미터로 추가
        for name, mask in aggregated_masks.items():
            aggregated_params[name + '_mask'] = mask

        # 집계된 파라미터를 글로벌 모델에 로드
        self.net.load_state_dict(aggregated_params, strict=False)


    def train(self, global_params=None, initial_global_params=None, sparsity=args.sparsity, pruning_ratio=args.pruning_ratio, server_round=None):
        '''Train the client network for a single round.'''

        ul_cost = 0
        dl_cost = 0
        model_merged = False  # 모델이 병합되었는지 여부를 추적
        pruning_done = False  # 모델 병합 직전에 pruning이 한 번만 적용되었는지 추적

        if global_params:
            # this is a FedAvg-like algorithm, where we need to reset
            # the client's weights every round
            mask_changed = self.reset_weights(global_state=global_params, use_global_mask=True)
            co_mask_changed = self.co_reset_weights(global_state=global_params, use_global_mask=True)
            
            # Try to reset the optimizer state.
            self.reset_optimizer()

            if mask_changed or co_mask_changed:
                dl_cost += self.net.mask_size # need to receive mask

            if not self.initial_global_params:
                self.initial_global_params = initial_global_params
                # no DL cost here: we assume that these are transmitted as a random seed
            else:
                # otherwise, there is a DL cost: we need to receive all parameters masked '1' and
                # all parameters that don't have a mask (e.g. biases in this case)
                dl_cost += (1-self.net.sparsity()) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)

        for epoch in range(self.local_epochs):

            # 병합 여부를 서버 라운드 기준으로 결정 (7라운드 동안 각각 훈련, 이후 병합)
            if (server_round - 1) % args.rounds_between_readjustments >= args.pruning_begin and not model_merged:
                
                # 병합 직전에 한 번만 pruning 적용 (net과 co_net 모두)
                if not pruning_done:
                    prune_sparsity = sparsity + (1 - sparsity) * args.readjustment_ratio

                    self.net.layer_prune(sparsity=prune_sparsity, sparsity_distribution=args.sparsity_distribution, pruning_type=args.pruning_type)
                    self.net.layer_grow(sparsity=sparsity, sparsity_distribution=args.sparsity_distribution)

                    self.co_net.layer_prune(sparsity=prune_sparsity, sparsity_distribution=args.sparsity_distribution, pruning_type=args.pruning_type)
                    self.co_net.layer_grow(sparsity=sparsity, sparsity_distribution=args.sparsity_distribution)

                    pruning_done = True  # pruning이 완료되었음을 기록

                self.merge_models()
                model_merged = True
                # 병합된 후에는 co_net의 옵티마이저가 더 이상 필요하지 않음
                del self.co_optimizer

            # 병합된 후에는 하나의 모델만 훈련
            if model_merged:
                self.net.train()
                for inputs, labels in self.train_data:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    if args.prox > 0:
                        loss += args.prox / 2. * self.net.proximal_loss(global_params)
                    loss.backward()
                    self.optimizer.step()
                    self.reset_weights()
                continue

            # 병합 전에는 두 모델을 병렬로 훈련
            self.net.train()
            self.co_net.train()
            for inputs, labels in self.train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                self.co_optimizer.zero_grad()

                outputs = self.net(inputs)
                co_outputs = self.co_net(inputs)

                loss = self.criterion(outputs, labels)
                co_loss = self.criterion(co_outputs, labels)

                if args.prox > 0:
                    loss += args.prox / 2. * self.net.proximal_loss(global_params)
                    co_loss += args.prox / 2. * self.co_net.proximal_loss(global_params)

                total_loss = loss + co_loss
                total_loss.backward()

                self.optimizer.step()
                self.co_optimizer.step()

                # 마스크 적용
                self.reset_weights()
                self.co_reset_weights()

        # we only need to transmit the masked weights and all biases
        if args.fp16:
            ul_cost += (1-self.net.sparsity()) * self.net.mask_size * 16 + (self.net.param_size - self.net.mask_size * 16)
        else:
            ul_cost += (1-self.net.sparsity()) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)

        ret = dict(state=self.net.state_dict(), dl_cost=dl_cost, ul_cost=ul_cost)

        # dprint(global_params['conv1.weight_mask'][0, 0, 0], '->', self.net.state_dict()['conv1.weight_mask'][0, 0, 0])
        # dprint(global_params['conv1.weight'][0, 0, 0], '->', self.net.state_dict()['conv1.weight'][0, 0, 0])
        return ret

    def test(self, model=None, n_batches=0):
        '''Evaluate the local model on the local test set.

        model - model to evaluate, or this client's model if None
        n_batches - number of minibatches to test on, or 0 for all of them
        '''
        correct = 0.
        total = 0.

        if model is None:
            model = self.net
            _model = self.net
        else:
            _model = model.to(self.device)

        _model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_data):
                if i > n_batches and n_batches > 0:
                    break
                if not args.cache_test_set_gpu:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                outputs = _model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)

        # remove copies if needed
        if model is not _model:
            del _model

        return correct / total


# initialize clients
dprint('Initializing clients...')

clients = {}
client_ids = []

accuracy_history = []
download_cost_history = []
upload_cost_history = []

for i, (client_id, client_loaders) in tqdm(enumerate(loaders.items())):
    cl = Client(client_id, *client_loaders, local_epochs=args.epochs,
                learning_rate=args.eta, target_sparsity=args.sparsity, net=all_models[args.dataset], co_net=all_models[args.dataset])

    clients[client_id] = cl
    client_ids.append(client_id)
    torch.cuda.empty_cache()

# initialize global model
global_model = all_models[args.dataset](device='cpu')
initialize_mask(global_model)

global_model.layer_prune(sparsity=args.sparsity, sparsity_distribution=args.sparsity_distribution, pruning_type=args.pruning_type)

initial_global_params = deepcopy(global_model.state_dict())

# we need to accumulate compute/DL/UL costs regardless of round number, resetting only
# when we actually report these numbers
compute_times = np.zeros(len(clients)) # time in seconds taken on client-side for round
download_cost = np.zeros(len(clients))
upload_cost = np.zeros(len(clients))

# for each round t = 1, 2, ... do
for server_round in tqdm(range(args.rounds)):

    # sample clients
    client_indices = rng.choice(list(clients.keys()), size=args.clients)

    global_params = global_model.state_dict()
    aggregated_params = {}
    aggregated_params_for_mask = {}
    aggregated_masks = {}
    # set server parameters to 0 in preparation for aggregation,
    for name, param in global_params.items():
        if name.endswith('_mask'):
            continue
        aggregated_params[name] = torch.zeros_like(param, dtype=torch.float, device='cpu')
        aggregated_params_for_mask[name] = torch.zeros_like(param, dtype=torch.float, device='cpu')
        if needs_mask(name):
            aggregated_masks[name] = torch.zeros_like(param, device='cpu')

    # for each client k \in S_t in parallel do
    total_sampled = 0
    for client_id in client_indices:
        client = clients[client_id]
        i = client_ids.index(client_id)

        # Local client training.
        t0 = time.process_time()
        
        if args.rate_decay_method == 'cosine':
            readjustment_ratio = args.readjustment_ratio * global_model._decay(server_round, alpha=args.readjustment_ratio, t_end=args.rate_decay_end)
        else:
            readjustment_ratio = args.readjustment_ratio

        readjust = (server_round - 1) % args.rounds_between_readjustments == 0 and readjustment_ratio > 0.
        if readjust:
            dprint('readjusting', readjustment_ratio)

        # determine sparsity desired at the end of this round
        # ...via linear interpolation
        if server_round <= args.rate_decay_end:
            round_sparsity = args.sparsity * (args.rate_decay_end - server_round) / args.rate_decay_end + args.final_sparsity * server_round / args.rate_decay_end
        else:
            round_sparsity = args.final_sparsity

        # actually perform training
        train_result = client.train(global_params=global_params, initial_global_params=initial_global_params)
        cl_params = train_result['state']
        download_cost[i] = train_result['dl_cost']
        upload_cost[i] = train_result['ul_cost']
            
        t1 = time.process_time()
        compute_times[i] = t1 - t0
        client.net.clear_gradients() # to save memory

        # add this client's params to the aggregate

        cl_weight_params = {}
        cl_mask_params = {}

        # first deduce masks for the received weights
        for name, cl_param in cl_params.items():
            if name.endswith('_orig'):
                name = name[:-5]
            elif name.endswith('_mask'):
                name = name[:-5]
                cl_mask_params[name] = cl_param.to(device='cpu', copy=True)
                continue

            cl_weight_params[name] = cl_param.to(device='cpu', copy=True)
            if args.fp16:
                cl_weight_params[name] = cl_weight_params[name].to(torch.bfloat16).to(torch.float)

        # at this point, we have weights and masks (possibly all-ones)
        # for this client. we will proceed by applying the mask and adding
        # the masked received weights to the aggregate, and adding the mask
        # to the aggregate as well.
        for name, cl_param in cl_weight_params.items():
            if name in cl_mask_params:
                # things like weights have masks
                cl_mask = cl_mask_params[name]
                sv_mask = global_params[name + '_mask'].to('cpu', copy=True)

                # calculate Hamming distance of masks for debugging
                if readjust:
                    dprint(f'{client.id} {name} d_h=', torch.sum(cl_mask ^ sv_mask).item())

                aggregated_params[name].add_(client.train_size() * cl_param * cl_mask)
                aggregated_params_for_mask[name].add_(client.train_size() * cl_param * cl_mask)
                aggregated_masks[name].add_(client.train_size() * cl_mask)
                if args.remember_old:
                    sv_mask[cl_mask] = 0
                    sv_param = global_params[name].to('cpu', copy=True)

                    aggregated_params_for_mask[name].add_(client.train_size() * sv_param * sv_mask)
                    aggregated_masks[name].add_(client.train_size() * sv_mask)
            else:
                # things like biases don't have masks
                aggregated_params[name].add_(client.train_size() * cl_param)

    # at this point, we have the sum of client parameters
    # in aggregated_params, and the sum of masks in aggregated_masks. We
    # can take the average now by simply dividing...
    for name, param in aggregated_params.items():

        # if this parameter has no associated mask, simply take the average.
        if name not in aggregated_masks:
            aggregated_params[name] /= sum(clients[i].train_size() for i in client_indices)
            continue

        # drop parameters with not enough votes
        aggregated_masks[name] = F.threshold_(aggregated_masks[name], args.min_votes, 0)

        # otherwise, we are taking the weighted average w.r.t. the number of 
        # samples present on each of the clients.
        aggregated_params[name] /= aggregated_masks[name]
        aggregated_params_for_mask[name] /= aggregated_masks[name]
        aggregated_masks[name] /= aggregated_masks[name]

        # it's possible that some weights were pruned by all clients. In this
        # case, we will have divided by zero. Those values have already been
        # pruned out, so the values here are only placeholders.
        aggregated_params[name] = torch.nan_to_num(aggregated_params[name],
                                                   nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_params_for_mask[name] = torch.nan_to_num(aggregated_params[name],
                                                   nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_masks[name] = torch.nan_to_num(aggregated_masks[name],
                                                  nan=0.0, posinf=0.0, neginf=0.0)

    # masks are parameters too!
    for name, mask in aggregated_masks.items():
        aggregated_params[name + '_mask'] = mask
        aggregated_params_for_mask[name + '_mask'] = mask

    # reset global params to aggregated values
    global_model.load_state_dict(aggregated_params_for_mask)

    if global_model.sparsity() < round_sparsity:
        # we now have denser networks than we started with at the beginning of
        # the round. reprune on the server to get back to the desired sparsity.
        # we use layer-wise magnitude pruning as before.
        global_model.layer_prune(sparsity=round_sparsity, sparsity_distribution=args.sparsity_distribution, pruning_type=args.pruning_type)

    # discard old weights and apply new mask
    global_params = global_model.state_dict()
    for name, mask in aggregated_masks.items():
        new_mask = global_params[name + '_mask']
        aggregated_params[name + '_mask'] = new_mask
        aggregated_params[name][~new_mask] = 0
    global_model.load_state_dict(aggregated_params)

    # evaluate performance
    torch.cuda.empty_cache()
    if server_round % args.eval_every == 0 and args.eval and server_round > 0:
        accuracies, sparsities = evaluate_global(clients, global_model, progress=True,
                                                 n_batches=args.test_batches)

        accuracy_history.append(np.mean(list(accuracies.values())))
        download_cost_history.append(sum(download_cost))
        upload_cost_history.append(sum(upload_cost))

    for client_id in clients:
        i = client_ids.index(client_id)
        if server_round % args.eval_every == 0 and args.eval and server_round > 0:
            print_csv_line(pid=args.pid,
                           dataset=args.dataset,
                           clients=args.clients,
                           total_clients=len(clients),
                           round=server_round,
                           batch_size=args.batch_size,
                           epochs=args.epochs,
                           target_sparsity=round_sparsity,
                           pruning_rate=args.readjustment_ratio,
                           initial_pruning_threshold='',
                           final_pruning_threshold='',
                           pruning_threshold_growth_method='',
                           pruning_method='',
                           lth=False,
                           client_id=client_id,
                           accuracy=accuracies[client_id],
                           sparsity=sparsities[client_id],
                           compute_time=compute_times[i],
                           download_cost=download_cost[i],
                           upload_cost=upload_cost[i])

        # if we didn't send initial global params to any clients in the first round, send them now.
        # (in the real world, this could be implemented as the transmission of
        # a random seed, so the time and place for this is not a concern to us)
        if server_round == 0:
            clients[client_id].initial_global_params = initial_global_params

    if server_round % args.eval_every == 0 and args.eval:
        # clear compute, UL, DL costs
        compute_times[:] = 0
        download_cost[:] = 0
        upload_cost[:] = 0

print2('OVERALL SUMMARY')
print2()
print2(f'{args.total_clients} clients, {args.clients} chosen each round')
print2(f'E={args.epochs} local epochs per round, B={args.batch_size} mini-batch size')
print2(f'{args.rounds} rounds of federated learning')
# print2(f'Target sparsity r_target={args.target_sparsity}, pruning rate (per round) r_p={args.pruning_rate}')
# print2(f'Accuracy threshold starts at {args.pruning_threshold} and ends at {args.final_pruning_threshold}')
# print2(f'Accuracy threshold growth method "{args.pruning_threshold_growth_method}"')
# print2(f'Pruning method: {args.pruning_method}, resetting weights: {args.reset_weights}')
print2()

accuracies = list(accuracies.values())
sparsities = list(sparsities.values())
print2(f'ACCURACY: mean={np.mean(accuracies)}, std={np.std(accuracies)}, min={np.min(accuracies)}, max={np.max(accuracies)}')
print2(f'SPARSITY: mean={np.mean(sparsities)}, std={np.std(sparsities)}, min={np.min(sparsities)}, max={np.max(sparsities)}')
print2()
print2()

filename = f"{args.outfile}.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(['Round', 'Accuracy (%)', 'Download Cost', 'Upload Cost'])
    
    for i in range(1, len(accuracy_history)):
        round_num = i * args.eval_every
        accuracy = accuracy_history[i] * 100  # 퍼센트로 변환
        download_cost = download_cost_history[i]
        upload_cost = upload_cost_history[i]
        
        writer.writerow([round_num, accuracy, download_cost, upload_cost])

print2('Training history saved to', filename)

print2('Training Complete')
