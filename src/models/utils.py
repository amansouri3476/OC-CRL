import numpy as np
import torch
from torch import nn
import collections.abc

def set_seed(args):
    np.random.seed(args.seed)

    # This makes sure that the seed is used for random initialization of nn modules provided by nn init
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed(SEED)
# Shouldn't we use this last one?
#     torch.backends.cudnn.deterministic = True


# Don't worry about randomization and seed here. It's taken care of by set_seed above, and pl seed_everything
def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def update(d, u):
    """Performs a multilevel overriding of the values in dictionary d with the values of dictionary u"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d