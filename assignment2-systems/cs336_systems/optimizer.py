import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Dict, Type, Any



class SharedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        # create a index for each params
        self.all_params = list(params)
        self.param2idx = {id(p): i for (i, p) in enumerate(self.all_params)}

        # setup distributed rank
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        # Each rank is responsible for a shard of params
        self.param_shard = self.all_params[self.rank::self.world_size]

        # local optimizer only matain the shard of params
        self.local_optimizer = optimizer_cls(self.param_shard, **kwargs)

        # initialize the parent class
        super().__init__(self.local_optimizer.param_groups, self.local_optimizer.defaults)
    
        
    @torch.no_grad()
    def step(self, closure=None):
        # local optimizer takes a step
        loss = self.local_optimizer.step(closure)

        # Synchronize updated params across all ranks
        if self.world_size > 1:
            for p in self.all_params:
                idx = self.param2idx[id(p)]
                owner_rank = idx % self.world_size
                dist.broadcast(p.data, src=owner_rank)
        
        return loss
