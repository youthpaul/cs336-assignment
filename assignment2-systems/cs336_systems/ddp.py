import torch 
import torch.nn as nn 
import torch.distributed as dist 



class DDPModel(nn.Module):
    """
    broadcast the weights before training
    and issue communication calls for gradient averaging
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = [] # async waiting handles

        # register a hook for every parameter
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook)

        # broadcast the weights from rank0
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            with torch.no_grad():
                for p in self.module.parameters():
                    dist.broadcast(p.data, src=0)
    
    def _hook(self, p):
        if p.grad is None or self.world_size <= 1:
            return
        
        handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        # wait for all asynchronous communication calls
        # to be queue on GPU

        if self.world_size <= 1:
            return

        for handle in self.handles:
            handle.wait()
        
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad /= self.world_size # average the gradients
        
        self.handles.clear()
        

