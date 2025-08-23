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
        


class DDPBucketed(nn.Module):
    """
    a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.
    """
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = [] # async waiting handles (bucket)

        # broadcast the weights from rank0
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            with torch.no_grad():
                for p in self.module.parameters():
                    dist.broadcast(p.data, src=0)
        
        # register a hook for every parameter
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook)
        
        # create a reverse order of parameter
        params_reverse = list(self.module.parameters())[::-1]

        bucket_size_mb = bucket_size_mb * 1024**2 # conver to bytes

        self.bucket = []
        cur_bucket = [] # current bucket of params waiting to be added into self.bucket
        cur_bucket_size = 0
        self.param2bucketid = {} # the bucket id to which each param belongs
        
        # parameter bucketing
        for p in params_reverse:
            if p.requires_grad:
                p_size = p.numel() * p.element_size()
                if cur_bucket_size + p_size > bucket_size_mb and cur_bucket:
                    self.bucket.append(cur_bucket)
                    cur_bucket = []
                    cur_bucket_size = 0
                cur_bucket.append(p)
                cur_bucket_size += p_size
                self.param2bucketid[id(p)] = len(self.bucket)
        
        # the last bucket
        if cur_bucket:
            self.bucket.append(cur_bucket)
        
        # the number of params ready to reduce
        self.bucket_ready_cnt = [0] * len(self.bucket)


    def _hook(self, p):
        if p.grad is None or self.world_size <= 1:
            return
        
        bucket_id = self.param2bucketid[id(p)]
        bucket = self.bucket[bucket_id]
        self.bucket_ready_cnt[bucket_id] += 1

        # all the params in this bucket raedy to reduce
        if self.bucket_ready_cnt[bucket_id] == len(bucket):
            grads_to_reduce = [p.grad for p in bucket]
            flat_grads = torch._utils._flatten_dense_tensors(grads_to_reduce)

            handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, bucket, flat_grads))

    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    
    def finish_gradient_synchronization(self):
        # wait for all asynchronous communication calls
        # to be queue on GPU

        if self.world_size <= 1:
            return

        for handle, params, flat_grads in self.handles:
            handle.wait()
            flat_grads /= self.world_size # averrage

            unflattened_grads = torch._utils._unflatten_dense_tensors(flat_grads, params)
            for param, grad in zip(params, unflattened_grads):
                param.grad = grad
        
        self.handles.clear()
        self.bucket_ready_cnt = [0] * len(self.bucket)