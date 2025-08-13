import torch
from collections.abc import Iterable
import math


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    given parameters, clip their gradient up to max_l2_norm
    
    Args:
        parameters:
        max_l2_norm
        
    """

    # ignore the parameter that has no gradient
    params_with_grads = [p for p in parameters if p.grad is not None]
    
    if len(params_with_grads) == 0: # no gradient to be clip
        return
    
    # compute L2-norm of all group
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grads]), 2
    )
    
    # clipping
    if total_norm > max_l2_norm:
        # scaling factor
        clip_factor = max_l2_norm / (total_norm + 1e-6)
        
        # update in place
        for p in params_with_grads:
            p.grad.detach().mul_(clip_factor)



class AdamW(torch.optim.Optimizer):
    """
    implement the AdamW optimizer, with weight decay
    
    parameters:
        params: parameters to be optimized
        lr: learning rate
        betas: control the updates to the moment estimates
        eps: smalle value
        weight_decay: hyperparameter
    """
    def __init__(
        self, 
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=1e-2
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        """
        take a optimize step
        """

        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                # model parameters
                state = self.state[p]
                
                # initialize
                if len(state) == 0:
                    # timestamp begin from 1
                    state['step'] = 1 # t
                    state['exp_avg'] = torch.zeros_like(p.data) # m
                    state['exp_avg_sq'] = torch.zeros_like(p.data) # v
                else:
                    state['step'] += 1
                    
                # hyperparameter
                beta1, beta2 = group['betas']
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                lr = group['lr']
                eps = group['eps']
                weight_decay = group['weight_decay']
                
                # updata moment estimate
                # m <- β₁m + (1-β₁)g
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                # v <- β₂v + (1-β₂)g²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                
                # α_t <- α * √(1-β₂ᵗ) / (1-β₁ᵗ)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # update parameters
                # θ <- θ - α_t * m / (√v + ε)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value = -step_size)
                
                # weight dacay
                # θ <- θ - αλθ
                p.data.mul_(1 - lr * weight_decay)
                
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    cosine learning rate schedule with warmup
    
    Args:
        it: the current iteration
        max_learning_rate: a_max, the maximun lr
        min_learning_rate: a_min, the minimun lr
        warmup_iters: T_w
        cosine_cycle_iters: T_c
    
    Returns:
        the current leraning rate
    """

    # warm-up stage, linear incresing
    if it < warmup_iters:
        if warmup_iters == 0:
            return max_learning_rate
        # t/T_w * α_max
        return (it / warmup_iters) * max_learning_rate
    
    # cosine annealing
    elif it <= cosine_cycle_iters:
        # α_min + 0.5 * (1 + cos((t-T_w)/(T_c-T_w) * π)) * (α_max - α_min)
        cosine_decay = 0.5 * (1 + math.cos(
            math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        ))
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)
    
    # post annealing
    else:
        return min_learning_rate