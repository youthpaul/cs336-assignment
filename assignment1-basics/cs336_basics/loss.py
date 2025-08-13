import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    compute cross-entropy, with numerical stability
    
    Args:
        logits: predicted logits, (..., vocab_size)
        targets: x[i + 1] (...)
    
    Returns:
        cross-entropy loss on average
    """

    # originall shape
    input_shape = logits.shape

    # reshape to 2D, merge all batch like dimention
    if len(input_shape) > 2:
        logits = logits.reshape(-1, input_shape[-1])
        targets = targets.reshape(-1)
    
    # for numerical stability, subtract the largest value
    logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_shifted = logits - logits_max
    
    # compute log_softmax
    # logits - log(sum(exp(logits)))
    exp_logits = torch.exp(logits_shifted)
    log_sum_exp = torch.log(torch.sum(exp_logits, dim=-1, keepdim=True))
    log_softmax = logits_shifted - log_sum_exp
    
    # take the x[i + 1] for each example
    batch_indices = torch.arange(log_softmax.shape[0], device=logits.device)
    target_log_probs = log_softmax[batch_indices, targets]
    
    # cross-entropy loss = -log(p(target))，take average of all batches
    return -target_log_probs.mean()