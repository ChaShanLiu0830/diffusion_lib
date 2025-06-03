import torch



def repeat(src: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast src (shape [B] or []) to like.shape."""
    return src.view(-1, *([1] * (like.ndim - 1)))
