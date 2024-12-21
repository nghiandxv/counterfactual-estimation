# Source: https://github.com/tadeephuy/GradientReversal
import torch.nn as nn
import torch
import torch.autograd as autograd


class GradientReversalFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


revgrad = GradientReversalFunction.apply


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
