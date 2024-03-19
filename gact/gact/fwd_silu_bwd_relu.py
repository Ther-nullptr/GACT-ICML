import torch

class ForwardSiLUBackwardReLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        mask = x < 0
        ctx.save_for_backward(mask)
        if beta < 10:
            return x / (1 + torch.exp(-beta * x))
        else:
            return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[mask] = 0
        return grad_input, None
    

class ForwardSiLUBackwardReLU(torch.nn.Module):
    def __init__(self, beta = 1, delta_beta = 0.):
        super(ForwardSiLUBackwardReLU, self).__init__()
        self.beta = beta
        self.delta_beta = delta_beta

    def forward(self, x):
        result = ForwardSiLUBackwardReLUFunc.apply(
            x,
            self.beta,
        )
        self.beta += self.delta_beta
        return result