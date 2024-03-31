import torch

class ForwardGeLUBackwardReLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        mask_1 = (x < 0.5)
        mask_2 = ((x > -2) & (x < -1))
        mask_3 = ((x > -1) & (x < 0))
        ctx.save_for_backward(mask_1, mask_2, mask_3)
        return x / (1 + torch.exp(-beta * x * 1.702))

    @staticmethod
    def backward(ctx, grad_output):
        mask_1, mask_2, mask_3 = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[mask_1] = 0
        # grad_input[mask_2] *= -0.1134
        # grad_input[mask_3] *= 0.1588
        return grad_input, None
    

class ForwardGeLUBackwardReLU(torch.nn.Module):
    def __init__(self, beta = 1, delta_beta = 0):
        super(ForwardGeLUBackwardReLU, self).__init__()
        self.beta = beta
        self.delta_beta = delta_beta
        self.switched = False

    def forward(self, x):

        result = ForwardGeLUBackwardReLUFunc.apply(
            x,
            self.beta
        )
        return result