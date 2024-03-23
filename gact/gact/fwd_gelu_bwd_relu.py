import torch

class ForwardGeLUBackwardReLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        mask_1 = (x < 0)
        ctx.save_for_backward(mask_1)
        if beta < 5.9: # 10 / 1.702
            return x / (1 + torch.exp(-beta * x * 1.702))
        else:
            return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        mask_1, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[mask_1] = 0
        return grad_input, None
    

class ForwardGeLUBackwardReLU(torch.nn.Module):
    def __init__(self, beta = 1, delta_beta = 0):
        super(ForwardGeLUBackwardReLU, self).__init__()
        self.beta = beta
        self.delta_beta = delta_beta
        self.switched = False

    def forward(self, x):
        if self.extract_mode:
            torch.save(input[0], f"output/{self.name}")

        result = ForwardGeLUBackwardReLUFunc.apply(
            x,
            self.beta
        )
        self.beta += self.delta_beta
        if self.beta > 5.9 and not self.switched:
            print('switch to ReLU')
            self.switched = True
        return result