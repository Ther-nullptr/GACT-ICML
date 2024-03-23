import torch
import torch.nn.functional as F

# just a interface to insert some operations
class EfficientMemoryReLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = F.relu(x)
        mask = x > 0
        ctx.save_for_backward(mask)
        return result.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[mask] = 0
        return grad_input


class EfficientMemoryReLU(torch.nn.Module):
    def __init__(self):
        super(EfficientMemoryReLU, self).__init__()

    def forward(self, x):
        if self.extract_mode:
            torch.save(x, f"output/{self.name}.pt")
        return EfficientMemoryReLUFunc.apply(x)