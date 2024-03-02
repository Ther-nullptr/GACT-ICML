import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=None):
        # 计算 softmax 的前向传播
        softmax_output = F.softmax(input, dim=dim)
        ctx.save_for_backward(softmax_output)
        ctx.dim = dim
        return softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        # 计算 softmax 的反向传播
        softmax_output, = ctx.saved_tensors
        dim = ctx.dim

        if dim is None:
            dim = softmax_output.dim() - 1

        grad_input = grad_output * (softmax_output - (grad_output * softmax_output).sum(dim=dim, keepdim=True))
        return grad_input, None

class Softmax(nn.Module):
    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return SoftmaxFunction.apply(input, self.dim)

# 使用Softmax激活函数
softmax = Softmax(dim=-1)  # 选择在维度1上进行softmax
input_data = torch.randn(3, 5, requires_grad=True)
output = softmax(input_data)

# 计算梯度
output.backward(torch.ones_like(output))

# 输出前向传播结果和梯度
print("Forward:", output)
print("Gradient:", input_data.grad)


# 使用官方的softmax激活函数
output = F.softmax(input_data, dim=-1)

# 计算梯度
output.backward(torch.ones_like(output))

# 输出前向传播结果和梯度
print("Forward:", output)
print("Gradient:", input_data.grad)
