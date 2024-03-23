import math
import torch
import typing
import bitsandbytes as bnb
import bitsandbytes.functional as F

class MixedSparseTraditionalMLPFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x1: torch.Tensor, 
        w_up: torch.Tensor, b_up: torch.Tensor, w_up_state: typing.Tuple, w_up_lora_a: torch.Tensor, w_up_lora_b: torch.Tensor, 
        w_down: torch.Tensor, b_down: torch.Tensor, w_down_state: typing.Tuple, w_down_lora_a: torch.Tensor, w_down_lora_b: torch.Tensor, 
        activation_forward: str, sparsity_ratio: float, maintain_channels: int
    ):
        # forward process: up_proj
        w_up_dequant = F.dequantize_4bit(w_up, w_up_state).to(x1.dtype).t()
        y1_main = x1 @ w_up_dequant + b_up if b_up is not None else x1 @ w_up_dequant
        y1_lora_a = x1 @ w_up_lora_a
        y1_lora = y1_lora_a @ w_up_lora_b
        y1 = y1_main + y1_lora

        # apply activation function
        if activation_forward == 'relu':
            x2 = torch.relu(y1)
        elif activation_forward == 'silu':
            x2 = torch.silu(y1)
        elif activation_forward == 'gelu':
            x2 = torch.gelu(y1)
        
        # forward process: down_proj
        w_down_dequant = F.dequantize_4bit(w_down, w_down_state).to(x1.dtype).t()
        y2_main = x2 @ w_down_dequant + b_down if b_down is not None else x2 @ w_down_dequant
        y2_lora_a = x2 @ w_down_lora_a
        y2_lora = y2_lora_a @ w_down_lora_b
        y2 = y2_main + y2_lora

        # save: x1, y1_lora_a, y1(for soft activations), mask(for hard activations), x2, y2_lora_a
        mask = y1 < 0
        if activation_forward != 'relu':
            x2 = torch.relu(x2) # cache the sparse version of x2

        #! notice that: the pruning of x2 etc. is not urgent, we can implement it in other place
        # for x2: delete useless channels(mostly 0)
        zero_counts_per_channel = (x2 == 0).sum(dim=-2) # [bs, seq_len, hidden_dim] -> [bs, hidden_dim]
        actual_maintain_channel = min(int(sparsity_ratio * x2.size(-1)), maintain_channels)
        # record the top sparsity_ratio channels
        _, topk_indices = zero_counts_per_channel.topk(actual_maintain_channel, dim=-1, largest=False)

        # delete the sparse channels, and also delete the corresponding x2 channels
        x2_save = torch.zeros((*x2.shape[:-1], actual_maintain_channel), dtype=x2.dtype)

        for i in range(len(topk_indices)):
            batch_idx = i
            col_indices = topk_indices[i]
            x2_save[i] = x2[batch_idx, :, col_indices]

        ctx.save_for_backward(x1, y1_lora_a, mask, x2_save, y2_lora_a, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b)
        ctx.quant_state = w_up_state, w_down_state
        ctx.topk_indices = topk_indices
        ctx.x2_shape = x2.shape
        ctx.x2_device = x2.device

        return y2

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        topk_indices = ctx.topk_indices
        w_up_state, w_down_state = ctx.quant_state
        x1, y1_lora_a, mask, x2_save, y2_lora_a, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b = ctx.saved_tensors

        # convert the x2 to the original shape
        x2 = torch.zeros(ctx.x2_shape, device=ctx.x2_device, dtype=x2_save.dtype)

        for i in range(x2_save.shape[0]):
            x2[i, :, topk_indices[i]] = x2_save[i]

        # down proj part
        # d L / d w_down_lora_a = x2.T @ d L / d y2 @ w_down_lora_b.T
        # TODO: x2 maybe sparse
        grad_w_down_lora_a = x2.T @ (grad_output @ w_down_lora_b.T)
        # d L / d w_down_lora_b = y2_lora_a.T @ d L / d y2
        grad_w_down_lora_b = y2_lora_a.T @ grad_output
        # d L / d x2 = d L / d y2 @ w_down.T + d L / d y2 @ w_down_lora_b.T @ w_down_lora_a.T
        w_down_dequant = F.dequantize_4bit(w_down, w_down_state).to(grad_output.dtype).t()
        grad_x2 = grad_output @ w_down_dequant.T + grad_output @ w_down_lora_b.T @ w_down_lora_a.T

        # activation part
        grad_y1 = grad_x2.clone()
        grad_y1[mask] = 0

        # up proj part
        # d L / d w_up_lora_a = x1.T @ d L / d y1 @ w_up_lora_b.T
        grad_w_up_lora_a = x1.T @ (grad_y1 @ w_up_lora_b.T)
        # d L / d w_up_lora_b = y1_lora_a.T @ d L / d y1
        grad_w_up_lora_b = y1_lora_a.T @ grad_y1
        # d L / d x1 = d L / d y1 @ w_up.T + d L / d y1 @ w_up_lora_b.T @ w_up_lora_a.T
        w_up_dequant = F.dequantize_4bit(w_up, w_up_state).to(grad_output.dtype).t()
        grad_x1 = grad_y1 @ w_up_dequant.T + grad_y1 @ w_up_lora_b.T @ w_up_lora_a.T

        # TODO: add bias support 
        return grad_x1, None, None, None, grad_w_up_lora_a, grad_w_up_lora_b, grad_x2, None, None, None, grad_w_down_lora_a, grad_w_down_lora_b, None, None, None


class MixedSparseTraditionalMLP(torch.nn.Module):
    def __init__(self, activation_forward='relu', sparsity_ratio=0.5, maintain_channels=10):
        super(MixedSparseTraditionalMLP, self).__init__()
        self.sparsity_ratio = sparsity_ratio
        self.maintain_channels = maintain_channels

        # activation function method. Now support: ReLU, SiLU, GELU. Notice that default activation_backward is relu
        self.activation_forward = activation_forward

    def forward(
        self, input: torch.Tensor,
        up_proj_base: bnb.nn.modules.Linear4bit, up_proj_lora_a: torch.nn.Linear, up_proj_lora_b: torch.nn.Linear,
        down_proj_base: bnb.nn.modules.Linear4bit, down_proj_lora_a: torch.nn.Linear, down_proj_lora_b: torch.nn.Linear
    ):
        #! Notice we use equation y = xW + b; instead of default y = xW^T + b
        return MixedSparseTraditionalMLPFunc.apply(
            input,
            up_proj_base.weight,
            up_proj_base.bias,
            up_proj_base.weight.quant_state,
            up_proj_lora_a.weight.T,
            up_proj_lora_b.weight.T,
            ############################
            down_proj_base.weight,
            down_proj_base.bias,
            down_proj_base.weight.quant_state,
            down_proj_lora_a.weight.T,
            down_proj_lora_b.weight.T,
            ############################
            self.activation_forward,
            self.sparsity_ratio,
            self.maintain_channels
        )